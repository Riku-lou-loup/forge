"""Bounded local policy agents: observe, triage, retrieve, draft, verify, review.

This graph makes conditional tool choices; it does not pretend to be an LLM.
No network tools, command execution, equipment actions, or arbitrary document
instructions are exposed to the graph.
"""

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypedDict
from uuid import uuid4

import numpy as np
from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

from forge.data.datasets import FEATURES, sensor_matrix
from forge.ml.metrics import causal_alerts, starts
from forge.rag.retrieval import Evidence, Retriever
from forge.reports.incident import Citation, IncidentReport, SuggestedCheck


@dataclass(frozen=True)
class Budget:
    max_steps: int = 8
    max_tool_calls: int = 3
    max_seconds: float = 20
    max_rows: int = 20000

    def __post_init__(self):
        if (
            not 1 <= self.max_steps <= 12
            or not 1 <= self.max_tool_calls <= 4
            or not 0 < self.max_seconds <= 60
            or not 1 <= self.max_rows <= 20000
        ):
            raise ValueError("Budget is outside supported bounds.")


class State(TypedDict, total=False):
    status: str
    observation: dict
    query: str
    attempts: int
    evidence: list[Evidence]
    checks: list[SuggestedCheck]
    trace: list[dict]
    steps: int
    tool_calls: int
    report: IncidentReport


def investigate(
    frame,
    detector,
    metadata,
    recording,
    recording_sha256,
    *,
    retriever: Retriever | None,
    budget=Budget(),
):
    # Strip evaluation annotations before any agent or retrieval step can see them.
    frame = frame.loc[:, ["datetime", *FEATURES]].copy()
    sensor_matrix(frame)
    if len(frame) > budget.max_rows:
        raise ValueError("Recording exceeds the investigation row budget.")
    began = time.monotonic()

    def bounded(name, function, *, tool=False):
        def run(state):
            steps = state.get("steps", 0)
            calls = state.get("tool_calls", 0)
            if (
                steps >= budget.max_steps
                or (tool and calls >= budget.max_tool_calls)
                or time.monotonic() - began >= budget.max_seconds
            ):
                return {
                    "status": "budget_exhausted",
                    "trace": [
                        *state.get("trace", []),
                        {"node": name, "outcome": "budget_exhausted"},
                    ],
                }
            result = function(state)
            if time.monotonic() - began >= budget.max_seconds:
                result["status"] = "budget_exhausted"
            result.update(
                steps=steps + 1,
                tool_calls=calls + int(tool),
                trace=[
                    *state.get("trace", []),
                    {
                        "node": name,
                        "outcome": result.get("status", state.get("status", "ok")),
                        "elapsed_ms": round((time.monotonic() - began) * 1000, 2),
                    },
                ],
            )
            return result

        return run

    def observe(state):
        scores = detector.score(frame)
        ready = detector.readiness(frame)
        alerts = causal_alerts(
            np.where(ready, scores, -np.inf),
            frame.datetime,
            metadata["threshold"],
            **metadata["policy"],
        )
        onsets = starts(alerts, frame.datetime, metadata["policy"]["max_gap_seconds"])
        observation = {
            "rows": len(frame),
            "scored_rows": int(ready.sum()),
            "initialization_rows": int((~ready).sum()),
            "detector": detector.describe()["family"],
            "initialization_requirement": detector.describe().get(
                "initialization_requirement", "No startup reference required."
            ),
            "score_max": float(scores.max()),
            "threshold": metadata["threshold"],
            "alert_policy": metadata["policy"],
            "alerted_rows": int(alerts.sum()),
            "alert_onsets": len(onsets),
            "sampling_gaps_over_two_seconds": int(
                frame.datetime.diff().dt.total_seconds().gt(2).sum()
            ),
        }
        if not ready.any():
            return {"status": "insufficient_data", "observation": observation}
        if not len(onsets):
            return {"status": "no_alert", "observation": observation}
        # Highest-scoring persistent episode, selected without looking at labels.
        peak = int(np.flatnonzero(alerts)[np.argmax(scores[alerts])])
        start = int(onsets[onsets <= peak][-1])
        end = peak
        while (
            end + 1 < len(frame)
            and alerts[end + 1]
            and (frame.datetime.iloc[end + 1] - frame.datetime.iloc[end]).total_seconds()
            <= metadata["policy"]["max_gap_seconds"]
        ):
            end += 1
        deviations = detector.deviations(frame.iloc[[peak]])[0]
        top = np.argsort(-np.abs(deviations))[:3]
        observation.update(
            alert_start=str(frame.datetime.iloc[start]),
            alert_end=str(frame.datetime.iloc[end]),
            peak_timestamp=str(frame.datetime.iloc[peak]),
            peak_score=float(scores[peak]),
            sensor_deviations=[
                {
                    "sensor": FEATURES[i],
                    "value": float(frame.iloc[peak][FEATURES[i]]),
                    "reference_median": float(detector.center[i]),
                    "robust_deviation": float(deviations[i]),
                }
                for i in top
            ],
        )
        return {"status": "observed", "observation": observation}

    def triage(state):
        if state["status"] in {"no_alert", "insufficient_data"}:
            return {}
        query = " ".join(item["sensor"] for item in state["observation"]["sensor_deviations"])
        return {"query": query, "attempts": 0, "status": "seeking_evidence"}

    def retrieve(state):
        try:
            evidence = retriever.search(state["query"]) if retriever is not None else []
        except (OSError, ValueError, KeyError):
            evidence = []
        if not isinstance(evidence, list) or any(
            not isinstance(item, Evidence) for item in evidence
        ):
            return {"status": "grounding_failed", "evidence": [], "attempts": state["attempts"] + 1}
        return {
            "evidence": evidence,
            "attempts": state["attempts"] + 1,
            "status": "evidence_found" if evidence else "insufficient_evidence",
        }

    def refine(state):
        # One narrower query, chosen from observed measurements, never from labels.
        return {
            "query": state["observation"]["sensor_deviations"][0]["sensor"],
            "status": "seeking_evidence",
        }

    def draft(state):
        return {
            "checks": [
                SuggestedCheck(text=e.check, citation=e.citation) for e in state.get("evidence", [])
            ]
        }

    def verify(state):
        evidence = state.get("evidence", [])
        valid = bool(
            evidence and retriever is not None and all(retriever.verifies(e) for e in evidence)
        )
        allowed = {(e.citation, e.check) for e in evidence}
        valid = valid and all((c.citation, c.text) in allowed for c in state.get("checks", []))
        if not valid:
            return {"status": "grounding_failed", "checks": [], "evidence": []}
        return {"status": "needs_review"}

    def finalize(state):
        status = state.get("status", "insufficient_evidence")
        supported = status == "needs_review"
        summaries = {
            "needs_review": "A persistent anomaly alert was found. Retrieved analytical checks are ready for human review; no root cause has been established.",
            "no_alert": "No persistent alert was found at the frozen threshold. This does not establish normal or safe equipment operation.",
            "insufficient_data": "The recording contains only initialization readings. No anomaly assessment is available; provide more readings from the same recording.",
            "insufficient_evidence": "An alert was found, but the available corpus did not support an analytical check. Request applicable documentation.",
            "budget_exhausted": "The investigation stopped at its execution budget. Any partial observations require a new review; no checks are proposed.",
            "grounding_failed": "The draft failed citation verification. Unsupported guidance was removed.",
        }
        report = IncidentReport(
            report_id=uuid4().hex,
            created_at=datetime.now(UTC).isoformat(),
            recording=recording,
            recording_sha256=recording_sha256,
            model_run=metadata["run_id"],
            model_sha256=metadata["model_sha256"],
            status=status,
            summary=summaries[status],
            observation=state.get("observation", {}),
            evidence=[
                Citation(
                    citation=e.citation,
                    title=e.title,
                    excerpt=e.text,
                    source=e.source,
                    version=e.version,
                    corpus_sha256=e.corpus_sha256,
                    relevance=e.relevance,
                    authorship=e.authorship,
                )
                for e in state.get("evidence", [])
            ]
            if supported
            else [],
            suggested_checks=state.get("checks", []) if supported else [],
            limitations=[
                "Recorded laboratory data from one SKAB testbed; no deployment reliability claim.",
                "Anomaly scores are not diagnoses or calibrated physical failure probabilities.",
                "Robust sensor deviations describe a pooled reference; they are not model feature attribution.",
                detector.describe().get(
                    "initialization_requirement", "No startup reference required."
                ),
                "The corpus is project-authored review guidance, not a manufacturer manual.",
                "Operating setpoints, manufacturer limits and asset history are unavailable.",
                "Local policy agents and extractive drafting were used; no LLM was called.",
                "Execution time is checked between local tools; this is not a hard process timeout.",
            ],
            trace=[
                *state.get("trace", []),
                {
                    "node": "human_review",
                    "outcome": "awaiting_user",
                    "steps": state.get("steps", 0),
                    "tool_calls": state.get("tool_calls", 0),
                },
            ],
        )
        return {"report": report}

    graph = StateGraph(State)
    for name, function, tool in [
        ("observe", observe, True),
        ("triage", triage, False),
        ("retrieve", retrieve, True),
        ("refine", refine, False),
        ("draft", draft, False),
        ("verify", verify, False),
    ]:
        graph.add_node(name, bounded(name, function, tool=tool))
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "observe")
    graph.add_conditional_edges(
        "observe", lambda s: "finalize" if s["status"] == "budget_exhausted" else "triage"
    )
    graph.add_conditional_edges(
        "triage",
        lambda s: (
            "finalize"
            if s["status"] in {"budget_exhausted", "no_alert", "insufficient_data"}
            else "retrieve"
        ),
    )
    graph.add_conditional_edges(
        "retrieve",
        lambda s: (
            "finalize"
            if s["status"] in {"budget_exhausted", "grounding_failed"}
            else ("draft" if s.get("evidence") else "refine" if s["attempts"] < 2 else "finalize")
        ),
    )
    graph.add_conditional_edges(
        "refine", lambda s: "finalize" if s["status"] == "budget_exhausted" else "retrieve"
    )
    graph.add_conditional_edges(
        "draft", lambda s: "finalize" if s["status"] == "budget_exhausted" else "verify"
    )
    graph.add_edge("verify", "finalize")
    graph.add_edge("finalize", END)
    # Explicitly disable ambient LangSmith tracing: recorded data stays local.
    with tracing_context(enabled=False):
        return graph.compile().invoke(
            {"trace": [], "steps": 0, "tool_calls": 0}, config={"recursion_limit": 12}
        )["report"]
