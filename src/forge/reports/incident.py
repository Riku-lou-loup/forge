"""Structured investigation records and review-aware portable exports."""

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    citation: str
    title: str
    excerpt: str
    source: str
    version: str
    corpus_sha256: str
    relevance: float
    authorship: str


class SuggestedCheck(BaseModel):
    text: str
    citation: str


class IncidentReport(BaseModel):
    schema_version: int = 1
    report_id: str
    created_at: str
    recording: str
    recording_sha256: str
    model_run: str
    model_sha256: str
    mode: Literal["local_policy_and_extractive_retrieval"] = "local_policy_and_extractive_retrieval"
    status: Literal[
        "needs_review",
        "no_alert",
        "insufficient_evidence",
        "budget_exhausted",
        "grounding_failed",
        "insufficient_data",
    ]
    summary: str
    observation: dict = Field(default_factory=dict)
    evidence: list[Citation] = Field(default_factory=list)
    suggested_checks: list[SuggestedCheck] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)
    review_status: Literal["unreviewed", "reviewed"] = "unreviewed"
    reviewer: str | None = None
    reviewed_at: str | None = None

    def reviewed(self, reviewer: str):
        if self.status != "needs_review":
            raise ValueError("Only a grounded investigation can be marked reviewed.")
        if not reviewer.strip() or len(reviewer.strip()) > 100:
            raise ValueError("A reviewer name of 1–100 characters is required.")
        return self.model_copy(
            update={
                "review_status": "reviewed",
                "reviewer": reviewer.strip(),
                "reviewed_at": datetime.now(UTC).isoformat(),
            }
        )


def safe_text(value):
    # Retrieved passages are text, never executable markup or an instruction source.
    return html.escape(str(value), quote=False).replace("[", "\\[").replace("]", "\\]")


def markdown(report: IncidentReport):
    lines = [
        "# FORGE investigation",
        "",
        f"Recording: {safe_text(report.recording)}",
        "",
        f"Status: {report.status} · Review: {report.review_status}",
        "",
        safe_text(report.summary),
        "",
        "## Measured observations",
        "",
        "```json",
        json.dumps(report.observation, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Suggested analytical checks",
        "",
    ]
    lines += [
        f"- {safe_text(item.text)} ({safe_text(item.citation)})" for item in report.suggested_checks
    ] or ["No supported checks available."]
    lines += ["", "## Retrieved evidence", ""]
    for item in report.evidence:
        lines += [
            f"### {safe_text(item.citation)} — {safe_text(item.title)}",
            "",
            safe_text(item.excerpt),
            "",
            f"Source: {safe_text(item.source)} · {safe_text(item.authorship)} · corpus SHA-256 `{item.corpus_sha256}`",
            "",
        ]
    lines += [
        "## Limits",
        "",
        *[f"- {safe_text(limit)}" for limit in report.limitations],
        "",
        "## Provenance",
        "",
        f"Report: `{report.report_id}` · created {report.created_at}",
        "",
        f"Model run: `{report.model_run}` · SHA-256 `{report.model_sha256}`",
        "",
        f"Recording SHA-256: `{report.recording_sha256}`",
        "",
        f"Mode: `{report.mode}`. No language-model API call was made.",
        "",
    ]
    if report.reviewer:
        lines += [
            f"Reviewed by {safe_text(report.reviewer)} at {report.reviewed_at}. Review is not permission to operate equipment.",
            "",
        ]
    lines += ["## Workflow trace", "", "```json", json.dumps(report.trace, indent=2), "```", ""]
    return "\n".join(lines)


def export(report: IncidentReport, directory: Path):
    """Write a draft or reviewed report, retaining status in both formats."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = report.report_id + ("-reviewed" if report.review_status == "reviewed" else "-draft")
    if not all(c.isalnum() or c == "-" for c in stem):
        raise ValueError("Invalid report identifier.")
    json_path, md_path = directory / f"{stem}.json", directory / f"{stem}.md"
    if json_path.exists() or md_path.exists():
        raise FileExistsError("Report export already exists; preserve the previous record.")
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    return json_path, md_path
