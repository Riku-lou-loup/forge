"""Offline tool boundaries use real scoring and retrieval."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from forge.agents import tools as agent_tools
from forge.agents.tool_runtime import ToolBudgetExceeded, ToolCall, ToolSession
from forge.config import PROJECT_ROOT
from forge.data.datasets import FEATURES
from forge.data.partitions import read_plan
from forge.ml.detectors import Detector


@pytest.fixture
def toolkit(monkeypatch):
    reference = pd.DataFrame({name: np.linspace(0, 10, 100) for name in FEATURES})
    detector = Detector.fit(reference)
    frame = reference.iloc[:8].copy()
    frame["datetime"] = pd.date_range("2020-01-01", periods=8, freq="s")
    frame["Pressure"] = 100
    frame["anomaly"] = frame["changepoint"] = 1
    original_score = detector.score

    def score(sensors):
        assert set(sensors.columns) == {"datetime", *FEATURES}
        return original_score(sensors)

    detector.score = score
    metadata = {
        "threshold": 3.0,
        "policy": {"persistence": 3, "max_gap_seconds": 2},
        "run_id": "synthetic-test",
        "model_sha256": "0" * 64,
    }
    monkeypatch.setattr(agent_tools, "load_model", lambda *a, **k: (detector, metadata, None))
    monkeypatch.setattr(
        agent_tools, "load_development_recording", lambda *a, **k: (frame, {"sha256": "1" * 64})
    )
    return agent_tools.InvestigationTools(root=PROJECT_ROOT), detector


def test_inspection_strips_annotations_and_returns_measured_findings(toolkit):
    result = toolkit[0].inspect_recording(agent_tools.InspectRecordingArgs(recording_id="valve1/1"))
    assert result.status == "alert"
    assert result.rows == result.scored_rows == 8
    assert result.alerted_rows == 6 and result.alert_onsets == 1
    assert result.first_alert_timestamp == "2020-01-01T00:00:02"
    assert result.last_alert_timestamp == "2020-01-01T00:00:07"
    assert result.recording_timezone == "Unspecified in source recording"
    assert result.sensors[0].sensor == "Pressure"
    assert result.model_run == "synthetic-test"
    assert "changepoint" not in result.model_dump_json()


def test_unavailable_scores_cannot_create_alerts(toolkit):
    tools, detector = toolkit
    detector.readiness = lambda frame: np.zeros(len(frame), dtype=bool)
    result = tools.inspect_recording(agent_tools.InspectRecordingArgs(recording_id="valve1/1"))
    assert result.status == "insufficient_data" and result.scored_rows == 0
    assert result.score_max is None and not result.sensors and result.alerted_rows == 0
    assert result.first_alert_timestamp is None and result.last_alert_timestamp is None


def test_search_uses_real_bm25_and_abstains(toolkit):
    tools, _ = toolkit
    result = tools.search_evidence(agent_tools.SearchEvidenceArgs(query="flow pressure"))
    assert result.backend == "bm25" and not result.abstained
    assert result.passages[0].citation == "flow-context@1.0"
    assert tools.search_evidence(agent_tools.SearchEvidenceArgs(query="sourdough pastry")).abstained


def test_search_rejects_forged_passages(toolkit, monkeypatch):
    tools, _ = toolkit
    passage = tools.retriever.search("flow pressure")[0]
    monkeypatch.setattr(
        tools.retriever, "search", lambda *a, **k: [replace(passage, text="forged")]
    )
    with pytest.raises(ValueError, match="verification"):
        tools.search_evidence(agent_tools.SearchEvidenceArgs(query="flow pressure"))


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": " "},
        {"query": "x" * 2001},
        {"query": 123},
        {"query": "pressure", "limit": True},
        {"query": "pressure", "limit": 4},
        {"query": "pressure", "command": "anything"},
    ],
)
def test_search_contract_rejects_invalid_arguments(arguments):
    with pytest.raises(ValidationError):
        agent_tools.SearchEvidenceArgs.model_validate(arguments)


def test_test_recording_is_rejected_before_model_loading(monkeypatch):
    _, plan = read_plan(PROJECT_ROOT)
    monkeypatch.setattr(agent_tools, "load_model", lambda *a, **k: pytest.fail("Model loaded"))
    with pytest.raises(ValueError, match="training or validation"):
        agent_tools.InvestigationTools().inspect_recording(
            agent_tools.InspectRecordingArgs(recording_id=plan["partitions"]["test"][0])
        )


def test_recording_contract_cannot_set_paths_or_model_configuration():
    for args in (
        {"recording_id": "../../secret"},
        {"recording_id": "valve1/1", "torch_artifact": "arbitrary"},
        {"recording_id": "valve1/1", "threshold": 0},
    ):
        with pytest.raises(ValidationError):
            agent_tools.InspectRecordingArgs.model_validate(args)


def test_session_dispatches_and_correlates_typed_results(toolkit):
    session = ToolSession(toolkit[0])
    result = session.execute(
        ToolCall(call_id="call-1", name="search_evidence", arguments={"query": "pressure"})
    )
    assert result.call_id == "call-1" and result.error is None
    assert result.result.kind == "evidence_search" and len(session.trace) == 1


def test_bad_calls_consume_budget_and_unknown_tools_never_execute(toolkit):
    session = ToolSession(toolkit[0], max_calls=2)
    first = session.execute(ToolCall(call_id="1", name="shell", arguments={"command": "unused"}))
    second = session.execute(ToolCall(call_id="2", name="search_evidence", arguments={"query": 4}))
    assert first.error.code == "unknown_tool" and second.error.code == "invalid_arguments"
    with pytest.raises(ToolBudgetExceeded):
        session.execute(ToolCall(call_id="3", name="search_evidence", arguments={"query": "flow"}))
    assert len(session.trace) == 2


def test_duplicate_call_id_is_not_executed_again(toolkit):
    session = ToolSession(toolkit[0])
    call = ToolCall(call_id="same", name="search_evidence", arguments={"query": "flow"})
    session.execute(call)
    assert session.execute(call).error.code == "duplicate_call_id"


def test_tool_failures_do_not_expose_internal_paths(toolkit, monkeypatch):
    def fail(args):
        raise OSError("private/path/credentials.txt")

    monkeypatch.setattr(toolkit[0], "search_evidence", fail)
    result = ToolSession(toolkit[0]).execute(
        ToolCall(call_id="1", name="search_evidence", arguments={"query": "flow"})
    )
    assert result.error.code == "tool_failed" and "credentials" not in result.model_dump_json()


def test_tool_schemas_are_closed_and_expose_only_two_capabilities():
    schemas = agent_tools.tool_definitions()
    assert {item["name"] for item in schemas} == {"inspect_recording", "search_evidence"}
    assert all(item["parameters"]["additionalProperties"] is False for item in schemas)


@pytest.mark.parametrize("budget", [0, 5, True, 1.5])
def test_session_rejects_invalid_budgets(toolkit, budget):
    with pytest.raises(ValueError, match="budget"):
        ToolSession(toolkit[0], max_calls=budget)


def test_response_limit_rejects_oversized_results(toolkit, monkeypatch):
    tools = toolkit[0]
    result = tools.search_evidence(agent_tools.SearchEvidenceArgs(query="pressure"))
    monkeypatch.setattr(
        tools, "search_evidence", lambda args: result.model_copy(update={"query": "x" * 24001})
    )
    response = ToolSession(tools).execute(
        ToolCall(call_id="1", name="search_evidence", arguments={"query": "pressure"})
    )
    assert response.error.code == "tool_failed" and response.result is None


def test_retrieval_does_not_load_a_detector(toolkit, monkeypatch):
    monkeypatch.setattr(agent_tools, "load_model", lambda *a, **k: pytest.fail("Model loaded"))
    assert toolkit[0].search_evidence(agent_tools.SearchEvidenceArgs(query="pressure")).passages


def test_schema_demo_needs_no_data_or_model(tmp_path, capsys):
    import json

    from forge.agents.tool_demo import main

    assert main(["--root", str(tmp_path), "--schemas"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2


def test_scripted_demo_reports_that_no_llm_was_called(toolkit, capsys):
    import json

    from forge.agents.tool_demo import main

    assert main(["--recording", "valve1/1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "scripted_offline_tool_demo" and not result["llm_called"]
    assert [entry["tool"] for entry in result["trace"]] == ["inspect_recording", "search_evidence"]
