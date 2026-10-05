"""Offline checks for model-selected calls, grounded fields and local transport."""

import copy
import json
from datetime import datetime, timedelta

import pytest
from test_agent_tools import toolkit as toolkit

from forge.agents.local_agent import investigate_local, markdown_report
from forge.agents.local_demo import main
from forge.agents.ollama_client import OllamaClient, OllamaError
from forge.agents.tools import InspectRecordingArgs, SearchEvidenceArgs


def message(calls=None, content=""):
    result = {"role": "assistant", "content": content}
    if calls is not None:
        result["tool_calls"] = calls
    return {"message": result, "done": True, "done_reason": "stop"}


def call(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


class ScriptedModel:
    """A fake provider, not a live model quality test."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def identity(self):
        return {"model": "test-model", "digest": "test-digest", "ollama_version": "test"}

    def chat(self, messages, **kwargs):
        self.requests.append((copy.deepcopy(messages), kwargs))
        return next(self.responses)


def draft_for(tools):
    inspection = tools.inspect_recording(InspectRecordingArgs(recording_id="valve1/1"))
    passage = tools.search_evidence(SearchEvidenceArgs(query="flow pressure")).passages[0]
    return {
        "recording_id": inspection.recording_id,
        "status": inspection.status,
        "alerted_rows": inspection.alerted_rows,
        "unavailable_rows": inspection.unavailable_rows,
        "explanation": "The detector raised alerts. The cause remains unconfirmed.",
        "checks": [{"citation": passage.citation, "check": passage.check}],
    }


def standard_model(draft):
    return ScriptedModel(
        [
            message([call("inspect_recording", recording_id="valve1/1")]),
            message([call("search_evidence", query="flow pressure")]),
            message(content="Ready."),
            message(content=json.dumps(draft)),
        ]
    )


def run(model, tools, **kwargs):
    return investigate_local(
        "Investigate this recording.", "valve1/1", client=model, tools=tools, enabled=True, **kwargs
    )


def test_model_selected_calls_receive_real_tool_results(toolkit):
    tools, _ = toolkit
    model = standard_model(draft_for(tools))
    result = run(model, tools)
    assert result["status"] == "draft" and result["review_required"]
    assert len(result["tool_trace"]) == 2 and len(result["model_trace"]) == 4
    assert result["tool_trace"][0]["result"]["alerted_rows"] == 6
    tool_messages = [m for m in model.requests[2][0] if m["role"] == "tool"]
    assert len(tool_messages) == 2 and "flow-context@1.0" in tool_messages[1]["content"]
    assert "schema" in model.requests[-1][1] and "tools" not in model.requests[-1][1]
    assert "unreviewed" in markdown_report(result)
    assert "Human review is required" in markdown_report(result)
    start = datetime.fromisoformat(result["investigation_started_at"])
    end = datetime.fromisoformat(result["generated_at"])
    assert start.utcoffset() == end.utcoffset() == timedelta(0)
    assert start <= end
    assert "2020-01-01T00:00:02" in markdown_report(result)
    assert result["generated_at"] in markdown_report(result)


@pytest.mark.parametrize("tamper", ["count", "citation", "check", "duplicate", "missing", "schema"])
def test_invalid_generated_claims_are_blocked(toolkit, tamper):
    tools, _ = toolkit
    draft = draft_for(tools)
    if tamper == "count":
        draft["alerted_rows"] += 1
    elif tamper == "citation":
        draft["checks"][0]["citation"] = "fabricated@1.0"
    elif tamper == "check":
        draft["checks"][0]["check"] = "Replace the pump immediately."
    elif tamper == "duplicate":
        draft["checks"] *= 2
    elif tamper == "missing":
        draft["checks"] = []
    else:
        draft["alerted_rows"] = True
    result = run(standard_model(draft), tools)
    assert result["status"] == "blocked" and result["draft"] is None
    assert draft["explanation"] not in markdown_report(result)


def test_explicit_enablement_precedes_any_provider_access(toolkit):
    with pytest.raises(ValueError, match="enablement"):
        investigate_local("Investigate", "valve1/1", client=None, tools=toolkit[0])
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


def test_wrong_recording_is_not_executed(toolkit):
    model = ScriptedModel([message([call("inspect_recording", recording_id="other/1")])])
    result = run(model, toolkit[0])
    assert result["status"] == "blocked" and result["tool_trace"] == []


def test_no_tool_use_cannot_produce_an_accepted_report(toolkit):
    result = run(ScriptedModel([message(content="It is safe.")] * 4), toolkit[0])
    assert result["status"] == "blocked" and len(result["model_trace"]) == 4


def test_unknown_tools_consume_budget_without_execution(toolkit):
    model = ScriptedModel([message([call("shell", command="ignored")])] * 3)
    result = run(model, toolkit[0])
    assert len(result["tool_trace"]) == 3
    assert all(t["error"]["code"] == "unknown_tool" for t in result["tool_trace"])
    assert result["status"] == "blocked"


def test_excessive_calls_are_rejected_before_execution(toolkit):
    result = run(
        ScriptedModel([message([call("search_evidence", query="pressure")] * 4)]), toolkit[0]
    )
    assert result["status"] == "blocked" and not result["tool_trace"]


@pytest.mark.parametrize(
    "bad",
    [[42], [{"function": []}], [{"function": {"name": "inspect_recording", "arguments": "{}"}}]],
)
def test_malformed_calls_fail_closed(toolkit, bad):
    result = run(ScriptedModel([message(bad)]), toolkit[0])
    assert result["status"] == "blocked" and result["draft"] is None


def test_retrieval_abstention_keeps_checks_empty(toolkit):
    tools, _ = toolkit
    draft = draft_for(tools)
    draft["checks"] = []
    draft["explanation"] = "Alerts were returned, but no relevant guidance was retrieved."
    model = ScriptedModel(
        [
            message([call("inspect_recording", recording_id="valve1/1")]),
            message([call("search_evidence", query="sourdough pastry")]),
            message(),
            message(content=json.dumps(draft)),
        ]
    )
    result = run(model, tools)
    assert result["status"] == "draft" and not result["draft"]["checks"]
    assert result["tool_trace"][1]["result"]["abstained"]


def test_no_alert_does_not_require_retrieval(toolkit, monkeypatch):
    tools, detector = toolkit
    import numpy as np

    monkeypatch.setattr(detector, "score", lambda frame: np.zeros(len(frame)))
    draft = draft_for(tools)
    draft["checks"] = []
    draft["explanation"] = "No alert was raised. This does not establish safe operation."
    model = ScriptedModel(
        [
            message([call("inspect_recording", recording_id="valve1/1")]),
            message(content=json.dumps(draft)),
        ]
    )
    result = run(model, tools)
    assert result["status"] == "draft" and len(result["tool_trace"]) == 1
    assert result["draft"]["status"] == "no_alert"


def test_provider_failure_keeps_trace_and_blocks_report(toolkit):
    class FailedModel(ScriptedModel):
        def chat(self, *args, **kwargs):
            if self.requests:
                raise OllamaError("Timeout")
            return super().chat(*args, **kwargs)

    model = FailedModel([message([call("inspect_recording", recording_id="valve1/1")])])
    result = run(model, toolkit[0])
    assert result["status"] == "blocked" and len(result["tool_trace"]) == 1


def test_deadline_prevents_chat(toolkit, monkeypatch):
    clock = iter([0, 601, 601])
    monkeypatch.setattr("forge.agents.local_agent.time.monotonic", lambda: next(clock))
    model = ScriptedModel([])
    result = run(model, toolkit[0])
    assert result["status"] == "blocked" and not model.requests


@pytest.mark.parametrize("model", ["qwen3:cloud", "https://remote cloud", "", "x" * 101])
def test_cloud_and_invalid_models_are_rejected(model):
    with pytest.raises(ValueError):
        OllamaClient(model)


def test_identity_requires_local_installed_tool_model(monkeypatch):
    client = OllamaClient()
    responses = iter(
        [
            {"version": "test"},
            {"models": [{"name": client.model, "digest": "abc"}]},
            {"capabilities": ["tools"], "remote_host": "https://cloud.example"},
        ]
    )
    monkeypatch.setattr(client, "_request", lambda *a, **k: next(responses))
    with pytest.raises(OllamaError, match="Remote-backed"):
        client.identity()


def test_transport_is_loopback_and_does_not_follow_redirects(monkeypatch):
    observed = {}

    class Connection:
        def __init__(self, host, port, timeout):
            observed.update(host=host, port=port)

        def request(self, *args):
            pass

        def getresponse(self):
            return self

        status = 302

        def read(self, size):
            return b"{}"

        def close(self):
            observed["closed"] = True

    monkeypatch.setattr("forge.agents.ollama_client.http.client.HTTPConnection", Connection)
    with pytest.raises(OllamaError):
        OllamaClient()._request("/api/version")
    assert observed == {"host": "127.0.0.1", "port": 11434, "closed": True}


@pytest.mark.parametrize(
    "response",
    [
        {"done": False},
        {"done": True, "done_reason": "length"},
        {"done": True, "message": {"role": "tool", "content": "x"}},
        {"done": True, "message": {"role": "assistant", "content": []}},
        {"done": True, "prompt_eval_count": 4000},
    ],
)
def test_invalid_provider_responses_are_rejected(monkeypatch, response):
    client = OllamaClient()
    monkeypatch.setattr(client, "_request", lambda *a, **k: response)
    with pytest.raises(OllamaError):
        client.chat([{"role": "user", "content": "test"}])


def test_chat_adapts_tools_and_bounds_generation(monkeypatch):
    client = OllamaClient()
    captured = {}

    def request(path, payload, **kwargs):
        captured.update(payload)
        return message(content="Done")

    monkeypatch.setattr(client, "_request", request)
    client.chat([{"role": "user", "content": "test"}], tools=[{"name": "example"}])
    assert captured["tools"] == [{"type": "function", "function": {"name": "example"}}]
    assert captured["think"] is False and captured["stream"] is False
    assert captured["options"]["num_ctx"] == 4096
    assert captured["options"]["num_predict"] == 512


def test_large_prompt_is_rejected_before_transport(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(client, "_request", lambda *a, **k: pytest.fail("Transport called"))
    with pytest.raises(OllamaError, match="character budget"):
        client.chat([{"role": "user", "content": "a" * 15000}])


def test_no_alert_rejects_alert_specific_recommendations(toolkit, monkeypatch):
    import numpy as np

    tools, detector = toolkit
    monkeypatch.setattr(detector, "score", lambda frame: np.zeros(len(frame)))
    draft = draft_for(tools)
    model = ScriptedModel(
        [
            message([call("inspect_recording", recording_id="valve1/1")]),
            message(content=json.dumps(draft)),
        ]
    )
    result = run(model, tools)
    assert result["status"] == "blocked"
    assert "without an alert" in result["failure"]
    assert len(result["tool_trace"]) == 1


def test_insufficient_data_skips_search(toolkit, monkeypatch):
    import numpy as np

    tools, detector = toolkit
    monkeypatch.setattr(detector, "readiness", lambda frame: np.zeros(len(frame), dtype=bool))
    draft = draft_for(tools)
    draft["checks"] = []
    draft["explanation"] = "No readings were available for scoring. No conclusion can be drawn."
    model = ScriptedModel(
        [
            message([call("inspect_recording", recording_id="valve1/1")]),
            message(content=json.dumps(draft)),
        ]
    )
    result = run(model, tools)
    assert result["status"] == "draft"
    assert result["draft"]["status"] == "insufficient_data"
    assert len(result["tool_trace"]) == 1 and len(model.requests) == 2


@pytest.mark.parametrize("ram,commit", [(5, 20), (20, 7)])
def test_low_windows_memory_prevents_model_loading(monkeypatch, ram, commit):
    from forge.agents import local_memory

    monkeypatch.setattr(
        local_memory,
        "windows_memory",
        lambda: {
            "available_ram_gib": ram,
            "available_commit_gib": commit,
        },
    )
    with pytest.raises(OllamaError, match="Local model not started"):
        local_memory.require_memory_headroom()


def test_unload_does_not_generate_text(monkeypatch):
    client = OllamaClient()
    observed = []
    monkeypatch.setattr(client, "_request", lambda *a, **k: observed.append((a, k)))
    client.unload()
    assert observed[0][0] == ("/api/generate", {"model": "qwen3.5:4b", "keep_alive": 0})


def test_cli_unloads_model_after_failed_investigation(monkeypatch):
    from forge.agents import local_demo

    released = []

    class LocalClient:
        def __init__(self, *a, **k):
            pass

        def unload(self):
            released.append(True)

    monkeypatch.setattr(local_demo, "require_memory_headroom", lambda: None)
    monkeypatch.setattr(local_demo, "create_client", lambda *a, **k: LocalClient())

    def fail(*args, **kwargs):
        raise OllamaError("Provider failed")

    monkeypatch.setattr(local_demo, "investigate_local", fail)
    with pytest.raises(SystemExit):
        local_demo.main(["--enable-llm"])
    assert released == [True]


def test_cli_unloads_after_success(monkeypatch):
    from forge.agents import local_demo

    released = []

    class LocalClient:
        def __init__(self, *a, **k):
            pass

        def unload(self):
            released.append(True)

    monkeypatch.setattr(local_demo, "require_memory_headroom", lambda: None)
    monkeypatch.setattr(local_demo, "create_client", lambda *a, **k: LocalClient())
    monkeypatch.setattr(local_demo, "markdown_report", lambda result: "Draft")
    monkeypatch.setattr(
        local_demo,
        "investigate_local",
        lambda *a, **k: {
            "status": "draft",
            "model": {"model": "test"},
            "elapsed_seconds": 0.1,
        },
    )
    assert local_demo.main(["--enable-llm"]) == 0
    assert released == [True]


def test_timestamped_export_keeps_source_time_separate(toolkit, monkeypatch, tmp_path):
    from forge.agents import local_demo

    tools, _ = toolkit
    result = run(standard_model(draft_for(tools)), tools)
    result["generated_at"] = "2026-10-07T21:22:06.123456+00:00"

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def unload(self):
            pass

    monkeypatch.setattr(local_demo, "require_memory_headroom", lambda: None)
    monkeypatch.setattr(local_demo, "create_client", lambda *a, **k: Client())
    monkeypatch.setattr(local_demo, "InvestigationTools", lambda **kwargs: tools)
    monkeypatch.setattr(local_demo, "investigate_local", lambda *a, **k: result)
    assert main(["--enable-llm", "--export", "--root", str(tmp_path)]) == 0
    traces = list((tmp_path / "reports/incidents").glob("*.json"))
    assert len(traces) == 1
    assert traces[0].name.startswith("local-llm-20261007T212206123456Z-valve1-1-")
    payload = json.loads(traces[0].read_text(encoding="utf-8"))
    assert payload["inspection"]["first_alert_timestamp"] == "2020-01-01T00:00:02"
    report = traces[0].with_suffix(".md").read_text(encoding="utf-8")
    assert "2026-10-07T21:22:06.123456+00:00" in report
    assert "2020-01-01T00:00:02" in report
