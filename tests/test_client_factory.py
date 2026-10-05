"""Offline provider routing and CLI lifecycle checks."""

import pytest

from forge.agents import local_demo
from forge.agents.client_factory import create_client
from forge.agents.ollama_client import OllamaClient, OllamaError


@pytest.fixture(autouse=True)
def forbid_model_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Provider tests must not contact a model server")

    monkeypatch.setattr("forge.agents.ollama_client.http.client.HTTPConnection", forbidden)


@pytest.mark.parametrize(
    "backend,port,expected",
    [
        ("ollama", None, 11434),
        ("ensicompute", None, 11435),
        ("ollama", 11440, 11440),
        ("ensicompute", 11441, 11441),
    ],
)
def test_factory_routes_without_requests(backend, port, expected):
    client = create_client(backend, model="test-model", port=port)
    assert isinstance(client, OllamaClient)
    assert client.port == expected
    assert client.model == "test-model"


def test_unknown_backend_has_no_fallback():
    with pytest.raises(ValueError, match="Unsupported backend"):
        create_client("unknown")


@pytest.mark.parametrize("backend", ["ollama", "ensicompute"])
@pytest.mark.parametrize("port", [0, -1, 65536, True])
def test_invalid_explicit_port_is_not_replaced(backend, port):
    with pytest.raises(ValueError, match="port"):
        create_client(backend, port=port)


@pytest.mark.parametrize(
    "flags,expected_port,expected_memory_checks",
    [
        ([], 11434, 1),
        (["--backend-llm", "ensicompute"], 11435, 0),
        (["--backend-llm", "ensicompute", "--port", "11440"], 11440, 0),
    ],
)
def test_cli_routes_and_applies_local_memory_guard(
    monkeypatch, flags, expected_port, expected_memory_checks
):
    memory_checks, releases, observed = [], [], []
    monkeypatch.setattr(local_demo, "require_memory_headroom", lambda: memory_checks.append(True))
    monkeypatch.setattr(OllamaClient, "unload", lambda self: releases.append(self))
    monkeypatch.setattr(local_demo, "InvestigationTools", lambda **kwargs: object())
    monkeypatch.setattr(local_demo, "markdown_report", lambda result: "Offline draft")

    def investigate(*args, client, **kwargs):
        observed.append(client)
        return {"status": "draft", "model": {"model": client.model}, "elapsed_seconds": 0}

    monkeypatch.setattr(local_demo, "investigate_local", investigate)
    assert local_demo.main(["--enable-llm", *flags]) == 0
    assert len(observed) == 1 and observed[0].port == expected_port
    assert len(memory_checks) == expected_memory_checks
    assert releases == observed


def test_memory_failure_prevents_client_creation(monkeypatch):
    def low_memory():
        raise OllamaError("Insufficient memory")

    monkeypatch.setattr(local_demo, "require_memory_headroom", low_memory)
    monkeypatch.setattr(
        local_demo,
        "create_client",
        lambda *a, **k: pytest.fail("Client created after guard failed"),
    )
    with pytest.raises(SystemExit) as error:
        local_demo.main(["--enable-llm"])
    assert error.value.code == 2


def test_remote_selection_still_requires_enablement(monkeypatch):
    monkeypatch.setattr(
        local_demo, "create_client", lambda *a, **k: pytest.fail("Client created without consent")
    )
    with pytest.raises(SystemExit) as error:
        local_demo.main(["--backend-llm", "ensicompute"])
    assert error.value.code == 2
