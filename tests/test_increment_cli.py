"""Command-line selection keeps research models explicit and the default intact."""

import sys
from pathlib import Path

import pytest

from forge import cli


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ([], {"retrieval_backend": "tfidf", "torch_artifact": None}),
        (
            ["--retriever", "bm25", "--torch-artifact", "models/research-run"],
            {"retrieval_backend": "bm25", "torch_artifact": Path("models/research-run")},
        ),
    ],
)
def test_investigation_cli_routes_explicit_choices(monkeypatch, capsys, options, expected):
    calls = []

    def investigate(recording, **kwargs):
        calls.append((recording, kwargs))
        return object()

    monkeypatch.setattr("forge.agents.service.investigate_recording", investigate)
    monkeypatch.setattr("forge.reports.incident.markdown", lambda report: "draft for review")
    monkeypatch.setattr(sys, "argv", ["forge", "investigate", "--recording", "valve1/1", *options])
    assert cli.main() == 0
    assert calls == [("valve1/1", expected)]
    assert "draft for review" in capsys.readouterr().out


def test_cli_rejects_unknown_retriever_before_loading_a_model(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["forge", "investigate", "--retriever", "invented"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_service_uses_explicit_artifact_without_loading_active_model(monkeypatch):
    import types

    import numpy as np
    import pandas as pd

    from forge.agents import service
    from forge.data.datasets import FEATURES
    from forge.ml.detectors import Detector

    reference = pd.DataFrame({name: np.linspace(0, 10, 100) for name in FEATURES})
    detector = Detector.fit(reference)
    frame = reference.iloc[:8].copy()
    frame["datetime"] = pd.date_range("2020-01-01", periods=8, freq="s")
    frame["Pressure"] = 100
    metadata = {
        "run_id": "explicit-research-run",
        "model_sha256": "0" * 64,
        "threshold": 3.0,
        "policy": {"persistence": 3, "max_gap_seconds": 2},
    }
    calls = []
    module = types.ModuleType("forge.ml.torch_artifacts")

    def load(folder, root):
        calls.append((folder, root))
        return detector, metadata

    module.load_torch_artifact = load
    monkeypatch.setitem(sys.modules, "forge.ml.torch_artifacts", module)
    monkeypatch.setattr(service, "load_model", lambda *a, **k: pytest.fail("Active model loaded"))
    monkeypatch.setattr(
        service, "load_development_recording", lambda *a: (frame, {"sha256": "1" * 64})
    )
    report = service.investigate_recording(
        "valve1/1", retrieval_backend="bm25", torch_artifact=Path("models/research-run")
    )
    assert calls == [(cli.PROJECT_ROOT / "models/research-run", cli.PROJECT_ROOT)]
    assert report.model_run == "explicit-research-run"
    assert report.status == "needs_review" and report.review_status == "unreviewed"
    assert report.evidence


def test_missing_optional_torch_has_an_install_message(monkeypatch):
    from forge.agents import service

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delitem(sys.modules, "forge.ml.torch_artifacts", raising=False)
    with pytest.raises(ValueError, match="requirements-torch.txt"):
        service.investigate_recording(torch_artifact=Path("models/research-run"))
