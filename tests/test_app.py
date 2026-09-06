"""Exercise Streamlit's actual investigation and explicit review controls."""

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from forge.config import PROJECT_ROOT
from forge.data.datasets import FEATURES
from forge.ml.detectors import Detector


def test_investigation_review_and_recording_change(monkeypatch, tmp_path):
    import forge.ui.investigation as ui

    training = pd.DataFrame({name: np.linspace(0, 10, 100) for name in FEATURES})
    detector = Detector.fit(training)
    frame = training.iloc[:8].copy()
    frame["Pressure"] = 100
    frame["datetime"] = pd.date_range("2020-01-01", periods=8, freq="s")
    metadata = {
        "run_id": "synthetic-ui",
        "model_sha256": "0" * 64,
        "threshold": 3.0,
        "policy": {"persistence": 3, "max_gap_seconds": 2},
    }
    record = {"experiment_id": "valve1/1", "sha256": "1" * 64}
    second = {**record, "experiment_id": "valve1/2"}
    monkeypatch.setattr(ui, "load_model", lambda: (detector, metadata, tmp_path))
    monkeypatch.setattr(ui, "available_recordings", lambda: [record, second])
    monkeypatch.setattr(ui, "recording_path", lambda r: tmp_path)
    monkeypatch.setattr(
        ui,
        "load_development_recording",
        lambda key: (frame, record if key == "valve1/1" else second),
    )
    app = AppTest.from_file(str(PROJECT_ROOT / "src/forge/ui/app.py")).run(timeout=30)
    assert not app.exception
    app.button(key="investigate").click().run()
    assert not app.exception
    assert app.session_state["incident_report"].review_status == "unreviewed"
    assert app.button(key="mark_reviewed").disabled
    app.text_input(key="reviewer_name").input("Test reviewer").run()
    app.checkbox(key="review_confirmed").check().run()
    app.button(key="mark_reviewed").click().run()
    assert not app.exception
    assert app.session_state["incident_report"].review_status == "reviewed"
    assert len(app.get("download_button")) == 2
    app.selectbox(key="investigation_recording").select("valve1/2").run()
    assert not app.exception and "incident_report" not in app.session_state


def test_missing_model_shows_setup_instructions(monkeypatch):
    import forge.ui.investigation as ui

    def absent():
        raise FileNotFoundError("No local model")

    monkeypatch.setattr(ui, "load_model", absent)
    app = AppTest.from_file(str(PROJECT_ROOT / "src/forge/ui/app.py")).run(timeout=30)
    assert not app.exception
    assert any("Train the local detector" in message.value for message in app.info)
