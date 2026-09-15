"""Only matching trusted local artifacts can reach deserialization."""

import pickle
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

from forge.data.datasets import FEATURES
from forge.ml.detectors import Detector
from forge.ml.training import digest, load_model, write_json


@pytest.fixture
def saved_model(tmp_path):
    folder = tmp_path / "models/test-run"
    folder.mkdir(parents=True)
    for relative in ["configs/baseline.json", "data/skab-splits.json", "data/skab-inventory.json"]:
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text("{}")
    frame = pd.DataFrame({name: np.arange(10.0) for name in FEATURES})
    detector = Detector.fit(frame)
    (folder / "detector.pkl").write_bytes(pickle.dumps(detector))
    metadata = {
        "run_id": "test-run",
        "sklearn_version": version("scikit-learn"),
        "model_sha256": digest(folder / "detector.pkl"),
        "detector": {"name": detector.name},
        "config_sha256": digest(tmp_path / "configs/baseline.json"),
        "split_sha256": digest(tmp_path / "data/skab-splits.json"),
        "inventory_sha256": digest(tmp_path / "data/skab-inventory.json"),
    }
    write_json(folder / "metadata.json", metadata)
    write_json(
        tmp_path / "models/latest.json",
        {"run_id": "test-run", "metadata_sha256": digest(folder / "metadata.json")},
    )
    return tmp_path, folder, detector, frame


def test_artifact_round_trip_preserves_scores(saved_model):
    root, _, original, frame = saved_model
    loaded, _, _ = load_model(root)
    np.testing.assert_array_equal(loaded.score(frame), original.score(frame))


def test_corrupted_pickle_is_rejected_before_deserialization(saved_model, monkeypatch):
    root, folder, _, _ = saved_model
    (folder / "detector.pkl").write_bytes(b"corrupted")
    monkeypatch.setattr(
        "forge.ml.training.pickle.loads", lambda _: pytest.fail("Must verify before deserializing")
    )
    with pytest.raises(ValueError, match="Model checksum"):
        load_model(root)


def test_changed_training_manifest_requires_matching_checkout(saved_model):
    root, _, _, _ = saved_model
    (root / "data/skab-splits.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="provenance changed"):
        load_model(root)


def test_development_activation_preserves_baseline_and_checks_provenance(saved_model):
    from forge.ml.activation import activate

    root, baseline_folder, detector, _ = saved_model
    baseline_pointer = (root / "models/latest.json").read_bytes()
    folder = root / "models/development-test"
    folder.mkdir()
    (folder / "detector.pkl").write_bytes((baseline_folder / "detector.pkl").read_bytes())
    config = {
        "minimum_point_recall": 0.6,
        "minimum_event_recall": 0.6,
        "minimum_group_point_recall": 0.2,
    }
    write_json(root / "configs/improvement-v2.json", config)
    result = {
        "schema_version": 2,
        "run_id": folder.name,
        "created_at": "synthetic",
        "sklearn_version": version("scikit-learn"),
        "config": config,
        "baseline_validation": {"precision": 0.5},
        "config_sha256": digest(root / "configs/improvement-v2.json"),
        "split_sha256": digest(root / "data/skab-splits.json"),
        "inventory_sha256": digest(root / "data/skab-inventory.json"),
        "scope": "synthetic test",
        "selected": {
            "metrics": {
                "precision": 0.9,
                "recall": 0.65,
                "event_recall": 0.8,
                "minimum_group_recall": 0.3,
            },
            "model_sha256": digest(folder / "detector.pkl"),
            "descriptor": detector.describe(),
            "threshold": 1.0,
            "policy": {"persistence": 5, "max_gap_seconds": 2},
        },
    }
    write_json(folder / "comparison.json", result)
    activate(folder.name, root)
    assert load_model(root, active=True)[1]["run_id"] == folder.name
    assert load_model(root)[1]["run_id"] == baseline_folder.name
    assert (root / "models/latest.json").read_bytes() == baseline_pointer
    (root / "configs/improvement-v2.json").write_text("{}")
    with pytest.raises(ValueError, match="provenance changed"):
        load_model(root, active=True)
    with pytest.raises(ValueError, match="provenance changed"):
        activate(folder.name, root)
    assert load_model(root)[1]["run_id"] == baseline_folder.name


def test_active_loader_falls_back_only_when_pointer_is_absent(saved_model):
    root, folder, _, _ = saved_model
    assert load_model(root, active=True)[1]["run_id"] == folder.name
    write_json(root / "models/active.json", {"run_id": "../../outside"})
    with pytest.raises(ValueError, match="Invalid model run path"):
        load_model(root, active=True)
