"""Behavioral checks for the isolated PyTorch development increment."""

import importlib

import numpy as np
import pandas as pd
import pytest

from forge.data.datasets import FEATURES


@pytest.fixture
def torch_runtime():
    return pytest.importorskip("torch", reason="Requires the optional PyTorch dependency.")


def recording(n=12, *, start="2024-01-01", offset=0, group="g"):
    frame = pd.DataFrame(
        {name: np.arange(n, dtype=float) + offset + i for i, name in enumerate(FEATURES)}
    )
    frame["datetime"] = pd.date_range(start, periods=n, freq="s")
    frame["anomaly"] = (np.arange(n) >= n // 2).astype(int)
    frame["changepoint"] = 0
    frame["_group"] = group
    return frame


def data_module():
    return importlib.import_module("forge.ml.torch_data")


def test_causal_windows_exclude_future_and_reset_at_gaps():
    causal_windows = data_module().causal_windows
    frame = recording()
    x, ready = causal_windows(frame, np.zeros(8), np.ones(8), window=4, max_gap_seconds=2)
    assert ready.tolist() == [False] * 3 + [True] * 9
    np.testing.assert_array_equal(x[3], frame[FEATURES].iloc[:4].to_numpy().T)
    changed = frame.copy()
    changed.loc[6:, FEATURES] = 99999
    later, _ = causal_windows(changed, np.zeros(8), np.ones(8), window=4, max_gap_seconds=2)
    np.testing.assert_array_equal(x[:6], later[:6])
    frame.loc[6:, "datetime"] += pd.Timedelta(seconds=30)
    _, ready = causal_windows(frame, np.zeros(8), np.ones(8), window=4, max_gap_seconds=2)
    assert ready.tolist() == [False] * 3 + [True] * 3 + [False] * 3 + [True] * 3


def test_annotations_and_absolute_time_are_not_inputs():
    causal_windows = data_module().causal_windows
    frame = recording()
    changed = frame.copy()
    changed["anomaly"] = 100
    changed["changepoint"] = 999
    changed["experiment_id"] = "unrelated"
    changed["datetime"] += pd.Timedelta(days=1000)
    for left, right in zip(
        causal_windows(frame, np.zeros(8), np.ones(8), window=4),
        causal_windows(changed, np.zeros(8), np.ones(8), window=4),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)


def test_dedup_retains_original_source_context_and_rejects_ambiguous_targets():
    prepare_windows = data_module().prepare_windows
    frame = recording()
    # First inventory source owns overlap context, even when the later source is ready.
    short = frame.iloc[4:].reset_index(drop=True)
    result = prepare_windows({"short": short, "full": frame}, np.zeros(8), np.ones(8), window=4)
    assert len(result.y) == 6  # full endpoint 3, short endpoints 7..11
    row = result.rows.loc[result.rows.datetime.eq(frame.datetime.iloc[7])].iloc[0]
    assert row["_source"] == "short"
    assert result.audit["g"]["duplicate_rows"] == 8
    conflict = frame.copy()
    conflict.loc[8, "anomaly"] = 0
    result = prepare_windows(
        {"first": frame, "conflict": conflict}, np.zeros(8), np.ones(8), window=4
    )
    assert len(result.y) == 8
    assert result.audit["g"]["ambiguous_rows"] == 1


def test_recordings_never_share_window_history():
    result = data_module().prepare_windows(
        {"a": recording(3), "b": recording(3, start="2024-02-01", group="other")},
        np.zeros(8),
        np.ones(8),
        window=4,
    )
    assert result.x.shape == (0, 8, 4)
    assert len(result.y) == 0


def test_invalid_timestamp_order_and_scaler_are_rejected():
    windows = data_module().causal_windows
    frame = recording()
    with pytest.raises(ValueError, match="timestamps"):
        windows(frame.iloc[::-1], np.zeros(8), np.ones(8), window=4)
    with pytest.raises(ValueError, match="scale"):
        windows(frame, np.zeros(8), np.zeros(8), window=4)


def tiny_config():
    return {
        "schema_version": 1,
        "seed": 42,
        "window": 8,
        "channels": 4,
        "kernel_size": 3,
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "cpu_threads": 1,
        "persistence": 1,
        "max_gap_seconds": 2,
        "minimum_recall": 0.3,
        "threshold_quantiles": 11,
    }


def tiny_fit():
    module = importlib.import_module("forge.ml.torch_model")
    frame = recording(80)
    normal = frame.loc[frame.anomaly.eq(0), FEATURES]
    return module.fit_detector({"source": frame}, normal, tiny_config())


@pytest.mark.usefixtures("torch_runtime")
def test_fit_is_deterministic_and_scaler_uses_supplied_training_normals_only():
    first, history, audit = tiny_fit()
    second, other_history, _ = tiny_fit()
    normal = recording(80).loc[lambda f: f.anomaly.eq(0), FEATURES]
    np.testing.assert_array_equal(first.center, normal.median().to_numpy())
    expected_scale = 1.4826 * (normal - normal.median()).abs().median().to_numpy()
    np.testing.assert_allclose(first.scale, expected_scale)
    np.testing.assert_array_equal(first.score(recording(80)), second.score(recording(80)))
    assert history == other_history
    assert history[-1]["training_loss"] < history[0]["training_loss"]
    assert audit["fit_rows"] == 73
    assert audit["positive_training_targets"] == 40
    assert len(history) == tiny_config()["epochs"]
    assert first.describe()["features"] == FEATURES


@pytest.mark.usefixtures("torch_runtime")
def test_adapter_is_causal_annotation_independent_and_readiness_aware():
    detector, _, _ = tiny_fit()
    frame = recording(80)
    scores = detector.score(frame)
    assert np.all(scores[:7] == 0)
    assert detector.readiness(frame).sum() == 73
    altered = frame.copy()
    altered["anomaly"] = 1 - frame.anomaly
    altered["changepoint"] = 100
    altered.loc[40:, FEATURES] = 90000
    np.testing.assert_array_equal(scores[:40], detector.score(altered)[:40])
    np.testing.assert_allclose(
        detector.deviations(frame), (frame[FEATURES] - detector.center) / detector.scale
    )


def pinned_provenance(root):
    from pathlib import Path

    from forge.ml.training import digest

    repo = Path(__file__).resolve().parents[1]
    names = ("data/skab-inventory.json", "data/skab-splits.json")
    for name in names:
        destination = root / name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes((repo / name).read_bytes())
    return {"manifests": {name: digest(root / name) for name in names}}


@pytest.mark.usefixtures("torch_runtime")
def test_artifact_roundtrip_scores_and_tamper_rejection(tmp_path):
    import json

    artifacts = importlib.import_module("forge.ml.torch_artifacts")
    detector, history, audit = tiny_fit()
    metadata = {
        "run_id": "tiny",
        "threshold": 0.5,
        "policy": {"persistence": 1, "max_gap_seconds": 2},
        "training_audit": audit,
        "provenance": pinned_provenance(tmp_path),
    }
    folder = tmp_path / "run"
    artifacts.save_torch_artifact(folder, detector, metadata, history)
    loaded, saved = artifacts.load_torch_artifact(folder, root=tmp_path)
    np.testing.assert_array_equal(detector.score(recording(80)), loaded.score(recording(80)))
    assert saved["model_sha256"]
    original = (folder / "weights.pt").read_bytes()
    (folder / "weights.pt").write_bytes(original + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        artifacts.load_torch_artifact(folder, root=tmp_path)
    (folder / "weights.pt").write_bytes(original)
    config = json.loads((folder / "config.json").read_text())
    config["window"] += 1
    (folder / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="checksum"):
        artifacts.load_torch_artifact(folder, root=tmp_path)


@pytest.mark.usefixtures("torch_runtime")
def test_loader_rejects_wrong_state_shape_even_with_updated_hash(tmp_path):
    import json

    import torch

    from forge.ml.training import digest, write_json

    artifacts = importlib.import_module("forge.ml.torch_artifacts")
    detector, history, _ = tiny_fit()
    folder = tmp_path / "run"
    artifacts.save_torch_artifact(
        folder,
        detector,
        {
            "run_id": "tiny",
            "threshold": 0.5,
            "policy": {"persistence": 1, "max_gap_seconds": 2},
            "provenance": pinned_provenance(tmp_path),
        },
        history,
    )
    state = torch.load(folder / "weights.pt", weights_only=True)
    key = next(iter(state))
    state[key] = state[key][:1]
    torch.save(state, folder / "weights.pt")
    metadata = json.loads((folder / "metadata.json").read_text())
    metadata["model_sha256"] = digest(folder / "weights.pt")
    write_json(folder / "metadata.json", metadata)
    manifest = json.loads((folder / "checksums.json").read_text())
    for name in ("weights.pt", "metadata.json"):
        manifest[name] = digest(folder / name)
    write_json(folder / "checksums.json", manifest)
    with pytest.raises(ValueError, match="state|shape"):
        artifacts.load_torch_artifact(folder, root=tmp_path)


@pytest.mark.usefixtures("torch_runtime")
def test_precision_first_threshold_requires_recall_and_reports_fallback():
    training = importlib.import_module("forge.ml.torch_training")
    frame = recording(10)
    frame["anomaly"] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    scores = {"g": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])}
    ready = {"g": np.ones(10, dtype=bool)}
    selected, scan = training.select_operating_point({"g": frame}, scores, ready, tiny_config())
    assert selected["metrics"]["precision"] == 1
    assert selected["metrics"]["recall"] >= 0.3
    assert not selected["fallback_used"]
    ready["g"][:8] = False
    config = {**tiny_config(), "minimum_recall": 0.9}
    selected, _ = training.select_operating_point({"g": frame}, scores, ready, config)
    assert selected["fallback_used"]
    assert 0 < selected["metrics"]["recall"] < 0.9
    assert selected["metrics"]["tp"] + selected["metrics"]["fp"] > 0
    assert len(scan) > 1


@pytest.mark.usefixtures("torch_runtime")
def test_training_never_fits_validation_and_preserves_pointers(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    training = importlib.import_module("forge.ml.torch_training")
    artifacts = importlib.import_module("forge.ml.torch_artifacts")
    repo = Path(__file__).resolve().parents[1]
    for name in ("data/skab-inventory.json", "data/skab-splits.json"):
        destination = tmp_path / name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes((repo / name).read_bytes())
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/pytorch-v1.json").write_text(json.dumps(tiny_config()))
    (tmp_path / "models").mkdir()
    for name in ("active.json", "latest.json"):
        (tmp_path / "models" / name).write_text("preserve-me")
    train = recording(80)
    validation = recording(80, start="2024-02-01", offset=5)
    requests = []

    def streams(partition, root):
        assert root == tmp_path
        requests.append(partition)
        assert partition in {"train", "validation"}
        return {"source": train if partition == "train" else validation}

    monkeypatch.setattr(training, "development_streams", streams)
    monkeypatch.setattr(
        training,
        "normal_training",
        lambda root: (train.loc[train.anomaly.eq(0), FEATURES], {"training_rows": 40}),
    )
    first_path, first_meta = training.run_training(tmp_path)
    validation["anomaly"] = 1 - validation.anomaly
    validation.loc[:, FEATURES] += 2
    second_path, _ = training.run_training(tmp_path)
    first, _ = artifacts.load_torch_artifact(first_path, root=tmp_path)
    second, _ = artifacts.load_torch_artifact(second_path, root=tmp_path)
    np.testing.assert_array_equal(first.center, second.center)
    np.testing.assert_array_equal(first.scale, second.scale)
    np.testing.assert_array_equal(first.score(train), second.score(train))
    assert requests == ["train", "validation", "train", "validation"]
    assert first_meta["validation_metrics"]["unavailable_rows"] == 7
    assert first_meta["provenance"]["partitions_read"] == ["train", "validation"]
    for name in ("active.json", "latest.json"):
        assert (tmp_path / "models" / name).read_text() == "preserve-me"


@pytest.mark.usefixtures("torch_runtime")
def test_loaded_detector_runs_existing_investigation(tmp_path):
    from forge.agents.workflow import investigate

    artifacts = importlib.import_module("forge.ml.torch_artifacts")
    detector, history, _ = tiny_fit()
    folder = tmp_path / "run"
    artifacts.save_torch_artifact(
        folder,
        detector,
        {
            "run_id": "tiny",
            "threshold": 0.99,
            "policy": {"persistence": 1, "max_gap_seconds": 2},
            "provenance": pinned_provenance(tmp_path),
        },
        history,
    )
    loaded, metadata = artifacts.load_torch_artifact(folder, root=tmp_path)
    report = investigate(recording(7), loaded, metadata, "synthetic", "synthetic", retriever=None)
    assert report.status == "insufficient_data"
    assert report.observation["scored_rows"] == 0


@pytest.mark.usefixtures("torch_runtime")
def test_module_cli_reloads_and_investigates_with_existing_retriever(tmp_path, monkeypatch, capsys):
    import json
    import sys
    from pathlib import Path

    artifacts = importlib.import_module("forge.ml.torch_artifacts")
    training = importlib.import_module("forge.ml.torch_training")
    detector, history, _ = tiny_fit()
    folder = tmp_path / "run"
    artifacts.save_torch_artifact(
        folder,
        detector,
        {
            "run_id": "tiny",
            "threshold": 0.1,
            "policy": {"persistence": 1, "max_gap_seconds": 2},
            "provenance": pinned_provenance(tmp_path),
        },
        history,
    )
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(training, "inspect_recording", lambda record, root: recording(80))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "torch_training",
            "--root",
            str(root),
            "--artifact",
            str(folder),
            "--recording",
            "valve1/1",
        ],
    )
    training.main()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "needs_review"
    assert report["observation"]["scored_rows"] == 73
    assert report["evidence"]


def test_windows_reject_explicitly_mixed_source_recordings():
    frame = recording()
    frame["experiment_id"] = ["a"] * 6 + ["b"] * 6
    with pytest.raises(ValueError, match="source recording"):
        data_module().causal_windows(frame, np.zeros(8), np.ones(8), window=4)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_provenance",
        "empty_provenance",
        "empty_manifests",
        "partial_manifests",
        "extra_manifest",
        "wrong_name",
        "wrong_window",
        "wrong_features",
    ],
)
@pytest.mark.usefixtures("torch_runtime")
def test_loader_rejects_incomplete_provenance_or_descriptor_after_rehash(tmp_path, mutation):
    import json

    from forge.ml.training import digest, write_json

    artifacts = importlib.import_module("forge.ml.torch_artifacts")
    detector, history, _ = tiny_fit()
    folder = tmp_path / "run"
    artifacts.save_torch_artifact(
        folder,
        detector,
        {
            "run_id": "tiny",
            "threshold": 0.5,
            "policy": {"persistence": 1, "max_gap_seconds": 2},
            "provenance": pinned_provenance(tmp_path),
        },
        history,
    )
    metadata = json.loads((folder / "metadata.json").read_text())
    if mutation == "missing_provenance":
        del metadata["provenance"]
    elif mutation == "empty_provenance":
        metadata["provenance"] = {}
    elif mutation == "empty_manifests":
        metadata["provenance"]["manifests"] = {}
    elif mutation == "partial_manifests":
        del metadata["provenance"]["manifests"]["data/skab-splits.json"]
    elif mutation == "extra_manifest":
        metadata["provenance"]["manifests"]["unexpected.json"] = "irrelevant"
    elif mutation == "wrong_name":
        metadata["detector"]["name"] = "other_detector"
    elif mutation == "wrong_window":
        metadata["detector"]["window"] += 1
    else:
        metadata["detector"]["features"] = list(reversed(FEATURES))
    write_json(folder / "metadata.json", metadata)
    manifest = json.loads((folder / "checksums.json").read_text())
    manifest["metadata.json"] = digest(folder / "metadata.json")
    write_json(folder / "checksums.json", manifest)
    with pytest.raises(ValueError, match="provenance|configuration"):
        artifacts.load_torch_artifact(folder, root=tmp_path)
