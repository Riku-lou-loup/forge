"""Fold isolation, selection failure handling and auditable outer predictions."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from forge.config import PROJECT_ROOT
from forge.data.datasets import FEATURES
from forge.ml import generalization as audit
from forge.ml.training import write_json


def synthetic_streams(count=6):
    streams = {}
    for index in range(count):
        rng = np.random.default_rng(100 + index)
        labels = ((np.arange(100) >= 40) & (np.arange(100) < 75)).astype(int)
        values = rng.normal(size=(100, 8)) + labels[:, None] * 4
        frame = pd.DataFrame(values, columns=FEATURES)
        frame["datetime"] = pd.date_range("2020-01-01", periods=100, freq="s") + pd.Timedelta(
            days=index
        )
        frame["anomaly"] = labels
        frame["_group"] = f"group-{index}"
        streams[f"source-{index}"] = frame
    return streams


def small_config():
    config = json.loads((PROJECT_ROOT / "configs/generalization-v1.json").read_text())
    config.update(
        outer_folds=3,
        inner_folds=2,
        reference_readings=10,
        rolling_readings=5,
        threshold_quantiles=11,
    )
    config["candidates"] = [
        {
            "id": "current_hgb",
            "max_iter": 5,
            "max_leaf_nodes": 3,
            "min_samples_leaf": 5,
            "l2_regularization": 1.0,
            "learning_rate": 0.1,
            "early_stopping": False,
        },
        {
            "id": "shallow_hgb",
            "max_iter": 5,
            "max_leaf_nodes": 2,
            "min_samples_leaf": 10,
            "l2_regularization": 5.0,
            "learning_rate": 0.1,
            "early_stopping": False,
        },
    ]
    config["isolation_forest"] = {"n_estimators": 5, "max_samples": 32}
    return config


def test_folds_hold_entire_overlap_groups_and_cover_every_group_once():
    streams = synthetic_streams()
    streams["overlap-copy"] = streams["source-0"].iloc[20:].copy()
    audit.validate_pool(streams)
    folds = audit.group_folds(streams, 3, 42)
    assessed = []
    for fold in folds:
        assert not set(fold["fit_groups"]) & set(fold["assessment_groups"])
        held = audit.subset(streams, fold["assessment_groups"])
        assert ("source-0" in held) == ("overlap-copy" in held)
        assessed.extend(fold["assessment_groups"])
    assert len(assessed) == len(set(assessed)) == 6
    modified = {name: frame.assign(anomaly=1 - frame.anomaly) for name, frame in streams.items()}
    assert audit.group_folds(modified, 3, 42) == folds


def test_retimestamped_duplicate_cannot_cross_group_boundary():
    streams = synthetic_streams(2)
    streams["source-1"].loc[:, FEATURES] = streams["source-0"][FEATURES].to_numpy()
    with pytest.raises(ValueError, match="cross recording groups"):
        audit.validate_pool(streams)


def test_scaling_uses_only_normal_observations_from_explicit_fit_groups():
    streams = synthetic_streams()
    chosen = audit.subset(streams, ["group-1", "group-2"])
    config = small_config()
    with threadpool_limits(limits=2):
        detector, evidence = audit.fit_candidate(chosen, config["candidates"][0], config)
    normal = pd.concat([f.loc[f.anomaly.eq(0), FEATURES] for f in chosen.values()])
    np.testing.assert_allclose(detector.center, normal.median().to_numpy())
    assert evidence["normal_scaling_rows"] == len(normal)
    assert evidence["fit_groups"] == ["group-1", "group-2"]
    assert evidence["positive_training_targets"] == 70


def test_infeasible_selection_is_exposed_without_changing_recall_floors():
    streams = synthetic_streams(2)
    groups, _ = audit.merge_streams(streams)
    config = small_config()
    snapshot = copy.deepcopy(config)
    scores = {name: np.zeros(len(frame)) for name, frame in groups.items()}
    ready = {name: np.ones(len(frame), dtype=bool) for name, frame in groups.items()}
    choice, _ = audit.choose_operating_point(groups, scores, ready, config)
    assert not choice["feasible"] and choice["status"] == "recall_constraints_unmet"
    assert config == snapshot


def create_workspace(root):
    (root / "configs").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "models").mkdir()
    write_json(root / "configs/generalization-v1.json", small_config())
    for name in [
        "data/skab-splits.json",
        "data/skab-inventory.json",
        "models/active.json",
        "models/latest.json",
    ]:
        write_json(root / name, {"synthetic": True})
    for relative in [
        "src/forge/ml/generalization.py",
        "src/forge/ml/development.py",
        "src/forge/ml/operating.py",
        "src/forge/ml/detectors.py",
        "src/forge/ml/metrics.py",
        "src/forge/data/datasets.py",
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((PROJECT_ROOT / relative).read_bytes())


def test_nested_run_excludes_outer_labels_from_selection_and_fitting(tmp_path, monkeypatch):
    original = synthetic_streams()
    streams = copy.deepcopy(original)
    partitions_read = []

    def permitted(partition, root):
        assert partition in {"train", "validation"}
        partitions_read.append(partition)
        return {
            name: frame
            for i, (name, frame) in enumerate(streams.items())
            if (i < 3) == (partition == "train")
        }

    monkeypatch.setattr(audit, "development_streams", permitted)
    create_workspace(tmp_path / "first")
    first_folder, first = audit.run_audit(tmp_path / "first")
    first_fold = first["fold_results"][0]
    outer_assessment = set(first_fold["assessment_groups"])
    inner = json.loads((first_folder / "outer-1-inner.json").read_text())
    for fit in inner["fits"]:
        assert not outer_assessment & set(fit["fit_audit"]["fit_groups"])
        assert set(fit["fit_groups"]).isdisjoint(fit["assessment_groups"])
        assert set(fit["fit_audit"]["fit_groups"]) == set(fit["fit_groups"])
    # Deliberately change only the first outer fold's labels. Its chosen
    # hyperparameters, thresholds, training reference and fitted model must not move.
    for frame in streams.values():
        if frame["_group"].iloc[0] in outer_assessment:
            frame["anomaly"] = 1 - frame.anomaly
    create_workspace(tmp_path / "second")
    _, second = audit.run_audit(tmp_path / "second")
    for procedure in ["isolation_forest", "current_hgb", "tuned_hgb"]:
        a = first_fold["procedures"][procedure]
        b = second["fold_results"][0]["procedures"][procedure]
        assert a["candidate"] == b["candidate"]
        assert a["selection"] == b["selection"]
        assert a["fit_audit"] == b["fit_audit"]
        assert a["model_sha256"] == b["model_sha256"]
        assert a["assessment"]["pooled"] != b["assessment"]["pooled"]
        assert set(first["procedures"][procedure]["per_group"]) == {f"group-{i}" for i in range(6)}
        assert (
            first["procedures"][procedure]["pooled"]["tp"]
            + first["procedures"][procedure]["pooled"]["fn"]
            == 210
        )
    assert set(partitions_read) == {"train", "validation"}
    assert first["final_refit"]["activated"] is False
    assert json.loads((tmp_path / "first/models/active.json").read_text()) == {"synthetic": True}
