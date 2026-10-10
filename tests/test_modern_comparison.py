"""Precision-policy behavior and leakage boundaries on synthetic recordings."""

import copy

import numpy as np
import pandas as pd
import pytest

from forge.data.datasets import FEATURES
from forge.ml.modern_comparison import (
    choose_precision,
    fhalf,
    precision_key,
    training_table,
)
from forge.ml.operating import OperatingDetector


def stream(length, *, group="a", day=0, offset=0, labels=None):
    values = np.arange(length, dtype=float)[:, None] * np.arange(1, 9)[None, :]
    frame = pd.DataFrame(values + offset, columns=FEATURES)
    frame["datetime"] = pd.date_range("2020", periods=length, freq="s") + pd.Timedelta(days=day)
    frame["anomaly"] = np.arange(length) % 2 if labels is None else labels
    frame["_group"] = group
    return frame


def detector():
    return OperatingDetector(
        name="synthetic",
        center=np.zeros(8),
        scale=np.ones(8),
        scale_methods=["synthetic"] * 8,
        estimator=None,
        representation="relative",
        reference_readings=2,
        rolling_readings=2,
        max_gap_seconds=2,
    )


def operating_choice(tp, fp, fn, *, threshold=0.5, onsets=1, feasible=False):
    return {
        "threshold": threshold,
        "policy": {"persistence": 3, "max_gap_seconds": 2},
        "metrics": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "false_onsets_per_available_normal_hour": onsets,
        },
        "feasible": feasible,
    }


@pytest.mark.parametrize(
    "counts,expected",
    [
        ({"tp": 20, "fp": 5, "fn": 20}, 25 / 35),
        ({"tp": 0, "fp": 0, "fn": 20}, 0),
        ({"tp": 0, "fp": 0, "fn": 0}, 0),
        ({"tp": 20, "fp": 0, "fn": 0}, 1),
    ],
)
def test_fhalf_uses_confusion_counts_including_all_missed_anomalies(counts, expected):
    assert fhalf(counts) == pytest.approx(expected)


def test_precision_policy_accepts_lower_recall_to_avoid_false_alerts():
    old_floor_choice = operating_choice(90, 900, 10, threshold=0.01, feasible=True)
    precise_choice = operating_choice(40, 2, 60, threshold=0.8, feasible=False)
    silent_choice = operating_choice(0, 0, 100, threshold=1.0)
    scan = [old_floor_choice, silent_choice, precise_choice]
    original = copy.deepcopy(scan)

    selected = choose_precision(scan)

    assert selected["threshold"] == 0.8
    assert selected["metrics"] == precise_choice["metrics"]
    assert selected["policy"] == precise_choice["policy"]
    assert selected["status"] == "f0.5_selected"
    assert selected["objective_f0_5"] == pytest.approx(fhalf(precise_choice["metrics"]))
    assert "feasible" not in selected
    assert scan == original


def test_precision_ties_prefer_fewer_false_onsets_then_higher_threshold():
    fragmented = operating_choice(20, 2, 10, threshold=0.9, onsets=10)
    fewer = operating_choice(20, 2, 10, threshold=0.6, onsets=2)
    conservative_tie = operating_choice(20, 2, 10, threshold=0.7, onsets=2)

    assert choose_precision([fragmented, fewer, conservative_tie])["threshold"] == 0.7
    assert precision_key(conservative_tie) == (
        fhalf(conservative_tie["metrics"]),
        conservative_tie["metrics"]["precision"],
        -2,
        conservative_tie["metrics"]["recall"],
        0.7,
    )


def test_empty_threshold_scan_does_not_create_an_operating_policy():
    with pytest.raises(ValueError):
        choose_precision([])


def test_training_retains_first_source_context_before_overlap_deduplication():
    first = stream(7)
    overlap = first.iloc[2:].copy()
    model = detector()
    expected, ready = model.transform(first)
    different_context, overlap_ready = model.transform(overlap)
    assert not np.allclose(expected[4:], different_context[overlap_ready])

    x, y, weights, audit = training_table(model, {"first": first, "overlap": overlap})

    np.testing.assert_allclose(x, expected[ready])
    np.testing.assert_array_equal(y, first.loc[ready, "anomaly"])
    np.testing.assert_allclose(weights, np.ones(5))
    assert audit["fit_rows"] == 5
    assert audit["positive_training_targets"] == int(y.sum())
    assert audit["representation"] == "relative"


def test_training_excludes_ambiguous_unknown_and_unavailable_targets():
    first = stream(7, labels=[1, 1, 0, 1, -1, 1, 0])
    overlap = first.iloc[2:6].copy()
    overlap.loc[3, "anomaly"] = 0
    model = detector()
    expected, _ = model.transform(first)

    x, y, weights, audit = training_table(model, {"first": first, "overlap": overlap})

    np.testing.assert_allclose(x, expected[[2, 5, 6]])
    np.testing.assert_array_equal(y, [0, 1, 0])
    assert np.isfinite(weights).all()
    assert audit["fit_rows"] == 3
    assert audit["positive_training_targets"] == 1


def test_training_assigns_equal_total_weight_to_unequal_length_groups():
    short = stream(4, group="short")
    long = stream(8, group="long", day=1, offset=100)

    x, _, weights, audit = training_table(detector(), {"short": short, "long": long})

    assert len(x) == 8
    assert weights[:2].sum() == pytest.approx(weights[2:].sum())
    assert weights.mean() == pytest.approx(1)
    assert (weights > 0).all()
    assert "group" in audit["group_weighting"].lower()


def test_targets_identifiers_and_changepoint_are_not_predictor_features():
    original = stream(7)
    altered = original.copy()
    altered["anomaly"] = 1 - altered.anomaly
    altered["changepoint"] = 1000
    altered["_group"] = "renamed"
    altered["experiment_id"] = "not-a-feature"

    a, labels_a, weights_a, _ = training_table(detector(), {"source": original})
    b, labels_b, weights_b, _ = training_table(detector(), {"renamed-source": altered})

    assert a.shape == (5, 32)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(labels_b, 1 - labels_a)
    np.testing.assert_array_equal(weights_a, weights_b)


def test_future_sensor_values_cannot_change_earlier_training_features():
    original = stream(9)
    altered = original.copy()
    altered.loc[7:, FEATURES] += 10000

    a, _, _, _ = training_table(detector(), {"source": original})
    b, _, _, _ = training_table(detector(), {"source": altered})

    np.testing.assert_array_equal(a[:5], b[:5])
    assert not np.array_equal(a[5:], b[5:])


def test_ranking_excludes_unknown_and_unavailable_values_and_marks_empty_groups():
    from forge.ml.modern_comparison import assess_ranking

    groups = {
        "mixed": stream(5, labels=[0, 1, -1, 1, 0]),
        "normal": stream(2, group="normal", labels=[0, 0]),
        "empty": stream(2, group="empty", labels=[1, 0]),
    }
    scores = {
        "mixed": np.array([0.1, 0.9, np.nan, np.nan, 0.2]),
        "normal": np.array([0.05, 0.06]),
        "empty": np.array([np.nan, np.nan]),
    }
    ready = {
        "mixed": np.array([True, True, True, False, True]),
        "normal": np.ones(2, dtype=bool),
        "empty": np.zeros(2, dtype=bool),
    }

    result = assess_ranking(groups, scores, ready)

    assert result["per_group"]["mixed"] == {
        "average_precision": 1.0,
        "available_labeled_rows": 3,
        "positive_rows": 1,
    }
    assert result["per_group"]["normal"]["average_precision"] is None
    assert result["per_group"]["empty"]["average_precision"] is None
    assert result["per_group"]["empty"]["available_labeled_rows"] == 0
    assert result["pooled_within_fold_average_precision"] == pytest.approx(1)
    assert result["zero_assessable_groups"] == ["empty"]


def test_nonfinite_available_score_is_rejected():
    from forge.ml.modern_comparison import assess_ranking

    groups = {"mixed": stream(2, labels=[0, 1])}
    with pytest.raises(ValueError, match="scores"):
        assess_ranking(
            groups, {"mixed": np.array([0.1, np.nan])}, {"mixed": np.ones(2, dtype=bool)}
        )


@pytest.mark.parametrize("labels", [[0, 0, 0, 0], [1, 1, 1, 1], [-1, -1, -1, -1]])
def test_training_rejects_missing_class_or_no_assessable_targets(labels):
    with pytest.raises(ValueError, match="both binary classes"):
        training_table(detector(), {"source": stream(4, labels=labels)})


@pytest.mark.parametrize("equal_metrics", [False, True])
def test_inner_models_exclude_their_assessment_groups_and_cover_each_group_once(
    monkeypatch, equal_metrics
):
    import forge.ml.modern_comparison as module

    streams = {name: stream(5, group=name, day=i) for i, name in enumerate(["a", "b", "c", "d"])}
    fit_calls = []
    scored = {"weak": [], "strong": []}

    def fit(fitting, candidate, config):
        names = set(fitting)
        fit_calls.append((candidate["id"], names))
        return (candidate["id"], names), {"fit_groups": sorted(names)}

    def score(fitted, assessment):
        candidate, fit_names = fitted
        assert fit_names.isdisjoint(assessment)
        assert fit_names | set(assessment) == set(streams)
        scored[candidate].extend(assessment)
        return (
            {
                name: np.full(len(frame), 0.8 if candidate == "strong" else 0.1)
                for name, frame in assessment.items()
            },
            {name: np.ones(len(frame), bool) for name, frame in assessment.items()},
        )

    def scan(groups, scores, ready, config):
        assert set(groups) == set(scores) == set(ready) == set(streams)
        strong = next(iter(scores.values()))[0] > 0.5
        candidate = operating_choice(
            10, 0 if strong or equal_metrics else 20, 10, threshold=0.9 if strong else 0.4
        )
        return candidate, [candidate, operating_choice(0, 0, 20, threshold=1)]

    monkeypatch.setattr(module, "fit_modern", fit)
    monkeypatch.setattr(module, "score_streams", score)
    monkeypatch.setattr(module, "choose_operating_point", scan)
    choices, evidence = module.select_inner(
        streams,
        {
            "inner_folds": 2,
            "seed": 42,
            "candidates": [{"id": "weak"}, {"id": "strong"}],
            "procedures": {
                "selected": {"selection": "f0.5", "candidates": ["weak", "strong"]},
                "reference": {"selection": "historical", "candidates": ["weak"]},
            },
        },
    )

    assert len(fit_calls) == len(evidence["fits"]) == 4
    # Scores from different estimators are not calibrated to a common scale.
    # An exact metric tie must retain the first candidate, not the largest threshold.
    assert choices["selected"]["candidate"] == ("weak" if equal_metrics else "strong")
    assert choices["reference"]["candidate"] == "weak"
    for names in scored.values():
        assert sorted(names) == sorted(streams)
        assert len(names) == len(set(names))


def test_outer_label_changes_cannot_change_fitting_selection_or_scores(tmp_path, monkeypatch):
    import json

    import forge.ml.modern_comparison as module
    from forge.config import PROJECT_ROOT
    from forge.ml.training import write_json

    config = {
        "seed": 42,
        "outer_folds": 2,
        "inner_folds": 2,
        "reference_readings": 5,
        "rolling_readings": 3,
        "max_gap_seconds": 2,
        "persistence_candidates": [1, 3],
        "threshold_quantiles": 5,
        "minimum_point_recall": 0.6,
        "minimum_event_recall": 0.6,
        "minimum_group_point_recall": 0.2,
        "candidates": [
            {
                "id": "tiny_hgb",
                "family": "hgb",
                "parameters": {
                    "max_iter": 4,
                    "max_leaf_nodes": 3,
                    "min_samples_leaf": 3,
                    "l2_regularization": 1.0,
                    "learning_rate": 0.1,
                    "early_stopping": False,
                },
            }
        ],
        "procedures": {
            "historical": {"selection": "historical", "candidates": ["tiny_hgb"]},
            "precision": {"selection": "f0.5", "candidates": ["tiny_hgb"]},
        },
    }
    streams = {}
    for index in range(6):
        labels = ((np.arange(45) >= 15) & (np.arange(45) < 32)).astype(int)
        frame = stream(45, group=f"g{index}", day=index, labels=labels)
        frame.loc[:, FEATURES] = (
            np.random.default_rng(index).normal(size=(45, 8)) + 3 * labels[:, None]
        )
        streams[f"s{index}"] = frame
    partitions = []

    def load(partition, root):
        assert partition in {"train", "validation"}
        partitions.append(partition)
        return {
            name: frame
            for i, (name, frame) in enumerate(streams.items())
            if (i < 3) == (partition == "train")
        }

    monkeypatch.setattr(module, "development_streams", load)
    # This test exercises the real small HGB fits; optional families are not fitted.
    monkeypatch.setattr(module, "version", lambda name: "synthetic-test")

    def workspace(name):
        root = tmp_path / name
        (root / "configs").mkdir(parents=True)
        write_json(root / "configs/modern-comparison-v1.json", config)
        for relative in [
            "models/active.json",
            "models/latest.json",
            "data/skab-splits.json",
            "data/skab-inventory.json",
            "docs/evaluation-results.json",
        ]:
            path = root / relative
            path.parent.mkdir(exist_ok=True)
            write_json(path, {"preserve": True})
        paths = [
            f"src/forge/ml/{name}.py"
            for name in [
                "modern_comparison",
                "tabm_estimator",
                "generalization",
                "development",
                "operating",
                "detectors",
                "metrics",
                "training",
            ]
        ]
        paths += ["src/forge/data/datasets.py", "src/forge/data/partitions.py"]
        for relative in paths:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((PROJECT_ROOT / relative).read_bytes())
        return root

    first_root = workspace("first")
    first_folder, first = module.run(first_root)
    first_fold = first["fold_results"][0]
    held = set(first_fold["assessment_groups"])
    for frame in streams.values():
        if frame["_group"].iloc[0] in held:
            frame["anomaly"] = 1 - frame.anomaly
    second_folder, second = module.run(workspace("second"))
    second_fold = second["fold_results"][0]

    assert first_fold["selection"] == second_fold["selection"]
    for candidate, first_audit in first_fold["fit_audit"].items():
        second_audit = second_fold["fit_audit"][candidate]
        stable_keys = set(first_audit) - {"fit_seconds", "score_seconds"}
        assert {key: first_audit[key] for key in stable_keys} == {
            key: second_audit[key] for key in stable_keys
        }
    with (
        np.load(first_folder / "outer-1-predictions.npz") as a,
        np.load(second_folder / "outer-1-predictions.npz") as b,
    ):
        assert a.files == b.files
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    assert first_fold["procedures"] != second_fold["procedures"]
    inner = json.loads((first_folder / "outer-1-inner.json").read_text())
    for fit in inner["fits"]:
        assert held.isdisjoint(fit["fit_audit"]["fit_groups"])
        assert set(fit["fit_groups"]).isdisjoint(fit["assessment_groups"])
    assert set(partitions) == {"train", "validation"}
    assert first["activated"] is False
    assert json.loads((first_root / "models/active.json").read_text()) == {"preserve": True}
    assert json.loads((first_root / "models/latest.json").read_text()) == {"preserve": True}
