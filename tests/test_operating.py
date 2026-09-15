"""Causality, startup availability and selection metric equivalence."""

import numpy as np
import pandas as pd
import pytest

from forge.data.datasets import FEATURES
from forge.ml.development import fast_alerts, score_streams, selection_metrics
from forge.ml.metrics import causal_alerts, evaluate
from forge.ml.operating import OperatingDetector, causal_features


def recording(length=100):
    rng = np.random.default_rng(10)
    frame = pd.DataFrame(rng.normal(size=(length, len(FEATURES))), columns=FEATURES)
    frame["datetime"] = pd.date_range("2020-01-01", periods=length, freq="s")
    frame["anomaly"] = (np.arange(length) >= length // 2).astype(int)
    return frame


@pytest.mark.parametrize(
    "mode", ["raw", "causal", "relative", "relative_noise", "relative_fraction"]
)
def test_future_rows_cannot_change_past_features(mode):
    frame = recording()
    expected, available = causal_features(frame, np.ones(8), mode)
    for end in (1, 20, 59, 60, 61, 85):
        actual, ready = causal_features(frame.iloc[:end], np.ones(8), mode)
        np.testing.assert_allclose(actual, expected[:end])
        np.testing.assert_array_equal(ready, available[:end])


def test_relative_representation_does_not_use_annotation_or_identity():
    frame = recording()
    original, ready = causal_features(frame, np.ones(8), "relative")
    frame["anomaly"] = 1 - frame.anomaly
    frame["changepoint"] = 1
    frame["experiment_id"] = "a-different-experiment"
    modified, _ = causal_features(frame, np.ones(8), "relative")
    np.testing.assert_array_equal(original, modified)
    assert not ready[:60].any() and ready[60:].all()


def test_gaps_reset_rolling_changes_but_preserve_initial_reference():
    frame = recording()
    frame.loc[70:, "datetime"] += pd.Timedelta(seconds=10)
    features, ready = causal_features(frame, np.ones(8), "relative")
    assert ready[70]
    np.testing.assert_allclose(
        features[70, :8],
        frame.loc[70, FEATURES].to_numpy(dtype=float)
        - frame.loc[:59, FEATURES].median().to_numpy(),
    )
    np.testing.assert_array_equal(features[70, 8:], 0)


def test_fast_alerts_match_existing_rule_with_gaps_and_unavailable_rows():
    frame = recording()
    frame.loc[40:, "datetime"] += pd.Timedelta(seconds=10)
    rng = np.random.default_rng(42)
    scores = rng.uniform(size=len(frame))
    ready = np.arange(len(frame)) > 10
    ready[45] = False
    for threshold in (0.1, 0.5, 0.9):
        for persistence in (1, 3, 5):
            expected = causal_alerts(
                np.where(ready, scores, np.nan), frame.datetime, threshold, persistence, 2
            )
            actual = fast_alerts(scores, frame.datetime, ready, threshold, persistence, 2)
            np.testing.assert_array_equal(actual, expected)


def test_available_baseline_selection_metrics_match_reference_evaluation():
    frame = recording()
    frame.loc[25, "anomaly"] = -1
    frame.loc[75:, "datetime"] += pd.Timedelta(seconds=10)
    groups = {"synthetic": frame}
    scores = {"synthetic": np.tile([0.0, 1.0, 1.0, 1.0, 1.0], 20)}
    ready = {"synthetic": np.ones(100, dtype=bool)}
    policy = {"persistence": 3, "max_gap_seconds": 2}
    expected = evaluate(groups, scores, 0.5, policy)["pooled"]
    actual = selection_metrics(groups, scores, ready, 0.5, policy)
    for field in (
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "events",
        "detected_events",
        "false_alarm_onsets",
    ):
        assert actual[field] == expected[field]
    assert (
        actual["false_onsets_per_available_normal_hour"] == expected["false_alarms_per_normal_hour"]
    )


def test_startup_anomalies_are_misses_and_normal_exposure_is_not_padded():
    frame = recording(10)
    groups = {"synthetic": frame}
    scores = {"synthetic": np.ones(10)}
    ready = {"synthetic": np.arange(10) >= 7}
    metrics = selection_metrics(
        groups, scores, ready, 0.5, {"persistence": 3, "max_gap_seconds": 2}
    )
    assert metrics["tp"] == 1 and metrics["fn"] == 4
    assert metrics["unavailable_anomaly_rows"] == 2
    assert metrics["available_normal_rows"] == 0
    assert metrics["fpr_available_normal"] is None


def test_overlapping_source_files_keep_separate_startup_references():
    class ReferenceProbe(OperatingDetector):
        def score(self, frame):
            values, ready = self.transform(frame)
            return np.where(ready, values[:, 0], 0)

    full = recording(160)
    full["_group"] = "overlap"
    first, second = full.iloc[:100].copy(), full.iloc[80:].copy()
    detector = ReferenceProbe(
        "probe", np.zeros(8), np.ones(8), ["test"] * 8, representation="relative"
    )
    scores, ready = score_streams(detector, {"first": first, "second": second})
    expected = np.r_[detector.score(first), detector.score(second)[20:]]
    np.testing.assert_array_equal(scores["overlap"], expected)
    assert not ready["overlap"][100:140].any()
    assert ready["overlap"][140:].all()


def test_fitted_classifier_roundtrip_and_prefix_predictions():
    import pickle

    from forge.ml.detectors import Detector
    from forge.ml.development import fit_operating

    frame = recording(150)
    frame["_group"] = "synthetic"
    config = {
        "seed": 42,
        "reference_readings": 60,
        "rolling_readings": 20,
        "max_gap_seconds": 2,
        "gradient_boosting": {"max_iter": 5, "min_samples_leaf": 5, "early_stopping": False},
    }
    detector, audit = fit_operating(
        Detector.fit(frame.iloc[:70]),
        {"synthetic": frame},
        "relative",
        config,
        estimator_kind="gradient_boosting",
    )
    assert audit["fit_rows"] == 90
    loaded = pickle.loads(pickle.dumps(detector))
    scores = detector.score(frame)
    np.testing.assert_array_equal(loaded.score(frame), scores)
    np.testing.assert_array_equal(loaded.score(frame.iloc[:100]), scores[:100])
    assert not scores[:60].any()


def test_subsecond_sampling_cannot_silently_inflate_exposure():
    frame = recording(10)
    frame["datetime"] = pd.date_range("2020-01-01", periods=10, freq="500ms")
    with pytest.raises(ValueError, match="sampling"):
        selection_metrics(
            {"x": frame},
            {"x": np.ones(10)},
            {"x": np.ones(10, dtype=bool)},
            0.5,
            {"persistence": 3, "max_gap_seconds": 2},
        )
