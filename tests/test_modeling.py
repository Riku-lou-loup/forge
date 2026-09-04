"""Behavioral checks; fixtures are synthetic and are not benchmark results."""

import numpy as np
import pandas as pd
import pytest

from forge.data.datasets import FEATURES, merge_group, sensor_matrix
from forge.ml.detectors import Detector
from forge.ml.metrics import causal_alerts, evaluate_group, select_threshold


def readings(values, labels=None, times=None):
    values = np.asarray(values, dtype=float)
    frame = pd.DataFrame({name: values + i for i, name in enumerate(FEATURES)})
    frame["datetime"] = (
        pd.to_datetime(times)
        if times is not None
        else pd.date_range("2020-01-01", periods=len(values), freq="s")
    )
    if labels is not None:
        frame["anomaly"] = labels
    return frame


def test_labels_never_enter_features():
    frame = readings([1, 2, 3], [0, 1, 0])
    original = sensor_matrix(frame)
    frame["anomaly"] = 999
    frame["changepoint"] = 555
    np.testing.assert_array_equal(sensor_matrix(frame), original)


def test_zero_mad_uses_iqr_and_does_not_mutate_input():
    frame = readings([0, 0, 0, 1, 2])
    before = frame.copy(deep=True)
    detector = Detector.fit(frame)
    assert detector.scale_methods == ["IQR"] * 8
    np.testing.assert_allclose(detector.scale, 1 / 1.349)
    assert detector.score(readings([100]))[0] > detector.score(readings([0]))[0]
    pd.testing.assert_frame_equal(frame, before)


def test_constant_sensor_requires_explicit_policy():
    with pytest.raises(ValueError, match="sensor-specific"):
        Detector.fit(readings([1, 1, 1]))


def test_persistence_is_causal_and_resets_at_gaps():
    times = pd.to_datetime(
        [
            "2020-01-01 00:00:00",
            "2020-01-01 00:00:01",
            "2020-01-01 00:00:02",
            "2020-01-01 00:00:10",
            "2020-01-01 00:00:11",
        ]
    )
    assert causal_alerts([2] * 5, times, 1).tolist() == [False, False, True, False, False]
    assert causal_alerts([2, 2], times[:2], 1).tolist() == [False, False]
    assert not causal_alerts([1] * 5, times, 1).any()


def test_duplicate_conflicting_annotations_are_not_voted_away():
    first = readings([1, 2, 3], [0, 0, 1])
    second = first.copy()
    second.loc[1, "anomaly"] = 1
    merged, audit = merge_group([first, second])
    assert merged.anomaly.tolist() == [0, -1, 1]
    assert audit["duplicate_rows"] == 3
    assert audit["ambiguous_rows"] == 1


def test_alert_before_event_does_not_count_as_detection():
    frame = readings(range(6), [0, 0, 0, 1, 1, 1])
    result = evaluate_group(frame, np.ones(6), 0.5, {"persistence": 3, "max_gap_seconds": 2})
    assert result["events"] == 1 and result["detected_events"] == 0
    assert result["false_alarm_onsets"] == 1
    assert result["recall"] == 1  # Point recall alone would conceal this failure.


def test_unknown_annotation_breaks_alerts_and_is_excluded():
    frame = readings(range(5), [0, 1, -1, 1, 1])
    result = evaluate_group(frame, np.ones(5), 0.5, {"persistence": 3, "max_gap_seconds": 2})
    assert result["rows"] == 4 and result["events"] == 2
    assert result["tp"] == 0


def test_normal_exposure_does_not_depend_on_pandas_timestamp_resolution():
    frame = readings(range(6), [0, 0, 0, 1, 1, 1])
    frame["datetime"] = frame.datetime.dt.as_unit("us")
    result = evaluate_group(frame, np.ones(6), 0.5, {"persistence": 3, "max_gap_seconds": 2})
    assert result["normal_hours"] == pytest.approx(3 / 3600)
    assert result["false_alarms_per_normal_hour"] == pytest.approx(1200)


def test_validation_selection_can_reject_normal_scores():
    frame = readings(range(16), [0] * 12 + [1] * 4)
    groups = {"synthetic": frame}
    scores = {"synthetic": np.array([0] * 12 + [2] * 4)}
    threshold, result = select_threshold(
        groups, scores, {"persistence": 3, "max_gap_seconds": 2}, 5
    )
    assert 0 <= threshold < 2
    assert result["pooled"]["fp"] == 0 and result["pooled"]["tp"] == 2


def test_forest_is_reproducible_and_scores_novel_values():
    frame = readings(np.linspace(0, 10, 100))
    one = Detector.fit(frame, max_samples=32, n_estimators=16)
    two = Detector.fit(frame, max_samples=32, n_estimators=16)
    query = readings([5, 100])
    np.testing.assert_array_equal(one.score(query), two.score(query))
    assert one.score(query)[1] > one.score(query)[0]
