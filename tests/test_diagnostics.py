"""Separate raw score statistics from causal alert rates in diagnostics."""

import numpy as np
import pandas as pd
import pytest

from forge.ml.diagnostics import histograms, summarize_group


def test_raw_threshold_exceedance_is_not_persistent_false_positive_rate():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=6, freq="s"),
            "anomaly": [0, 0, 0, 1, 1, 1],
        }
    )
    result = summarize_group(frame, np.ones(6), 0.5, {"persistence": 3, "max_gap_seconds": 2})
    assert result["normal_above_threshold"] == 1
    assert result["fpr"] == pytest.approx(1 / 3)
    assert result["recall"] == 1
    assert result["roc_auc"] == 0.5


def test_histogram_normalization_excludes_unknown_and_handles_class_imbalance():
    groups = {"synthetic": pd.DataFrame({"anomaly": [0, 0, 0, 1, -1]})}
    scores = {"synthetic": np.array([0.1, 0.2, 0.3, 0.9, 100])}
    binned = histograms(groups, scores, bins=4)
    assert binned.groupby("label")["count"].sum().to_dict() == {0: 3, 1: 1}
    assert binned.groupby("label")["percent"].sum().to_dict() == {0: 100.0, 1: 100.0}
    assert binned.right.max() == 0.9


def test_single_class_is_not_reported_as_zero_recall_or_auc():
    frame = pd.DataFrame(
        {"datetime": pd.date_range("2020-01-01", periods=3, freq="s"), "anomaly": [0, 0, 0]}
    )
    result = summarize_group(frame, [0.1, 0.2, 0.3], 0.5, {"persistence": 3, "max_gap_seconds": 2})
    assert result["recall"] is None and result["roc_auc"] is None
    assert result["anomaly_p50"] is None and result["fpr"] == 0
