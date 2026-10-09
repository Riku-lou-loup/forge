import numpy as np
import pandas as pd
import pytest

from forge.ml.refinement import causal_median, select_inner


def test_median_is_causal_and_resets_on_gap_and_invalid_score():
    times = pd.to_datetime(
        [
            "2020-01-01 00:00:00",
            "2020-01-01 00:00:01",
            "2020-01-01 00:00:02",
            "2020-01-01 00:00:10",
            "2020-01-01 00:00:11",
            "2020-01-01 00:00:12",
            "2020-01-01 00:00:13",
            "2020-01-01 00:00:14",
        ]
    )
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan, 8.0])
    result, ready = causal_median(scores, np.ones(8, bool), times, window=3, max_gap_seconds=2)
    assert ready.tolist() == [False, False, True, False, False, True, False, False]
    assert result[2] == 2 and result[5] == 5
    future = scores.copy()
    future[3:] = 999
    altered, _ = causal_median(future, np.ones(8, bool), times, window=3, max_gap_seconds=2)
    np.testing.assert_array_equal(result[:3], altered[:3])


def test_readiness_interrupts_smoothing_and_window_one_reproduces_scores():
    times = pd.date_range("2020", periods=6, freq="s")
    scores = np.arange(6, dtype=float)
    available = np.array([True, True, False, True, True, True])
    result, ready = causal_median(scores, available, times, window=3, max_gap_seconds=2)
    assert ready.tolist() == [False, False, False, False, False, True]
    assert result[-1] == 4
    identity, identity_ready = causal_median(scores, available, times, window=1, max_gap_seconds=2)
    np.testing.assert_array_equal(identity_ready, available)
    np.testing.assert_array_equal(identity[available], scores[available])


def test_timestamp_order_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        causal_median(
            [1, 2], [True, True], pd.to_datetime(["2020", "2020"]), window=1, max_gap_seconds=2
        )


def test_inner_selection_only_fits_complement_and_selects_complete_oof(monkeypatch):
    import forge.ml.refinement as module

    streams = {name: pd.DataFrame({"_group": [name]}) for name in ["a", "b", "c", "d"]}
    fit_calls = []
    monkeypatch.setattr(module, "merge_streams", lambda values: (values, {}))

    def fit(values, candidate, config):
        names = set(values)
        fit_calls.append(names)
        return names, {"fit_groups": sorted(names)}

    def score(fit_names, values, config):
        assert fit_names.isdisjoint(values)
        assert fit_names | set(values) == set(streams)
        return {
            name: {
                "scores": {key: np.array([1.0]) for key in values},
                "ready": {key: np.array([True]) for key in values},
            }
            for name in config["procedures"]
        }

    def choose(groups, scores, ready, config):
        assert set(groups) == set(scores) == set(ready) == set(streams)
        return {"threshold": 0.5}, []

    monkeypatch.setattr(module, "fit_candidate", fit)
    monkeypatch.setattr(module, "score_pair", score)
    monkeypatch.setattr(module, "choose_operating_point", choose)
    choices, audit = select_inner(
        streams,
        {
            "inner_folds": 2,
            "seed": 42,
            "procedures": ["current_hgb", "median15_hgb"],
            "candidates": [{"id": "current_hgb"}],
        },
    )
    assert len(fit_calls) == 2 and len(choices) == 2 and len(audit["fits"]) == 2


@pytest.mark.parametrize("window,gap", [(0, 2), (1.5, 2), (True, 2), (3, 0), (3, float("nan"))])
def test_invalid_window_and_gap_rejected(window, gap):
    with pytest.raises(ValueError, match="Invalid"):
        causal_median(
            [1.0], [True], pd.date_range("2020", periods=1), window=window, max_gap_seconds=gap
        )


def test_outer_labels_do_not_change_that_folds_model_or_selection(tmp_path, monkeypatch):
    import copy
    import json
    from pathlib import Path

    import forge.ml.refinement as module
    from forge.config import PROJECT_ROOT
    from forge.data.datasets import FEATURES
    from forge.ml.training import write_json

    config = json.loads((PROJECT_ROOT / "configs/refinement-v1.json").read_text())
    config.update(
        outer_folds=2,
        inner_folds=2,
        reference_readings=5,
        rolling_readings=3,
        threshold_quantiles=5,
        smoothing_readings=3,
    )
    config["candidates"][0].update(max_iter=3, max_leaf_nodes=3, min_samples_leaf=3)
    streams = {}
    for index in range(6):
        labels = ((np.arange(40) >= 15) & (np.arange(40) < 30)).astype(int)
        frame = pd.DataFrame(
            np.random.default_rng(index).normal(size=(40, 8)) + labels[:, None] * 3,
            columns=FEATURES,
        )
        frame["datetime"] = pd.date_range("2020", periods=40, freq="s") + pd.Timedelta(days=index)
        frame["anomaly"] = labels
        frame["_group"] = f"g{index}"
        streams[f"s{index}"] = frame
    original = copy.deepcopy(streams)
    calls = []

    def load(partition, root):
        assert partition in {"train", "validation"}
        calls.append(partition)
        return {
            name: frame
            for index, (name, frame) in enumerate(streams.items())
            if (index < 3) == (partition == "train")
        }

    monkeypatch.setattr(module, "development_streams", load)

    def workspace(name):
        root = tmp_path / name
        (root / "configs").mkdir(parents=True)
        write_json(root / "configs/refinement-v1.json", config)
        for relative in [
            "models/active.json",
            "models/latest.json",
            "data/skab-splits.json",
            "data/skab-inventory.json",
            "docs/evaluation-results.json",
        ]:
            path = root / relative
            path.parent.mkdir(exist_ok=True)
            write_json(path, {"untouched": True})
        for name in [
            "refinement.py",
            "generalization.py",
            "development.py",
            "operating.py",
            "metrics.py",
        ]:
            path = root / "src/forge/ml" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((PROJECT_ROOT / "src/forge/ml" / name).read_bytes())
        return root

    folder, first = module.run(workspace("first"))
    held = set(first["fold_results"][0]["assessment_groups"])
    for frame in streams.values():
        if frame["_group"].iloc[0] in held:
            frame["anomaly"] = 1 - frame.anomaly
    _, second = module.run(workspace("second"))
    assert first["fold_results"][0]["selection"] == second["fold_results"][0]["selection"]
    assert first["fold_results"][0]["fit_audit"] == second["fold_results"][0]["fit_audit"]
    a = np.load(folder / "outer-1-predictions.npz")
    b = np.load(
        Path(tmp_path / "second/artifacts/refinement" / second["run_id"])
        / "outer-1-predictions.npz"
    )
    for key in a.files:
        np.testing.assert_array_equal(a[key], b[key])
    assert first["fold_results"][0]["procedures"] != second["fold_results"][0]["procedures"]
    assert set(calls) == {"train", "validation"}
    assert first["activated"] is False
    assert json.loads((tmp_path / "first/models/active.json").read_text()) == {"untouched": True}
    assert set(first["procedures"]["current_hgb"]["per_group"]) == {f"g{i}" for i in range(6)}
    for frame in original.values():
        assert len(frame) == 40


def test_score_median_restarts_at_source_boundary_before_merge():
    from forge.data.datasets import FEATURES
    from forge.ml.refinement import score_pair

    class FixedDetector:
        def score(self, frame):
            return frame[FEATURES[0]].to_numpy()

        def readiness(self, frame):
            return np.ones(len(frame), dtype=bool)

    streams = {}
    for index in range(2):
        values = np.arange(index * 5, index * 5 + 5, dtype=float)
        frame = pd.DataFrame({name: values for name in FEATURES})
        frame["datetime"] = pd.date_range("2020", periods=5, freq="s") + pd.Timedelta(
            seconds=index * 5
        )
        frame["anomaly"] = 0
        frame["_group"] = "same-overlap-group"
        streams[str(index)] = frame
    result = score_pair(FixedDetector(), streams, {"smoothing_readings": 3, "max_gap_seconds": 2})
    assert result["median15_hgb"]["ready"]["same-overlap-group"].tolist() == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
    ]
