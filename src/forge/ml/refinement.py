"""Paired grouped development evaluation of a causal score median; never activates."""

import argparse
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from forge.config import PROJECT_ROOT
from forge.ml.development import development_streams, merge_streams
from forge.ml.generalization import (
    aggregate,
    assess,
    choose_operating_point,
    fit_candidate,
    group_folds,
    pool_description,
    save_predictions,
    subset,
    validate_pool,
)
from forge.ml.training import digest, write_json


def causal_median(scores, ready, timestamps, *, window, max_gap_seconds):
    """Require a full past/current ready window, resetting at gaps or unavailability."""
    scores = np.asarray(scores, dtype=float)
    ready = np.asarray(ready, dtype=bool)
    times = pd.DatetimeIndex(timestamps)
    if (
        not isinstance(window, int)
        or isinstance(window, bool)
        or window < 1
        or not np.isfinite(max_gap_seconds)
        or max_gap_seconds <= 0
        or len(scores) != len(ready)
        or len(times) != len(scores)
    ):
        raise ValueError("Invalid score-window inputs.")
    if not len(scores):
        return scores.copy(), ready.copy()
    deltas = np.diff(times.as_unit("ns").asi8) / 1e9
    if times.isna().any() or (deltas <= 0).any():
        raise ValueError("Score timestamps must be present and strictly increasing.")
    valid = ready & np.isfinite(scores)
    resets = np.r_[True, (deltas > max_gap_seconds) | ~valid[:-1]] | ~valid
    segments = np.cumsum(resets)
    result = (
        pd.Series(np.where(valid, scores, np.nan))
        .groupby(segments)
        .transform(lambda part: part.rolling(window, min_periods=window).median())
        .to_numpy()
    )
    available = valid & np.isfinite(result)
    return np.where(available, result, 0.0), available


def score_pair(detector, streams, config, *, timings=None):
    """Transform each source before overlap deduplication, preserving serving context."""
    raw, smoothed = {}, {}
    elapsed = {"raw_score_seconds": 0.0, "median_seconds": 0.0}
    for name, frame in streams.items():
        started = perf_counter()
        scores, ready = detector.score(frame), detector.readiness(frame)
        elapsed["raw_score_seconds"] += perf_counter() - started
        base = frame.copy()
        base["_score"], base["_ready"] = scores, ready
        raw[name] = base
        candidate = frame.copy()
        started = perf_counter()
        candidate["_score"], candidate["_ready"] = causal_median(
            scores,
            ready,
            frame.datetime,
            window=config["smoothing_readings"],
            max_gap_seconds=config["max_gap_seconds"],
        )
        elapsed["median_seconds"] += perf_counter() - started
        smoothed[name] = candidate
    result = {}
    for procedure, scored in [("current_hgb", raw), ("median15_hgb", smoothed)]:
        groups, _ = merge_streams(scored)
        result[procedure] = {
            "scores": {name: frame["_score"].to_numpy() for name, frame in groups.items()},
            "ready": {name: frame["_ready"].to_numpy(dtype=bool) for name, frame in groups.items()},
        }
    if timings is not None:
        timings.update(elapsed)
    return result


def select_inner(streams, config):
    folds = group_folds(streams, config["inner_folds"], config["seed"])
    groups, _ = merge_streams(streams)
    predictions = {name: {"scores": {}, "ready": {}} for name in config["procedures"]}
    fits = []
    for index, fold in enumerate(folds):
        started = perf_counter()
        detector, audit = fit_candidate(
            subset(streams, fold["fit_groups"]), config["candidates"][0], config
        )
        paired = score_pair(detector, subset(streams, fold["assessment_groups"]), config)
        for name, prediction in paired.items():
            if set(predictions[name]["scores"]) & set(prediction["scores"]):
                raise ValueError("Duplicate inner assessment group.")
            for field in ("scores", "ready"):
                predictions[name][field].update(prediction[field])
        fits.append({**fold, "fit_audit": audit, "elapsed_seconds": perf_counter() - started})
        print(f"  inner {index + 1}/{len(folds)} complete", flush=True)
    choices, scans = {}, {}
    for name, prediction in predictions.items():
        if set(prediction["scores"]) != set(groups):
            raise ValueError("Incomplete inner assessment coverage.")
        choices[name], scans[name] = choose_operating_point(groups, **prediction, config=config)
    return choices, {"folds": folds, "fits": fits, "threshold_scans": scans}


def run(root=PROJECT_ROOT):
    root = Path(root)
    config_path = root / "configs/refinement-v1.json"
    config = json.loads(config_path.read_text())
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    folder = root / "artifacts/refinement" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    # Freeze the configuration before data access or fitting.
    write_json(folder / "config.json", config)
    protected = [
        "models/active.json",
        "models/latest.json",
        "data/skab-splits.json",
        "data/skab-inventory.json",
        "docs/evaluation-results.json",
    ]
    preserved = {name: digest(root / name) for name in protected if (root / name).exists()}
    provenance = {
        "config_sha256": digest(config_path),
        "preserved_sha256": preserved,
        "sklearn_version": version("scikit-learn"),
        "source_sha256": {
            name: digest(root / "src/forge/ml" / name)
            for name in [
                "refinement.py",
                "generalization.py",
                "development.py",
                "operating.py",
                "metrics.py",
            ]
        },
    }
    started = perf_counter()
    streams = {**development_streams("train", root), **development_streams("validation", root)}
    validate_pool(streams)
    folds = group_folds(streams, config["outer_folds"], config["seed"])
    plan = {"pool": pool_description(streams), "outer_folds": folds}
    write_json(folder / "plan.json", plan)
    write_json(folder / "provenance.json", provenance)
    metrics = {name: {} for name in config["procedures"]}
    records = []
    with threadpool_limits(limits=1):
        for index, fold in enumerate(folds):
            fold_start = perf_counter()
            print(f"Outer {index + 1}/{len(folds)}: {fold['assessment_groups']}", flush=True)
            fitting = subset(streams, fold["fit_groups"])
            choices, inner = select_inner(fitting, config)
            write_json(folder / f"outer-{index + 1}-selection.json", choices)
            write_json(folder / f"outer-{index + 1}-inner.json", inner)
            fit_start = perf_counter()
            detector, audit = fit_candidate(fitting, config["candidates"][0], config)
            fit_seconds = perf_counter() - fit_start
            assessment = subset(streams, fold["assessment_groups"])
            groups, _ = merge_streams(assessment)
            score_start = perf_counter()
            scoring_times = {}
            predictions = score_pair(detector, assessment, config, timings=scoring_times)
            score_seconds = perf_counter() - score_start
            outcomes = {}
            for name, prediction in predictions.items():
                outcomes[name] = assess(groups, **prediction, choice=choices[name])
                if set(metrics[name]) & set(outcomes[name]["per_group"]):
                    raise ValueError("Duplicate outer assessment group.")
                metrics[name].update(outcomes[name]["per_group"])
                print(
                    f"  {name}: {outcomes[name]['pooled']['precision']:.4f} precision, "
                    f"{outcomes[name]['pooled']['recall']:.4f} recall",
                    flush=True,
                )
            record = {
                **fold,
                "selection": choices,
                "fit_audit": audit,
                "procedures": outcomes,
                "fit_seconds": fit_seconds,
                "paired_score_seconds": score_seconds,
                **scoring_times,
                "elapsed_seconds": perf_counter() - fold_start,
            }
            records.append(record)
            save_predictions(folder / f"outer-{index + 1}-predictions.npz", predictions)
            write_json(folder / f"outer-{index + 1}-results.json", record)
    expected = {item["group"] for item in plan["pool"]}
    if any(set(values) != expected for values in metrics.values()):
        raise ValueError("Incomplete outer assessment coverage.")
    if any(digest(root / name) != value for name, value in preserved.items()):
        raise ValueError("Protected files changed.")
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "config": config,
        **plan,
        "provenance": provenance,
        "fold_results": records,
        "procedures": {
            name: {"pooled": aggregate(values), "per_group": values}
            for name, values in metrics.items()
        },
        "elapsed_seconds": perf_counter() - started,
        "activated": False,
    }
    write_json(folder / "results.json", result)
    print(f"Results: {folder}", flush=True)
    return folder, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    run(parser.parse_args().root)
