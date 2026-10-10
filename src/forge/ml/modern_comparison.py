"""Matched nested development comparison of HGB, ordered CatBoost and compact TabM."""

import argparse
import json
import pickle
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from forge.config import PROJECT_ROOT
from forge.data.datasets import FEATURES, OBSERVATION_KEY
from forge.ml.detectors import Detector
from forge.ml.development import development_streams, merge_streams, score_streams
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
from forge.ml.operating import OperatingDetector
from forge.ml.training import digest, write_json


def fhalf(metrics):
    denominator = 1.25 * metrics["tp"] + 0.25 * metrics["fn"] + metrics["fp"]
    return 1.25 * metrics["tp"] / denominator if denominator else 0.0


def precision_key(choice):
    metrics = choice["metrics"]
    onsets = metrics["false_onsets_per_available_normal_hour"]
    return (
        fhalf(metrics),
        metrics["precision"],
        -(onsets if onsets is not None else float("inf")),
        metrics["recall"],
        choice["threshold"],
    )


def choose_precision(scan):
    """No recall floor: a no-alert policy has zero F0.5, never perfect precision."""
    if not scan:
        raise ValueError("Cannot select from an empty threshold scan.")
    best = max(scan, key=precision_key)
    return {
        **{key: value for key, value in best.items() if key != "feasible"},
        "status": "f0.5_selected",
        "objective_f0_5": fhalf(best["metrics"]),
    }


def training_table(detector, streams):
    """Reuse the existing source-first causal features, deduplication and group weights."""
    inputs, frames = [], []
    for frame in streams.values():
        features, ready = detector.transform(frame)
        rows = frame[[*OBSERVATION_KEY, "anomaly", "_group"]].copy()
        rows["_ready"] = ready
        frames.append(rows)
        inputs.append(features)
    if not frames:
        raise ValueError("No training streams.")
    rows = pd.concat(frames, ignore_index=True)
    ambiguous = rows.groupby(OBSERVATION_KEY, dropna=False).anomaly.transform("nunique").gt(1)
    eligible = rows["_ready"] & rows.anomaly.ge(0) & ~ambiguous & ~rows.duplicated(OBSERVATION_KEY)
    x = np.concatenate(inputs)[eligible.to_numpy()]
    retained = rows.loc[eligible]
    y = retained.anomaly.to_numpy()
    if not len(x) or set(np.unique(y)) != {0, 1} or not np.isfinite(x).all():
        raise ValueError("Training requires finite ready features and both binary classes.")
    weights = 1 / retained.groupby("_group").anomaly.transform("size").to_numpy()
    weights *= len(weights) / weights.sum()
    audit = {
        "fit_rows": len(x),
        "positive_training_targets": int(y.sum()),
        "group_weighting": "Equal total weight per training leakage group",
        "representation": "relative",
        "feature_count": x.shape[1],
    }
    return x, y, weights, audit


def fit_modern(streams, candidate, config):
    if candidate["family"] == "hgb":
        return fit_candidate(streams, {"id": candidate["id"], **candidate["parameters"]}, config)
    groups, _ = merge_streams(streams)
    normal = pd.concat([frame.loc[frame.anomaly.eq(0), FEATURES] for frame in groups.values()])
    if normal.empty:
        raise ValueError("Fold has no normal observations for scaling.")
    reference = Detector.fit(normal)
    detector = OperatingDetector(
        candidate["id"],
        reference.center,
        reference.scale,
        reference.scale_methods,
        estimator=None,
        representation="relative",
        classifier=True,
        reference_readings=config["reference_readings"],
        rolling_readings=config["rolling_readings"],
        max_gap_seconds=config["max_gap_seconds"],
    )
    x, y, weights, audit = training_table(detector, streams)
    if candidate["family"] == "catboost":
        from catboost import CatBoostClassifier

        estimator = CatBoostClassifier(
            **candidate["parameters"],
            random_seed=config["seed"],
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
        estimator.fit(x, y, sample_weight=weights)
        boosting = estimator.get_param("boosting_type")
        if boosting != "Ordered":
            raise ValueError("This protocol requires explicitly configured ordered boosting.")
        training = {"iterations": estimator.tree_count_, "boosting_type": boosting}
    elif candidate["family"] == "tabm":
        from forge.ml.tabm_estimator import fit_tabm

        estimator = fit_tabm(x, y, weights, candidate["parameters"], config["seed"])
        training = estimator.training_audit
    else:
        raise ValueError("Unknown modern model family.")
    detector.estimator = estimator
    return detector, {
        **audit,
        "training": training,
        "fit_groups": sorted(groups),
        "fit_sources": list(streams),
        "normal_scaling_rows": len(normal),
        "center": detector.center.tolist(),
        "scale": detector.scale.tolist(),
    }


def assess_ranking(groups, scores, ready):
    """Raw-score AP uses available labeled readings, never thresholded alerts."""
    per_group, labels, values = {}, [], []
    for name, frame in groups.items():
        score = np.asarray(scores[name], dtype=float)
        available = np.asarray(ready[name], dtype=bool) & frame.anomaly.ge(0).to_numpy()
        if len(score) != len(frame) or not np.isfinite(score[available]).all():
            raise ValueError("Invalid assessment scores.")
        y, s = frame.anomaly.to_numpy()[available], score[available]
        both = len(np.unique(y)) == 2
        per_group[name] = {
            "average_precision": float(average_precision_score(y, s)) if both else None,
            "available_labeled_rows": len(y),
            "positive_rows": int((y == 1).sum()),
        }
        labels.append(y)
        values.append(s)
    y, s = np.concatenate(labels), np.concatenate(values)
    return {
        "per_group": per_group,
        "pooled_within_fold_average_precision": float(average_precision_score(y, s))
        if len(np.unique(y)) == 2
        else None,
        "zero_assessable_groups": [
            name for name, row in per_group.items() if not row["available_labeled_rows"]
        ],
    }


def select_inner(streams, config):
    folds = group_folds(streams, config["inner_folds"], config["seed"])
    groups, _ = merge_streams(streams)
    predictions = {c["id"]: {"scores": {}, "ready": {}} for c in config["candidates"]}
    fits = []
    for index, fold in enumerate(folds):
        fitting, assessment = (
            subset(streams, fold["fit_groups"]),
            subset(streams, fold["assessment_groups"]),
        )
        for candidate in config["candidates"]:
            started = perf_counter()
            detector, audit = fit_modern(fitting, candidate, config)
            scores, ready = score_streams(detector, assessment)
            stored = predictions[candidate["id"]]
            if set(stored["scores"]) & set(scores):
                raise ValueError("Duplicate inner assessment group.")
            stored["scores"].update(scores)
            stored["ready"].update(ready)
            fits.append(
                {
                    "fold": index + 1,
                    "candidate": candidate["id"],
                    **fold,
                    "fit_audit": audit,
                    "elapsed_seconds": perf_counter() - started,
                }
            )
        print(f"  inner {index + 1}/{len(folds)} all families fitted", flush=True)
    choices, scans, historical = {}, {}, {}
    for name, prediction in predictions.items():
        if set(prediction["scores"]) != set(groups):
            raise ValueError("Incomplete inner assessment coverage.")
        historical[name], scans[name] = choose_operating_point(groups, **prediction, config=config)
        choices[name] = choose_precision(scans[name])
    selected = {}
    for procedure, definition in config["procedures"].items():
        if definition["selection"] == "historical":
            name = definition["candidates"][0]
            choice = historical[name]
        else:
            name = max(definition["candidates"], key=lambda key: precision_key(choices[key])[:-1])
            choice = choices[name]
        selected[procedure] = {"candidate": name, **choice}
    return selected, {
        "folds": folds,
        "fits": fits,
        "threshold_scans": scans,
        "candidate_precision_choices": choices,
        "scan_feasibility_note": "The feasible field records historical recall floors only; ignored by F0.5 selection.",
    }


def summarize(per_group):
    metrics = aggregate(per_group)
    return {
        **metrics,
        "f0_5": fhalf(metrics),
        "positive_groups": sum(row["tp"] + row["fn"] > 0 for row in per_group.values()),
        "zero_recall_positive_groups": [
            name for name, row in per_group.items() if row["tp"] == 0 and row["fn"] > 0
        ],
        "false_positive_sample_exposure_hours": metrics["fp"] / 3600,
    }


def run(root=PROJECT_ROOT):
    root = Path(root)
    config_path = root / "configs/modern-comparison-v1.json"
    config = json.loads(config_path.read_text())
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    folder = root / "artifacts/modern" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "config.json", config)
    protected = [
        "models/active.json",
        "models/latest.json",
        "data/skab-splits.json",
        "data/skab-inventory.json",
        "docs/evaluation-results.json",
    ]
    preserved = {name: digest(root / name) for name in protected if (root / name).exists()}
    source_paths = [
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
    source_paths += ["src/forge/data/datasets.py", "src/forge/data/partitions.py"]
    provenance = {
        "config_sha256": digest(config_path),
        "preserved_sha256": preserved,
        "python_version": sys.version,
        "versions": {
            name: version(name)
            for name in [
                "numpy",
                "pandas",
                "scikit-learn",
                "catboost",
                "tabm",
                "torch",
                "rtdl_num_embeddings",
            ]
        },
        "source_sha256": {name: digest(root / name) for name in source_paths},
    }
    write_json(folder / "provenance.json", provenance)
    started = perf_counter()
    streams = {**development_streams("train", root), **development_streams("validation", root)}
    validate_pool(streams)
    folds = group_folds(streams, config["outer_folds"], config["seed"])
    plan = {"pool": pool_description(streams), "outer_folds": folds}
    write_json(folder / "plan.json", plan)
    definitions = {candidate["id"]: candidate for candidate in config["candidates"]}
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
            # Freeze every choice before exposing outer assessment labels to reporting.
            assessment = subset(streams, fold["assessment_groups"])
            groups, _ = merge_streams(assessment)
            fitted, predictions, outcomes, fit_audits = {}, {}, {}, {}
            for procedure, choice in choices.items():
                name = choice["candidate"]
                if name not in fitted:
                    fit_start = perf_counter()
                    detector, audit = fit_modern(fitting, definitions[name], config)
                    fit_seconds = perf_counter() - fit_start
                    score_start = perf_counter()
                    scores, ready = score_streams(detector, assessment)
                    score_seconds = perf_counter() - score_start
                    artifact = folder / f"outer-{index + 1}-{name}.pkl"
                    artifact.write_bytes(pickle.dumps(detector, protocol=pickle.HIGHEST_PROTOCOL))
                    fitted[name] = {"scores": scores, "ready": ready}
                    fit_audits[name] = {
                        **audit,
                        "fit_seconds": fit_seconds,
                        "score_seconds": score_seconds,
                        "model_sha256": digest(artifact),
                    }
                prediction = fitted[name]
                measured = assess(groups, **prediction, choice=choice)
                ranking = assess_ranking(groups, **prediction)
                if set(metrics[procedure]) & set(measured["per_group"]):
                    raise ValueError("Duplicate outer assessment group.")
                metrics[procedure].update(measured["per_group"])
                outcomes[procedure] = {
                    **measured,
                    "ranking": ranking,
                    "f0_5": fhalf(measured["pooled"]),
                }
                predictions[procedure] = prediction
                print(
                    f"  {procedure}: precision={measured['pooled']['precision']:.4f}, "
                    f"recall={measured['pooled']['recall']:.4f}",
                    flush=True,
                )
            record = {
                **fold,
                "selection": choices,
                "fit_audit": fit_audits,
                "procedures": outcomes,
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
            name: {"pooled": summarize(values), "per_group": values}
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
