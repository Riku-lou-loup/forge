"""Reproducible precision-first development comparison; never opens test data."""

import hashlib
import json
import pickle
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, IsolationForest

from forge.config import PROJECT_ROOT
from forge.data.datasets import OBSERVATION_KEY, merge_group, normal_training
from forge.data.partitions import inspect_recording, read_plan, select_records
from forge.ml.detectors import Detector
from forge.ml.metrics import evaluate, starts
from forge.ml.operating import OperatingDetector
from forge.ml.training import digest, load_model, write_json


def development_streams(partition, root=PROJECT_ROOT):
    if partition not in {"train", "validation"}:
        raise ValueError("Development streams permit training and validation only.")
    inventory, plan = read_plan(root)
    streams = {}
    for record in select_records(inventory, plan, partition):
        frame = inspect_recording(record, root).copy()
        if not record["annotations_present"]:
            # Internal training target from catalog provenance; source CSV stays unmodified.
            frame["anomaly"] = 0
        frame["_group"] = record["leakage_group"]
        streams[record["experiment_id"]] = frame
    return streams


def merge_streams(streams):
    frames = {}
    for frame in streams.values():
        frames.setdefault(frame["_group"].iloc[0], []).append(frame)
    grouped, audits = {}, {}
    for name, items in frames.items():
        grouped[name], audits[name] = merge_group(items)
    return grouped, audits


def score_streams(detector, streams):
    """Initialize each source CSV exactly as serving does, then deduplicate scores."""
    scored = {}
    for name, frame in streams.items():
        result = frame.copy()
        result["_score"] = detector.score(frame)
        result["_ready"] = (
            detector.readiness(frame) if isinstance(detector, OperatingDetector) else True
        )
        scored[name] = result
    groups, _ = merge_streams(scored)
    return (
        {name: frame["_score"].to_numpy() for name, frame in groups.items()},
        {name: frame["_ready"].to_numpy(dtype=bool) for name, frame in groups.items()},
    )


def fast_alerts(scores, timestamps, ready, threshold, persistence, max_gap_seconds):
    """Vectorized equivalent of causal_alerts, with explicit unavailable observations."""
    above = (np.asarray(scores) > threshold) & np.asarray(ready)
    gaps = np.r_[
        False, np.diff(pd.DatetimeIndex(timestamps).as_unit("ns").asi8) / 1e9 > max_gap_seconds
    ]
    indices = np.arange(len(above))
    reset_at = np.where(~above, indices, np.where(gaps, indices - 1, -1))
    run = indices - np.maximum.accumulate(reset_at)
    return above & (run >= persistence)


def selection_metrics(groups, scores, ready, threshold, policy):
    tp = fp = fn = tn = events = detected = false_onsets = 0
    available_normal = total_normal = unavailable_positive = unavailable_rows = 0
    group_recalls = {}
    for name, frame in groups.items():
        intervals = np.diff(pd.DatetimeIndex(frame.datetime).as_unit("ns").asi8) / 1e9
        if (intervals < 1).any():
            raise ValueError("Development exposure assumes SKAB sampling of at least one second.")
        labels = frame.anomaly.to_numpy(dtype=int)
        valid = labels >= 0
        available = ready[name] & valid
        alerts = fast_alerts(scores[name], frame.datetime, available, threshold, **policy)
        positive, normal = labels == 1, labels == 0
        this_tp = int((positive & alerts).sum())
        this_positive = int(positive.sum())
        tp += this_tp
        fp += int((normal & alerts).sum())
        fn += int((positive & ~alerts).sum())
        tn += int((normal & ~alerts).sum())
        available_normal += int((available & normal).sum())
        total_normal += int(normal.sum())
        unavailable_positive += int((~ready[name] & positive).sum())
        unavailable_rows += int((~ready[name] & valid).sum())
        if this_positive:
            group_recalls[name] = this_tp / this_positive
        event_starts = starts(positive, frame.datetime, policy["max_gap_seconds"])
        alert_starts = starts(alerts, frame.datetime, policy["max_gap_seconds"])
        false_onsets += int(normal[alert_starts].sum())
        events += len(event_starts)
        # Assign each positive sample its contiguous event index, including gaps.
        event_ids = np.cumsum(np.isin(np.arange(len(frame)), event_starts))
        hit_ids = set(event_ids[alert_starts[positive[alert_starts]]])
        detected += len(hit_ids)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "fpr_available_normal": fp / available_normal if available_normal else None,
        "fpr_all_normal": fp / total_normal if total_normal else None,
        "false_alarm_onsets": false_onsets,
        "false_onsets_per_available_normal_hour": false_onsets / (available_normal / 3600)
        if available_normal
        else None,
        "events": events,
        "detected_events": detected,
        "event_recall": detected / events if events else None,
        "minimum_group_recall": min(group_recalls.values()) if group_recalls else None,
        "group_recalls": group_recalls,
        "available_normal_rows": available_normal,
        "unavailable_rows": unavailable_rows,
        "unavailable_anomaly_rows": unavailable_positive,
    }


def feasible(metrics, config):
    return (
        metrics["recall"] >= config["minimum_point_recall"]
        and metrics["event_recall"] is not None
        and metrics["event_recall"] >= config["minimum_event_recall"]
        and metrics["minimum_group_recall"] is not None
        and metrics["minimum_group_recall"] >= config["minimum_group_point_recall"]
    )


def selection_key(metrics):
    return (
        metrics["precision"],
        -metrics["false_onsets_per_available_normal_hour"],
        metrics["recall"],
    )


def threshold_scan(groups, scores, ready, config):
    values = np.concatenate(
        [
            scores[name][ready[name] & frame.anomaly.ge(0).to_numpy()]
            for name, frame in groups.items()
        ]
    )
    if not len(values):
        raise ValueError("No available observations for threshold selection.")
    thresholds = np.unique(
        np.r_[
            np.nextafter(values.min(), -np.inf),
            np.quantile(values, np.linspace(0, 1, config["threshold_quantiles"])),
        ]
    )
    # Startup zeros must never trigger an alert; readiness enforces this even for a negative threshold.
    rows = []
    best = None
    for persistence in config["persistence_candidates"]:
        policy = {"persistence": persistence, "max_gap_seconds": config["max_gap_seconds"]}
        for threshold in thresholds:
            metrics = selection_metrics(groups, scores, ready, float(threshold), policy)
            item = {
                "threshold": float(threshold),
                "policy": policy,
                "metrics": metrics,
                "feasible": feasible(metrics, config),
            }
            rows.append(item)
            if item["feasible"] and (
                best is None or selection_key(metrics) > selection_key(best["metrics"])
            ):
                best = item
    return best, rows


def fit_operating(
    reference, groups, representation, config, *, classifier=True, estimator_kind="extra_trees"
):
    name = f"{estimator_kind}_{representation}" if classifier else "isolation_forest_relative"
    detector = OperatingDetector(
        name,
        reference.center,
        reference.scale,
        reference.scale_methods,
        estimator=None,
        representation=representation,
        classifier=classifier,
        reference_readings=config["reference_readings"],
        rolling_readings=config["rolling_readings"],
        max_gap_seconds=config["max_gap_seconds"],
    )
    inputs, row_frames = [], []
    for frame in groups.values():
        features, ready = detector.transform(frame)
        rows = frame[[*OBSERVATION_KEY, "anomaly", "_group"]].copy()
        rows["_ready"] = ready
        row_frames.append(rows)
        inputs.append(features)
    rows = pd.concat(row_frames, ignore_index=True)
    ambiguous = rows.groupby(OBSERVATION_KEY, dropna=False).anomaly.transform("nunique").gt(1)
    eligible = rows["_ready"] & rows.anomaly.ge(0) & ~ambiguous & ~rows.duplicated(OBSERVATION_KEY)
    if not classifier:
        eligible &= rows.anomaly.eq(0)
    x = np.concatenate(inputs)[eligible.to_numpy()]
    retained = rows.loc[eligible]
    y = retained.anomaly.to_numpy()
    weight = 1 / retained.groupby("_group").anomaly.transform("size").to_numpy()
    weight *= len(weight) / weight.sum()
    if classifier:
        if estimator_kind == "extra_trees":
            estimator = ExtraTreesClassifier(**config["extra_trees"], random_state=config["seed"])
        elif estimator_kind == "gradient_boosting":
            estimator = HistGradientBoostingClassifier(
                **config["gradient_boosting"], random_state=config["seed"]
            )
        else:
            raise ValueError("Unknown classifier family.")
        estimator.fit(x, y, sample_weight=weight)
    else:
        estimator = IsolationForest(
            n_estimators=256, max_samples=1024, random_state=config["seed"], n_jobs=2
        )
        estimator.fit(x, sample_weight=weight)
    detector.estimator = estimator
    return detector, {
        "fit_rows": len(x),
        "positive_training_targets": int(y.sum()),
        "group_weighting": "Equal total weight per training leakage group",
        "representation": representation,
    }


def run_development(root=PROJECT_ROOT, *, followup=False):
    root = Path(root)
    config = json.loads((root / "configs/improvement-v2.json").read_text())
    if followup:
        config.update(json.loads((root / "configs/improvement-v2-followup.json").read_text()))
    baseline, baseline_metadata, _ = load_model(root)
    training = development_streams("train", root)
    _, training_audit = merge_streams(training)
    normal, normal_audit = normal_training(root)
    reference = Detector.fit(normal)
    validation_streams = development_streams("validation", root)
    validation, validation_audit = merge_streams(validation_streams)
    run_id = "development-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    folder = root / "models" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    results, selected = [], None
    candidates = (
        []
        if followup
        else [("frozen_isolation_forest", baseline, {}), ("robust_max", reference, {})]
    )
    definitions = (
        config["followup_candidates"]
        if followup
        else [
            {"representation": mode, "estimator": "extra_trees"}
            for mode in config["candidate_features"]
        ]
    )
    for definition in definitions:
        print(
            f"Fitting {definition['estimator']} with {definition['representation']} features",
            flush=True,
        )
        detector, audit = fit_operating(
            reference,
            training,
            definition["representation"],
            config,
            estimator_kind=definition["estimator"],
        )
        candidates.append((detector.name, detector, audit))
    if not followup:
        detector, audit = fit_operating(reference, training, "relative", config, classifier=False)
        candidates.append((detector.name, detector, audit))
    baseline_scores = {name: baseline.score(frame) for name, frame in validation.items()}
    baseline_ready = {name: np.ones(len(frame), dtype=bool) for name, frame in validation.items()}
    baseline_result = selection_metrics(
        validation,
        baseline_scores,
        baseline_ready,
        baseline_metadata["threshold"],
        baseline_metadata["policy"],
    )
    for name, detector, fit_audit in candidates:
        print(f"Selecting operating point: {name}", flush=True)
        scores, ready = score_streams(detector, validation_streams)
        best, scan = threshold_scan(validation, scores, ready, config)
        result = {"name": name, "fit_audit": fit_audit, "best": best, "scan": scan}
        results.append(result)
        if best is not None:
            print(json.dumps({"candidate": name, **best["metrics"]}), flush=True)
            if selected is None or selection_key(best["metrics"]) > selection_key(
                selected[2]["metrics"]
            ):
                selected = name, detector, best, scores, ready
        else:
            print(f"{name}: no operating point meets the recall floors", flush=True)
    summary = {
        "schema_version": 2,
        "sklearn_version": version("scikit-learn"),
        "prediction_boundary": "Source recording: initialize and compute features before observation deduplication. First inventory occurrence provides prediction context for shared observations.",
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "config": config,
        "config_sha256": digest(root / "configs/improvement-v2.json"),
        "followup_config_sha256": digest(root / "configs/improvement-v2-followup.json")
        if followup
        else None,
        "baseline_run": baseline_metadata["run_id"],
        "baseline_model_sha256": baseline_metadata["model_sha256"],
        "split_sha256": digest(root / "data/skab-splits.json"),
        "inventory_sha256": digest(root / "data/skab-inventory.json"),
        "training_audit": training_audit,
        "normal_scaling_audit": normal_audit,
        "validation_audit": validation_audit,
        "baseline_validation": baseline_result,
        "candidates": [
            {"name": result["name"], "fit_audit": result["fit_audit"], "best": result["best"]}
            for result in results
        ],
        "selected": None,
        "scope": "Development validation selection; not independent held-out performance. No new test scoring.",
    }
    if selected:
        name, detector, best, scores, ready = selected
        # Require nonnegative thresholds so legacy score consumers cannot turn startup zeros into alerts.
        if best["threshold"] < 0:
            raise ValueError(
                "Selected negative threshold needs an explicit readiness-aware serving boundary."
            )
        selected_path = folder / "detector.pkl"
        selected_path.write_bytes(pickle.dumps(detector, protocol=pickle.HIGHEST_PROTOCOL))
        full_metrics = evaluate(validation, scores, best["threshold"], best["policy"])
        summary["selected"] = {
            "name": name,
            **best,
            "descriptor": detector.describe(),
            "model_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
            "full_validation_metrics": full_metrics,
            "coverage_note": "Startup unavailable readings count as missed anomalies in recall. FPR and false onset exposure use available normal observations. Full confusion metrics retain suppressed startup readings; inspect availability counts.",
        }
    write_json(folder / "comparison.json", summary)
    write_json(folder / "threshold-scans.json", results)
    print(f"Development artifacts: {folder}", flush=True)
    return folder, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--followup", action="store_true")
    args = parser.parse_args()
    run_development(followup=args.followup)
