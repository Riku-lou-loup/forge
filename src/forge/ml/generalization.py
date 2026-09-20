"""Nested grouped development audit with fold-local fitting and bounded tuning."""

import argparse
import json
import pickle
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from forge.config import PROJECT_ROOT
from forge.data.datasets import FEATURES
from forge.ml.detectors import Detector
from forge.ml.development import (
    development_streams,
    fast_alerts,
    fit_operating,
    merge_streams,
    score_streams,
    selection_key,
    selection_metrics,
    threshold_scan,
)
from forge.ml.metrics import evaluate_group
from forge.ml.training import digest, write_json


def subset(streams, group_names):
    wanted = set(group_names)
    selected = {key: frame for key, frame in streams.items() if frame["_group"].iloc[0] in wanted}
    if {frame["_group"].iloc[0] for frame in selected.values()} != wanted:
        raise ValueError("Requested group is absent from the permitted recording pool.")
    return selected


def validate_pool(streams):
    """Reject a mislabeled overlap boundary before creating any folds."""
    if not streams:
        raise ValueError("No development streams.")
    for frame in streams.values():
        if frame.empty or frame["_group"].nunique() != 1:
            raise ValueError("Each source must belong to exactly one nonempty group.")
    rows = pd.concat(
        [frame[[*FEATURES, "_group"]] for frame in streams.values()], ignore_index=True
    )
    if rows.groupby(FEATURES, dropna=False)["_group"].nunique().gt(1).any():
        raise ValueError("Identical sensor observations cross recording groups.")


def group_folds(streams, count, seed):
    names = np.array(sorted({frame["_group"].iloc[0] for frame in streams.values()}))
    if not 2 <= count <= len(names):
        raise ValueError("Fold count must be between two and the number of groups.")
    splitter = GroupKFold(n_splits=count, shuffle=True, random_state=seed)
    return [
        {"fit_groups": names[fit].tolist(), "assessment_groups": names[assessment].tolist()}
        for fit, assessment in splitter.split(np.zeros(len(names)), groups=names)
    ]


def pool_description(streams):
    groups, audits = merge_streams(streams)
    return [
        {
            "group": name,
            "sources": [key for key, frame in streams.items() if frame["_group"].iloc[0] == name],
            "rows": len(frame),
            "normal": int(frame.anomaly.eq(0).sum()),
            "anomalous": int(frame.anomaly.eq(1).sum()),
            "unknown": int(frame.anomaly.lt(0).sum()),
            **audits[name],
        }
        for name, frame in groups.items()
    ]


def fit_candidate(streams, candidate, config):
    """Only explicit fit streams can reach global scaling and estimator fitting."""
    groups, _ = merge_streams(streams)
    normal = pd.concat([frame.loc[frame.anomaly.eq(0), FEATURES] for frame in groups.values()])
    if normal.empty:
        raise ValueError("Fold has no normal observations for scaling.")
    if candidate["id"] == "isolation_forest":
        detector = Detector.fit(normal, seed=config["seed"], **config["isolation_forest"])
        fit_audit = {"fit_rows": len(normal), "positive_training_targets": 0}
    else:
        reference = Detector.fit(normal)
        settings = {
            **config,
            "gradient_boosting": {k: v for k, v in candidate.items() if k != "id"},
        }
        detector, fit_audit = fit_operating(
            reference, streams, "relative", settings, estimator_kind="gradient_boosting"
        )
    detector.name = candidate["id"]
    return detector, {
        **fit_audit,
        "fit_groups": sorted(groups),
        "fit_sources": list(streams),
        "normal_scaling_rows": len(normal),
        "center": detector.center.tolist(),
        "scale": detector.scale.tolist(),
    }


def recall_fulfillment(metrics, config):
    return min(
        min(1.0, (metrics[field] or 0.0) / config[requirement])
        for field, requirement in [
            ("recall", "minimum_point_recall"),
            ("event_recall", "minimum_event_recall"),
            ("minimum_group_recall", "minimum_group_point_recall"),
        ]
    )


def choice_key(choice, config):
    return (
        int(choice["feasible"]),
        recall_fulfillment(choice["metrics"], config),
        *selection_key(choice["metrics"]),
    )


def choose_operating_point(groups, scores, ready, config):
    best, scan = threshold_scan(groups, scores, ready, config)
    if best is None:
        best = max(scan, key=lambda row: choice_key(row, config))
    return {
        **best,
        "status": "recall_feasible" if best["feasible"] else "recall_constraints_unmet",
    }, scan


def inner_selection(streams, candidates, config, fold_count, *, progress="inner"):
    folds = group_folds(streams, fold_count, config["seed"])
    groups, _ = merge_streams(streams)
    prediction_sets = {candidate["id"]: {"scores": {}, "ready": {}} for candidate in candidates}
    fits = []
    for index, fold in enumerate(folds):
        fitting = subset(streams, fold["fit_groups"])
        assessment = subset(streams, fold["assessment_groups"])
        print(
            f"{progress} fold {index + 1}/{len(folds)}: fit {len(fitting)} source files", flush=True
        )
        for candidate in candidates:
            detector, audit = fit_candidate(fitting, candidate, config)
            scores, ready = score_streams(detector, assessment)
            stored = prediction_sets[candidate["id"]]
            if set(stored["scores"]) & set(scores):
                raise ValueError("A group received more than one out-of-fold prediction.")
            stored["scores"].update(scores)
            stored["ready"].update(ready)
            fits.append({"fold": index, "candidate": candidate["id"], **fold, "fit_audit": audit})
    choices, scans = {}, {}
    for candidate in candidates:
        name = candidate["id"]
        stored = prediction_sets[name]
        if set(stored["scores"]) != set(groups):
            raise ValueError(
                "Out-of-fold predictions do not cover the permitted pool exactly once."
            )
        choices[name], scans[name] = choose_operating_point(groups, **stored, config=config)
        print(
            f"{progress} selected {name}: {choices[name]['status']}, precision={choices[name]['metrics']['precision']:.4f}",
            flush=True,
        )
    return choices, {"folds": folds, "fits": fits, "threshold_scans": scans}


def aggregate(per_group):
    counts = {
        key: sum(value[key] for value in per_group.values())
        for key in (
            "tp",
            "fp",
            "fn",
            "tn",
            "events",
            "detected_events",
            "false_alarm_onsets",
            "available_normal_rows",
            "unavailable_rows",
            "unavailable_anomaly_rows",
        )
    }
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    delays = [
        delay for value in per_group.values() for delay in value["detected_event_delays_seconds"]
    ]
    available = counts["available_normal_rows"]
    return {
        **counts,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "fpr_available_normal": fp / available if available else None,
        "fpr_all_normal": fp / (fp + tn) if fp + tn else None,
        "false_onsets_per_available_normal_hour": counts["false_alarm_onsets"] / (available / 3600)
        if available
        else None,
        "event_recall": counts["detected_events"] / counts["events"] if counts["events"] else None,
        "macro_group_precision": float(np.mean([m["precision"] for m in per_group.values()])),
        "macro_group_recall": float(np.mean([m["recall"] for m in per_group.values()])),
        "minimum_group_recall": min(m["recall"] for m in per_group.values()),
        "median_detected_event_delay_seconds": float(np.median(delays)) if delays else None,
        "alerted_readings": tp + fp,
    }


def assess(groups, scores, ready, choice):
    per_group = {}
    for name, frame in groups.items():
        metrics = selection_metrics(
            {name: frame},
            {name: scores[name]},
            {name: ready[name]},
            choice["threshold"],
            choice["policy"],
        )
        alerts = fast_alerts(
            scores[name],
            frame.datetime,
            ready[name] & frame.anomaly.ge(0).to_numpy(),
            choice["threshold"],
            **choice["policy"],
        )
        # The alert rule is already applied. Reuse the reference event accounting
        # with one-step binary alerts; discard ranking metrics on these binary values.
        events = evaluate_group(
            frame,
            alerts.astype(float),
            0.5,
            {"persistence": 1, "max_gap_seconds": choice["policy"]["max_gap_seconds"]},
        )
        per_group[name] = {
            **metrics,
            "detected_event_delays_seconds": events["detected_event_delays_seconds"],
        }
    return {"pooled": aggregate(per_group), "per_group": per_group}


def save_predictions(path, prediction_sets):
    arrays = {}
    for procedure, predictions in prediction_sets.items():
        names = list(predictions["scores"])
        arrays[f"{procedure}_groups"] = np.array(names)
        for index, name in enumerate(names):
            arrays[f"{procedure}_{index}_scores"] = predictions["scores"][name]
            arrays[f"{procedure}_{index}_ready"] = predictions["ready"][name]
    np.savez_compressed(path, **arrays)


def run_audit(root=PROJECT_ROOT, *, plan_only=False):
    root = Path(root)
    config_path = root / "configs/generalization-v1.json"
    config = json.loads(config_path.read_text())
    streams = {**development_streams("train", root), **development_streams("validation", root)}
    validate_pool(streams)
    folds = group_folds(streams, config["outer_folds"], config["seed"])
    plan = {"protocol": config["protocol"], "pool": pool_description(streams), "outer_folds": folds}
    if plan_only:
        print(json.dumps(plan, indent=2))
        return plan
    run_id = (
        "generalization-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    folder = root / "models" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "plan.json", plan)
    preservation_paths = [
        "models/active.json",
        "models/latest.json",
        "docs/improvement-results.json",
        "docs/evaluation-results.json",
    ]
    preserved = {name: digest(root / name) for name in preservation_paths if (root / name).exists()}
    provenance = {
        "config_sha256": digest(config_path),
        "split_sha256": digest(root / "data/skab-splits.json"),
        "inventory_sha256": digest(root / "data/skab-inventory.json"),
        "sklearn_version": version("scikit-learn"),
        "python_version": sys.version,
        "source_sha256": {
            str(path.relative_to(root)).replace("\\", "/"): digest(path)
            for path in [
                root / "src/forge/ml/generalization.py",
                root / "src/forge/ml/development.py",
                root / "src/forge/ml/operating.py",
                root / "src/forge/ml/detectors.py",
                root / "src/forge/ml/metrics.py",
                root / "src/forge/data/datasets.py",
            ]
        },
    }
    write_json(folder / "provenance.json", provenance)
    candidates = [{"id": "isolation_forest"}, *config["candidates"]]
    definitions = {candidate["id"]: candidate for candidate in candidates}
    records = []
    all_metrics = {name: {} for name in ["isolation_forest", "current_hgb", "tuned_hgb"]}
    with threadpool_limits(limits=config["threads"]):
        for index, fold in enumerate(folds):
            print(
                f"OUTER {index + 1}/{len(folds)} excludes {fold['assessment_groups']}", flush=True
            )
            fitting = subset(streams, fold["fit_groups"])
            # Only fitting streams enter inner selection. Outer assessment labels
            # are consumed below, after all candidate and operating choices are fixed.
            choices, inner_audit = inner_selection(
                fitting,
                candidates,
                config,
                config["inner_folds"],
                progress=f"outer-{index + 1} inner",
            )
            tuned_id = max(
                [c["id"] for c in config["candidates"]],
                key=lambda name: choice_key(choices[name], config),
            )
            selected_ids = {
                "isolation_forest": "isolation_forest",
                "current_hgb": config["reference_candidate"],
                "tuned_hgb": tuned_id,
            }
            frozen = {"selected_ids": selected_ids, "choices": choices}
            write_json(folder / f"outer-{index + 1}-selection.json", frozen)
            write_json(folder / f"outer-{index + 1}-inner.json", inner_audit)
            assessment = subset(streams, fold["assessment_groups"])
            assessment_groups, _ = merge_streams(assessment)
            fitting_groups, _ = merge_streams(fitting)
            fitted = {}
            outcomes, saved = {}, {}
            for procedure, name in selected_ids.items():
                if name not in fitted:
                    detector, fit_audit = fit_candidate(fitting, definitions[name], config)
                    fitted[name] = detector, fit_audit
                detector, fit_audit = fitted[name]
                scores, ready = score_streams(detector, assessment)
                measured = assess(assessment_groups, scores, ready, choices[name])
                train_scores, train_ready = score_streams(detector, fitting)
                train_metrics = assess(fitting_groups, train_scores, train_ready, choices[name])[
                    "pooled"
                ]
                outcomes[procedure] = {
                    "candidate": name,
                    "selection": choices[name],
                    "fit_audit": fit_audit,
                    "assessment": measured,
                    "training": train_metrics,
                }
                if set(all_metrics[procedure]) & set(measured["per_group"]):
                    raise ValueError("Duplicate outer assessment group.")
                all_metrics[procedure].update(measured["per_group"])
                saved[procedure] = {"scores": scores, "ready": ready}
                artifact = folder / f"outer-{index + 1}-{procedure}.pkl"
                artifact.write_bytes(pickle.dumps(detector, protocol=pickle.HIGHEST_PROTOCOL))
                outcomes[procedure]["model_sha256"] = digest(artifact)
                print(
                    f"outer-{index + 1} {procedure}: precision={measured['pooled']['precision']:.4f}, recall={measured['pooled']['recall']:.4f}",
                    flush=True,
                )
            record = {"fold": index + 1, **fold, "procedures": outcomes}
            records.append(record)
            save_predictions(folder / f"outer-{index + 1}-predictions.npz", saved)
            write_json(folder / f"outer-{index + 1}-results.json", record)
        print(
            "FINAL grouped selection on full development pool; not independent evaluation",
            flush=True,
        )
        final_choices, final_audit = inner_selection(
            streams, config["candidates"], config, config["outer_folds"], progress="final"
        )
        final_id = max(final_choices, key=lambda name: choice_key(final_choices[name], config))
        detector, final_fit = fit_candidate(streams, definitions[final_id], config)
        model_path = folder / "detector.pkl"
        model_path.write_bytes(pickle.dumps(detector, protocol=pickle.HIGHEST_PROTOCOL))
        write_json(folder / "final-selection-audit.json", final_audit)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "config": config,
        "provenance": provenance,
        **plan,
        "fold_results": records,
        "procedures": {
            name: {"pooled": aggregate(per_group), "per_group": per_group}
            for name, per_group in all_metrics.items()
        },
        "final_refit": {
            "candidate": final_id,
            "selection": final_choices[final_id],
            "all_choices": final_choices,
            "fit_audit": final_fit,
            "model_sha256": digest(model_path),
            "descriptor": detector.describe(),
            "activated": False,
        },
        "preserved_sha256": preserved,
        "scope": config["scope"],
    }
    if any(digest(root / name) != checksum for name, checksum in preserved.items()):
        raise ValueError("A preserved app artifact or historical report changed during the audit.")
    if any(
        set(values) != {item["group"] for item in plan["pool"]} for values in all_metrics.values()
    ):
        raise ValueError("Outer assessment coverage is incomplete.")
    write_json(folder / "audit-results.json", summary)
    print(f"Audit artifacts: {folder}", flush=True)
    return folder, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", action="store_true", help="Print group inventory and folds without fitting."
    )
    args = parser.parse_args()
    run_audit(plan_only=args.plan)
