"""Train one frozen CPU CNN on original training data; select a development threshold."""

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np

from forge.config import PROJECT_ROOT
from forge.data.datasets import normal_training
from forge.data.partitions import inspect_recording, read_plan, select_records
from forge.ml.development import development_streams, merge_streams, selection_metrics
from forge.ml.torch_artifacts import load_torch_artifact, save_torch_artifact
from forge.ml.torch_model import fit_detector, validate_config
from forge.ml.training import digest, write_json


def score_validation(detector, streams):
    """Score each source independently, then apply the shared endpoint dedup policy."""
    scored = {}
    for name, frame in streams.items():
        frame = frame.copy()
        frame["_score"] = detector.score(frame)
        frame["_ready"] = detector.readiness(frame)
        scored[name] = frame
    groups, audit = merge_streams(scored)
    scores = {name: frame._score.to_numpy() for name, frame in groups.items()}
    ready = {name: frame._ready.to_numpy(dtype=bool) for name, frame in groups.items()}
    return groups, scores, ready, audit


def select_operating_point(groups, scores, ready, config):
    """Maximize precision subject to recall floor; report any relaxed fallback.

    Nondegenerate candidates must detect a positive and leave at least one
    available labeled endpoint unalerted. If the recall floor is infeasible,
    choose maximum recall, then precision among these candidates. If none exist,
    fail explicitly rather than saving a silent all-negative detector.
    """
    values = np.concatenate(
        [
            scores[name][ready[name] & frame.anomaly.ge(0).to_numpy()]
            for name, frame in groups.items()
        ]
    )
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("No finite available validation scores.")
    thresholds = np.unique(
        np.r_[
            np.nextafter(values.min(), -np.inf),
            np.quantile(values, np.linspace(0, 1, config["threshold_quantiles"])),
        ]
    )
    policy = {key: config[key] for key in ("persistence", "max_gap_seconds")}
    scan, candidates = [], []
    for threshold in thresholds:
        metrics = selection_metrics(groups, scores, ready, float(threshold), policy)
        total = sum(metrics[key] for key in ("tp", "fp", "fn", "tn"))
        available = total - metrics["unavailable_rows"]
        nondegenerate = metrics["tp"] > 0 and metrics["tp"] + metrics["fp"] < available
        row = {
            "threshold": float(threshold),
            "metrics": metrics,
            "nondegenerate": nondegenerate,
            "meets_recall_floor": metrics["recall"] >= config["minimum_recall"],
        }
        scan.append(row)
        if nondegenerate:
            candidates.append(row)
    if not candidates:
        raise ValueError("No nondegenerate validation operating point with a true positive.")
    feasible = [row for row in candidates if row["meets_recall_floor"]]
    if feasible:
        best = max(
            feasible,
            key=lambda row: (
                row["metrics"]["precision"],
                row["metrics"]["recall"],
                row["threshold"],
            ),
        )
    else:
        best = max(
            candidates,
            key=lambda row: (
                row["metrics"]["recall"],
                row["metrics"]["precision"],
                row["threshold"],
            ),
        )
    return {**best, "fallback_used": not bool(feasible)}, scan


def provenance(root, inventory, plan):
    source_root = Path(__file__).resolve().parents[3]
    modules = [
        "ml/torch_data.py",
        "ml/torch_model.py",
        "ml/torch_training.py",
        "ml/torch_artifacts.py",
        "ml/development.py",
        "ml/detectors.py",
        "ml/training.py",
        "ml/metrics.py",
        "data/datasets.py",
        "data/partitions.py",
        "data/sample.py",
    ]
    revision = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={source_root.as_posix()}",
            "-C",
            str(source_root),
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    records = [
        {
            "partition": partition,
            "experiment_id": record["experiment_id"],
            "sha256": record["sha256"],
            "leakage_group": record["leakage_group"],
        }
        for partition in ("train", "validation")
        for record in select_records(inventory, plan, partition)
    ]
    return {
        "partitions_read": ["train", "validation"],
        "split_id": plan["split_id"],
        "dataset_revision": inventory["revision"],
        "manifests": {
            name: digest(root / name)
            for name in (
                "data/skab-splits.json",
                "data/skab-inventory.json",
            )
        },
        "source_config_sha256": digest(root / "configs/pytorch-v1.json"),
        "recordings": records,
        "code_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "code_sha256": {
            f"src/forge/{name}": digest(source_root / "src/forge" / name) for name in modules
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in (
                    "torch",
                    "numpy",
                    "pandas",
                    "scikit-learn",
                )
            },
        },
    }


def run_training(root=PROJECT_ROOT):
    root = Path(root).resolve()
    config = json.loads((root / "configs/pytorch-v1.json").read_text(encoding="utf-8-sig"))
    validate_config(config)
    inventory, plan = read_plan(root)
    training = development_streams("train", root)
    normals, normal_audit = normal_training(root)
    detector, history, training_audit = fit_detector(training, normals, config)
    # Validation is only opened after all weights and scaling have been fitted.
    validation = development_streams("validation", root)
    groups, scores, ready, validation_audit = score_validation(detector, validation)
    best, scan = select_operating_point(groups, scores, ready, config)
    run_id = "pytorch-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    folder = root / "models" / run_id
    coverage = {
        "total_unique_rows": sum(len(frame) for frame in groups.values()),
        "available_unique_rows": sum(int(mask.sum()) for mask in ready.values()),
        "unavailable_rows": best["metrics"]["unavailable_rows"],
        "unavailable_anomaly_rows": best["metrics"]["unavailable_anomaly_rows"],
        "ambiguous_rows": sum(item["ambiguous_rows"] for item in validation_audit.values()),
    }
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "threshold": best["threshold"],
        "policy": {key: config[key] for key in ("persistence", "max_gap_seconds")},
        "training_audit": training_audit,
        "normal_scaling_audit": normal_audit,
        "validation_audit": validation_audit,
        "validation_metrics": best["metrics"],
        "validation_coverage": coverage,
        "threshold_selection": {
            "minimum_recall": config["minimum_recall"],
            "fallback_used": best["fallback_used"],
            "selection_rule": "Precision first with recall floor; otherwise nondegenerate recall-first fallback.",
            "nondegenerate": "At least one true positive and one unalerted available labeled endpoint.",
            "candidate_count": len(scan),
        },
        "scope": "Reused development validation for threshold selection, not independent performance. No original test CSV access. Predicts anomaly annotations, not proven physical failures.",
        "provenance": provenance(root, inventory, plan),
    }
    metadata = save_torch_artifact(folder, detector, metadata, history)
    write_json(folder / "threshold-scan.json", scan)
    # Check serialization on every validation source, retaining original context.
    loaded, _ = load_torch_artifact(folder, root=root)
    for frame in validation.values():
        if not np.array_equal(detector.score(frame), loaded.score(frame)):
            raise ValueError("Reloaded detector scores differ from the trained detector.")
    print(
        json.dumps(
            {
                "artifact": str(folder),
                "validation": best["metrics"],
                "coverage": coverage,
                "threshold_selection": metadata["threshold_selection"],
                "history": history,
            },
            indent=2,
        )
    )
    return folder, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository containing frozen config, pinned data and models output.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        help="Reload this explicit run instead of training; relative to --root.",
    )
    parser.add_argument(
        "--recording", help="Investigate this train/validation experiment after reload."
    )
    args = parser.parse_args()
    if args.recording and args.artifact is None:
        parser.error("--recording requires --artifact")
    if args.artifact is None:
        run_training(args.root)
        return
    artifact = args.artifact if args.artifact.is_absolute() else args.root / args.artifact
    detector, metadata = load_torch_artifact(artifact, root=args.root)
    if args.recording:
        from forge.agents.workflow import investigate
        from forge.rag.retrieval import Retriever

        inventory, plan = read_plan(args.root)
        records = [
            record
            for partition in ("train", "validation")
            for record in select_records(inventory, plan, partition)
            if record["experiment_id"] == args.recording
        ]
        if len(records) != 1:
            parser.error("Recording must belong to original train or validation.")
        record = records[0]
        frame = inspect_recording(record, args.root)
        report = investigate(
            frame,
            detector,
            metadata,
            record["experiment_id"],
            record["sha256"],
            retriever=Retriever(args.root / "knowledge/playbook.json"),
        )
        print(report.model_dump_json(indent=2))
    else:
        print(
            json.dumps(
                {
                    "run_id": metadata["run_id"],
                    "detector": detector.describe(),
                    "validation": metadata["validation_metrics"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
