"""Reproducible training, immutable run folders, and explicit held-out evaluation."""

import hashlib
import json
import pickle
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from forge.config import PROJECT_ROOT
from forge.data.datasets import evaluation_groups, normal_training
from forge.data.partitions import read_plan
from forge.ml.detectors import Detector
from forge.ml.metrics import evaluate, select_threshold


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train(root=PROJECT_ROOT):
    root = Path(root)
    config_path = root / "configs/baseline.json"
    config = json.loads(config_path.read_text())
    inventory, plan = read_plan(root)
    training, audit = normal_training(root)
    groups, group_audit = evaluation_groups("validation", root=root)
    policy = {key: config[key] for key in ("persistence", "max_gap_seconds")}
    candidates, winner = [], None
    for max_samples in [None, *config["max_samples"]]:
        detector = Detector.fit(
            training,
            max_samples=max_samples,
            seed=config["seed"],
            n_estimators=config["n_estimators"],
        )
        scores = {name: detector.score(frame) for name, frame in groups.items()}
        threshold, metrics = select_threshold(groups, scores, policy, config["threshold_quantiles"])
        candidate = {"name": detector.name, "threshold": threshold, "metrics": metrics}
        candidates.append(candidate)
        key = (metrics["pooled"]["f1"], -metrics["pooled"]["false_alarm_onsets"], threshold)
        print(
            f"{detector.name}: validation F1={metrics['pooled']['f1']:.4f}, AP={metrics['pooled']['average_precision']:.4f}",
            flush=True,
        )
        if winner is None or key > winner[0]:
            winner = key, detector, candidate
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    folder = root / "models" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    model_path = folder / "detector.pkl"
    model_path.write_bytes(pickle.dumps(winner[1], protocol=pickle.HIGHEST_PROTOCOL))
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "split_id": plan["split_id"],
        "dataset_revision": inventory["revision"],
        "config": config,
        "config_sha256": digest(config_path),
        "split_sha256": digest(root / "data/skab-splits.json"),
        "inventory_sha256": digest(root / "data/skab-inventory.json"),
        "sklearn_version": version("scikit-learn"),
        "model_sha256": digest(model_path),
        "training_audit": audit,
        "validation_audit": group_audit,
        "detector": winner[1].describe(),
        "threshold": winner[2]["threshold"],
        "policy": policy,
        "validation_candidates": candidates,
        "test_evaluated": False,
    }
    write_json(folder / "metadata.json", metadata)
    write_json(
        root / "models/latest.json",
        {"run_id": run_id, "metadata_sha256": digest(folder / "metadata.json")},
    )
    return folder, metadata


def load_model(root=PROJECT_ROOT, *, active=False):
    """Load ONLY this workspace's trusted local artifact. Never accept uploads.

    Hashes detect accidental corruption, not a maliciously replaced model and
    manifest. Pickle artifacts must come from your own training command.
    """
    root = Path(root)
    pointer_path = root / "models/active.json"
    if not active or not pointer_path.exists():
        pointer_path = root / "models/latest.json"
    pointer = json.loads(pointer_path.read_text())
    folder = (root / "models" / pointer["run_id"]).resolve()
    if folder.parent != (root / "models").resolve():
        raise ValueError("Invalid model run path.")
    if digest(folder / "metadata.json") != pointer["metadata_sha256"]:
        raise ValueError("Model metadata checksum mismatch.")
    metadata = json.loads((folder / "metadata.json").read_text())
    if version("scikit-learn") != metadata["sklearn_version"]:
        raise ValueError(
            "Model requires the training scikit-learn version; retrain in this environment."
        )
    if digest(folder / "detector.pkl") != metadata["model_sha256"]:
        raise ValueError("Model checksum mismatch.")
    config_file = metadata.get("config_file", "configs/baseline.json")
    if config_file not in {"configs/baseline.json", "configs/improvement-v2.json"}:
        raise ValueError("Unrecognized model configuration.")
    provenance = [
        ("data/skab-splits.json", "split_sha256"),
        ("data/skab-inventory.json", "inventory_sha256"),
        (config_file, "config_sha256"),
    ]
    if metadata.get("followup_config_sha256"):
        provenance.append(("configs/improvement-v2-followup.json", "followup_config_sha256"))
    for name, field in provenance:
        if digest(root / name) != metadata[field]:
            raise ValueError(
                f"Training provenance changed: {name}. Use the matching checkout or retrain."
            )
    detector = pickle.loads((folder / "detector.pkl").read_bytes())
    if not isinstance(detector, Detector) or detector.name != metadata["detector"]["name"]:
        raise ValueError("Invalid detector artifact.")
    return detector, metadata, folder


def evaluate_test(*, allow_test=False, root=PROJECT_ROOT):
    if not allow_test:
        raise ValueError("Held-out evaluation requires --allow-test after freezing selection.")
    detector, metadata, folder = load_model(root)
    output = folder / "test-evaluation.json"
    if output.exists():
        # Return the saved result rather than quietly rerunning the holdout.
        return json.loads(output.read_text())
    groups, audit = evaluation_groups("test", allow_test=True, root=root)
    scores = {name: detector.score(frame) for name, frame in groups.items()}
    result = {
        "run_id": metadata["run_id"],
        "model_sha256": metadata["model_sha256"],
        "evaluated_at": datetime.now(UTC).isoformat(),
        "split_id": metadata["split_id"],
        "partition": "test",
        "audit": audit,
        "metrics": evaluate(groups, scores, metadata["threshold"], metadata["policy"]),
    }
    write_json(output, result)
    return result
