"""Activate a verified local development winner without replacing the baseline."""

import json
from pathlib import Path

from forge.config import PROJECT_ROOT
from forge.ml.development import feasible
from forge.ml.training import digest, load_model, write_json


def activate(run_id, root=PROJECT_ROOT):
    root = Path(root)
    folder = (root / "models" / run_id).resolve()
    if folder.parent != (root / "models").resolve():
        raise ValueError("Invalid development run path.")
    result = json.loads((folder / "comparison.json").read_text())
    if result["schema_version"] != 2 or result["run_id"] != run_id:
        raise ValueError("Only corrected development runs can be activated.")
    selected = result["selected"]
    if not selected or not feasible(selected["metrics"], result["config"]):
        raise ValueError("No candidate meets the declared recall floors.")
    if selected["metrics"]["precision"] <= result["baseline_validation"]["precision"]:
        raise ValueError("Candidate does not improve baseline precision.")
    for path, key in [
        ("configs/improvement-v2.json", "config_sha256"),
        ("data/skab-splits.json", "split_sha256"),
        ("data/skab-inventory.json", "inventory_sha256"),
    ]:
        if digest(root / path) != result[key]:
            raise ValueError(f"Development provenance changed: {path}")
    if (
        result.get("followup_config_sha256")
        and digest(root / "configs/improvement-v2-followup.json")
        != result["followup_config_sha256"]
    ):
        raise ValueError("Follow-up configuration changed.")
    if digest(folder / "detector.pkl") != selected["model_sha256"]:
        raise ValueError("Development model checksum mismatch.")
    metadata = {
        "schema_version": 2,
        "run_id": run_id,
        "created_at": result["created_at"],
        "sklearn_version": result["sklearn_version"],
        "model_sha256": selected["model_sha256"],
        "detector": selected["descriptor"],
        "threshold": selected["threshold"],
        "policy": selected["policy"],
        "config_file": "configs/improvement-v2.json",
        "config_sha256": result["config_sha256"],
        "followup_config_sha256": result.get("followup_config_sha256"),
        "split_sha256": result["split_sha256"],
        "inventory_sha256": result["inventory_sha256"],
        "comparison_sha256": digest(folder / "comparison.json"),
        "scope": result["scope"],
        "test_evaluated": False,
    }
    metadata_path = folder / "metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("Existing activation metadata differs; preserve this run.")
    else:
        write_json(metadata_path, metadata)
    # Validate through the normal loader before switching the app's active pointer.
    pointer = {"run_id": run_id, "metadata_sha256": digest(metadata_path)}
    active_path = root / "models/active.json"
    previous = active_path.read_bytes() if active_path.exists() else None
    try:
        write_json(active_path, pointer)
        load_model(root, active=True)
    except Exception:
        if previous is None:
            active_path.unlink(missing_ok=True)
        else:
            active_path.write_bytes(previous)
        raise
    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(activate(args.run_id), indent=2))
