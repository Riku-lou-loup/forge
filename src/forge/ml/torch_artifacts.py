"""Explicit PyTorch state-dict artifacts; no active/latest pointer mutation."""

import json
from pathlib import Path

import numpy as np
import torch

from forge.config import PROJECT_ROOT
from forge.data.datasets import FEATURES
from forge.ml.torch_model import TemporalCNN, TorchDetector, validate_config
from forge.ml.training import digest, write_json

ARTIFACT_FILES = {"weights.pt", "config.json", "metadata.json", "history.json"}


def save_torch_artifact(folder, detector, metadata, history):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    torch.save(detector.model.state_dict(), folder / "weights.pt")
    write_json(folder / "config.json", detector.config)
    write_json(folder / "history.json", history)
    metadata = {
        **metadata,
        "schema_version": 1,
        "model_sha256": digest(folder / "weights.pt"),
        "config_sha256": digest(folder / "config.json"),
        "config": detector.config,
        "detector": detector.describe(),
        "torch_version": str(torch.__version__),
    }
    write_json(folder / "metadata.json", metadata)
    write_json(
        folder / "checksums.json", {name: digest(folder / name) for name in sorted(ARTIFACT_FILES)}
    )
    return metadata


def load_torch_artifact(folder, root=PROJECT_ROOT):
    """Load a trusted local run with safe tensor loading and consistency checks.

    Checksums detect corruption, not authenticity when an attacker can replace
    the complete artifact. Never load untrusted uploaded checkpoints.
    """
    folder, root = Path(folder), Path(root)
    checksums = json.loads((folder / "checksums.json").read_text(encoding="utf-8"))
    if set(checksums) != ARTIFACT_FILES:
        raise ValueError("Unexpected artifact checksum manifest.")
    for name, checksum in checksums.items():
        if digest(folder / name) != checksum:
            raise ValueError(f"Artifact checksum mismatch: {name}.")
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    if (
        metadata["schema_version"] != 1
        or metadata["config"] != config
        or metadata["config_sha256"] != checksums["config.json"]
        or metadata["model_sha256"] != checksums["weights.pt"]
        or metadata["detector"]["features"] != FEATURES
        or metadata["detector"]["name"] != "temporal_cnn_v1"
        or metadata["detector"]["window"] != config["window"]
        or metadata["policy"] != {key: config[key] for key in ("persistence", "max_gap_seconds")}
        or not np.isfinite(metadata["threshold"])
    ):
        raise ValueError("Inconsistent artifact configuration or metadata.")
    if metadata["torch_version"] != str(torch.__version__):
        raise ValueError("Artifact requires the recorded PyTorch version.")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Missing artifact provenance.")
    manifests = provenance.get("manifests")
    if not isinstance(manifests, dict) or set(manifests) != {
        "data/skab-splits.json",
        "data/skab-inventory.json",
    }:
        raise ValueError("Artifact provenance requires exactly the pinned split and inventory.")
    for name, checksum in manifests.items():
        if digest(root / name) != checksum:
            raise ValueError(f"Training provenance changed: {name}.")
    descriptor = metadata["detector"]
    center, scale = np.asarray(descriptor["center"]), np.asarray(descriptor["scale"])
    if (
        center.shape != (len(FEATURES),)
        or scale.shape != center.shape
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError("Invalid artifact scaler shape or values.")
    torch.set_num_threads(config["cpu_threads"])
    model = TemporalCNN(config)
    state = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Invalid model state keys.")
    for key, tensor in state.items():
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != expected[key].shape
            or tensor.dtype != expected[key].dtype
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"Invalid model state shape, dtype or values: {key}.")
    model.load_state_dict(state, strict=True)
    model.eval()
    return TorchDetector(model, center, scale, descriptor["scale_methods"], config), metadata
