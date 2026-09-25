"""Small deterministic CPU temporal classifier and investigation adapter."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from forge.data.datasets import FEATURES, sensor_matrix
from forge.ml.detectors import Detector
from forge.ml.torch_data import causal_windows, prepare_windows


def validate_config(config):
    integer_bounds = {
        "schema_version": (1, 1),
        "seed": (0, 2**32 - 1),
        "window": (3, 512),
        "channels": (1, 128),
        "kernel_size": (1, 31),
        "epochs": (1, 50),
        "batch_size": (1, 4096),
        "cpu_threads": (1, 8),
        "persistence": (1, 60),
        "threshold_quantiles": (3, 1001),
    }
    float_bounds = {
        "learning_rate": (0, 1),
        "weight_decay": (0, 1),
        "max_gap_seconds": (0, 60),
        "minimum_recall": (0, 1),
    }
    if set(config) != integer_bounds.keys() | float_bounds.keys():
        raise ValueError("Unexpected or missing PyTorch configuration fields.")
    for key, (low, high) in integer_bounds.items():
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f"Invalid configuration: {key}.")
    for key, (low, high) in float_bounds.items():
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f"Invalid configuration: {key}.")
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid configuration: {key}.")
    if config["learning_rate"] <= 0 or config["max_gap_seconds"] <= 0:
        raise ValueError("Learning rate and gap limit must be positive.")
    if config["window"] < 2 * (config["kernel_size"] - 1) + 1:
        raise ValueError("Window is too short for the two convolution layers.")


class TemporalCNN(nn.Module):
    """Classify the endpoint from its past window of eight sensor channels."""

    def __init__(self, config):
        super().__init__()
        validate_config(config)
        channels, kernel = config["channels"], config["kernel_size"]
        self.network = nn.Sequential(
            nn.Conv1d(len(FEATURES), channels, kernel),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


@dataclass
class TorchDetector:
    model: TemporalCNN
    center: np.ndarray
    scale: np.ndarray
    scale_methods: list[str]
    config: dict
    name: str = "temporal_cnn_v1"

    def _windows(self, frame):
        return causal_windows(
            frame,
            self.center,
            self.scale,
            window=self.config["window"],
            max_gap_seconds=self.config["max_gap_seconds"],
        )

    def readiness(self, frame):
        return self._windows(frame)[1]

    def score(self, frame):
        x, ready = self._windows(frame)
        scores = np.zeros(len(frame), dtype=np.float64)
        indices = np.flatnonzero(ready)
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(indices), self.config["batch_size"]):
                batch = indices[start : start + self.config["batch_size"]]
                scores[batch] = self.model(torch.from_numpy(x[batch])).sigmoid().numpy()
        return scores

    def deviations(self, frame):
        return (sensor_matrix(frame) - self.center) / self.scale

    def describe(self):
        return {
            "name": self.name,
            "family": "Supervised temporal 1D CNN (PyTorch, CPU)",
            "features": FEATURES,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "scale_methods": self.scale_methods,
            "architecture": "Conv1d/ReLU/Conv1d/ReLU/mean-pool/linear",
            "window": self.config["window"],
            "score_direction": "Larger means more like training anomaly annotations; not a calibrated failure probability.",
            "initialization_requirement": (
                f"Requires {self.config['window']} consecutive readings in each source recording "
                f"and after gaps over {self.config['max_gap_seconds']} seconds. "
                "Earlier rows are unavailable."
            ),
        }


def fit_detector(training_streams, normal_training_rows, config):
    """Fit scaling on original training normals and weights on labeled train windows."""
    validate_config(config)
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    reference = Detector.fit(normal_training_rows)
    dataset = prepare_windows(
        training_streams,
        reference.center,
        reference.scale,
        window=config["window"],
        max_gap_seconds=config["max_gap_seconds"],
    )
    if len(np.unique(dataset.y)) != 2:
        raise ValueError("Training windows require both normal and anomalous targets.")
    model = TemporalCNN(config)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss()
    x, y = torch.from_numpy(dataset.x), torch.from_numpy(dataset.y)
    generator = torch.Generator().manual_seed(config["seed"])
    history = []
    for epoch in range(config["epochs"]):
        model.train()
        total = 0.0
        order = torch.randperm(len(y), generator=generator)
        for batch in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[batch]), y[batch])
            if not torch.isfinite(loss):
                raise ValueError("Training produced nonfinite loss.")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        history.append({"epoch": epoch + 1, "training_loss": total / len(y)})
    model.eval()
    detector = TorchDetector(
        model, reference.center, reference.scale, reference.scale_methods, dict(config)
    )
    return (
        detector,
        history,
        {
            "fit_rows": len(y),
            "positive_training_targets": int(dataset.y.sum()),
            "normal_training_targets": int((dataset.y == 0).sum()),
            "groups": dataset.audit,
            "endpoint_policy": "First source occurrence per leakage group; conflicting labels excluded.",
            "loss": "Unweighted binary cross entropy on deduplicated available endpoints.",
        },
    )
