"""Train-only robust reference and learned Isolation Forest scores."""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

from forge.data.datasets import FEATURES, sensor_matrix


@dataclass
class Detector:
    name: str
    center: np.ndarray
    scale: np.ndarray
    scale_methods: list[str]
    forest: IsolationForest | None = None

    @classmethod
    def fit(cls, training, *, max_samples=None, seed=42, n_estimators=256):
        values = sensor_matrix(training)
        center = np.median(values, axis=0)
        mad = np.median(np.abs(values - center), axis=0)
        iqr = np.diff(np.quantile(values, [0.25, 0.75], axis=0), axis=0)[0]
        scale = np.where(mad > 0, 1.4826 * mad, iqr / 1.349)
        methods = ["MAD" if value > 0 else "IQR" for value in mad]
        if (scale <= 0).any():
            names = [FEATURES[i] for i in np.flatnonzero(scale <= 0)]
            raise ValueError(f"No robust spread for {names}; define a sensor-specific scale first.")
        forest = None
        name = "robust_max"
        if max_samples is not None:
            name = f"isolation_forest_{max_samples}"
            forest = IsolationForest(
                n_estimators=n_estimators, max_samples=max_samples, random_state=seed, n_jobs=1
            ).fit(values)
        return cls(name, center, scale, methods, forest)

    def deviations(self, frame):
        return (sensor_matrix(frame) - self.center) / self.scale

    def score(self, frame):
        if self.forest is not None:
            return -self.forest.score_samples(sensor_matrix(frame))
        return np.abs(self.deviations(frame)).max(axis=1)

    def readiness(self, frame):
        return np.ones(len(frame), dtype=bool)

    def describe(self):
        return {
            "name": self.name,
            "family": "Isolation Forest" if self.forest is not None else "Robust deviation",
            "features": FEATURES,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "scale_methods": self.scale_methods,
            "score_direction": "larger means more unusual; not a failure probability",
        }
