"""Causal representations for operating-condition-aware development experiments."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forge.data.datasets import FEATURES, sensor_matrix
from forge.ml.detectors import Detector


def causal_features(
    frame, scale, mode, *, reference_readings=60, rolling_readings=20, max_gap_seconds=2
):
    """Return features and readiness without labels, IDs, elapsed time or future rows.

    Relative mode uses an unlabeled startup reference and cannot detect a fault
    already present throughout initialization. It is not automatic calibration
    to a verified healthy state. Gaps reset rolling history but not the reference.
    """
    values = sensor_matrix(frame)
    if mode not in {"raw", "causal", "relative", "relative_noise", "relative_fraction"}:
        raise ValueError("Unknown feature representation.")
    if reference_readings < 2 or rolling_readings < 2 or max_gap_seconds <= 0:
        raise ValueError("Invalid causal feature configuration.")
    scale = np.asarray(scale, dtype=float)
    if scale.shape != (len(FEATURES),) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("A positive finite training scale is required per sensor.")
    times = pd.DatetimeIndex(frame.datetime)
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Causal features require unique, increasing timestamps.")
    ready = np.ones(len(frame), dtype=bool)
    if mode == "raw":
        return values, ready
    breaks = np.r_[True, np.diff(times.as_unit("ns").asi8) / 1e9 > max_gap_seconds]
    blocks = np.flatnonzero(breaks)
    changes = np.zeros_like(values)
    local_deviation = np.zeros_like(values)
    variation = np.zeros_like(values)
    for start, stop in zip(blocks, np.r_[blocks[1:], len(frame)], strict=True):
        segment = pd.DataFrame(values[start:stop])
        changes[start:stop] = segment.diff().fillna(0).to_numpy() / scale
        previous_median = segment.shift(1).rolling(rolling_readings, min_periods=1).median()
        local_deviation[start:stop] = (segment - previous_median).fillna(0).to_numpy() / scale
        variation[start:stop] = (
            segment.rolling(rolling_readings, min_periods=1).std(ddof=0).to_numpy() / scale
        )
    base = values
    if mode.startswith("relative"):
        ready[:reference_readings] = False
        base = np.zeros_like(values)
        if len(frame) > reference_readings:
            reference = np.median(values[:reference_readings], axis=0)
            relative_scale = scale
            if mode == "relative_noise":
                initial = values[:reference_readings]
                mad = np.median(np.abs(initial - reference), axis=0)
                iqr = np.diff(np.quantile(initial, [0.25, 0.75], axis=0), axis=0)[0]
                relative_scale = np.maximum.reduce([1.4826 * mad, iqr / 1.349, 0.05 * scale])
            elif mode == "relative_fraction":
                relative_scale = np.maximum(np.abs(reference), scale)
            base[reference_readings:] = (values[reference_readings:] - reference) / relative_scale
            for derived in (changes, local_deviation, variation):
                derived[reference_readings:] *= scale / relative_scale
    features = np.column_stack([base, changes, local_deviation, variation])
    features[~ready] = 0
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite causal feature values.")
    return features, ready


@dataclass
class OperatingDetector(Detector):
    estimator: object = None
    representation: str = "raw"
    classifier: bool = True
    reference_readings: int = 60
    rolling_readings: int = 20
    max_gap_seconds: int = 2

    def transform(self, frame):
        return causal_features(
            frame,
            self.scale,
            self.representation,
            reference_readings=self.reference_readings,
            rolling_readings=self.rolling_readings,
            max_gap_seconds=self.max_gap_seconds,
        )

    def readiness(self, frame):
        return self.transform(frame)[1]

    def score(self, frame):
        features, ready = self.transform(frame)
        scores = np.zeros(len(frame), dtype=float)
        if ready.any():
            if self.classifier:
                positive = list(self.estimator.classes_).index(1)
                scores[ready] = self.estimator.predict_proba(features[ready])[:, positive]
            else:
                scores[ready] = -self.estimator.score_samples(features[ready])
        return scores

    def describe(self):
        return {
            **super().describe(),
            "family": type(self.estimator).__name__,
            "representation": self.representation,
            "input_sensor_columns": FEATURES,
            "derived_features": len(FEATURES) * (1 if self.representation == "raw" else 4),
            "reference_readings": self.reference_readings
            if self.representation.startswith("relative")
            else 0,
            "rolling_readings": self.rolling_readings if self.representation != "raw" else 0,
            "initialization_requirement": "First reference readings must represent an appropriate operating reference; startup faults may be missed."
            if self.representation.startswith("relative")
            else "No startup reference required.",
            "score_direction": "larger means more anomaly-like; not a calibrated physical failure probability",
        }
