"""Inspect a source-tracked development recording without training a model."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from forge.config import PROJECT_ROOT

SENSOR_UNITS = {
    "Accelerometer1RMS": "g",
    "Accelerometer2RMS": "g",
    "Current": "A",
    "Pressure": "bar",
    "Temperature": "°C",
    "Thermocouple": "°C",
    "Voltage": "V",
    "Volume Flow RateRMS": "L/min",
}
LABEL_COLUMNS = ("anomaly", "changepoint")


def load_recording(path: Path, expected_sha256: str | None = None) -> pd.DataFrame:
    """Read a SKAB CSV, preserving its order and keeping labels separate from sensors."""
    if expected_sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Recording checksum differs from the source manifest.")
    frame = pd.read_csv(path, sep=";")
    required = {"datetime", *SENSOR_UNITS, *LABEL_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("The recording has no rows.")
    frame["datetime"] = pd.to_datetime(frame["datetime"], format="%Y-%m-%d %H:%M:%S")
    if frame["datetime"].isna().any():
        raise ValueError("The recording contains missing timestamps.")
    if not frame["datetime"].is_monotonic_increasing or frame["datetime"].duplicated().any():
        raise ValueError("Timestamps must be unique and increasing; raw data was not reordered.")
    numeric = [*SENSOR_UNITS, *LABEL_COLUMNS]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("The recording contains missing or non-finite measurements or labels.")
    for label in LABEL_COLUMNS:
        if not frame[label].isin([0, 1]).all():
            raise ValueError(f"{label} must contain only 0 or 1.")
        frame[label] = frame[label].astype(int)
    return frame


def sample_manifest() -> dict:
    return json.loads((PROJECT_ROOT / "data/sample-manifest.json").read_text(encoding="utf-8"))


def sample_path(manifest: dict) -> Path:
    path = (PROJECT_ROOT / manifest["local_path"]).resolve()
    if (PROJECT_ROOT / "data/raw").resolve() not in path.parents:
        raise ValueError("Sample path must remain inside data/raw.")
    return path


def load_sample() -> pd.DataFrame:
    manifest = sample_manifest()
    return load_recording(sample_path(manifest), manifest["sha256"])


def audit_recording(frame: pd.DataFrame) -> dict:
    intervals = frame["datetime"].diff().dt.total_seconds().dropna()
    return {
        "rows": len(frame),
        "sensor_columns": list(SENSOR_UNITS),
        "annotation_columns": list(LABEL_COLUMNS),
        "first_timestamp": str(frame["datetime"].iloc[0]),
        "last_timestamp": str(frame["datetime"].iloc[-1]),
        "median_interval_seconds": float(intervals.median()) if len(intervals) else None,
        "interval_counts_seconds": {str(k): int(v) for k, v in intervals.value_counts().items()},
        "missing_cells": int(frame.isna().sum().sum()),
        "anomalous_rows": int(frame["anomaly"].sum()),
        "normal_rows": int((frame["anomaly"] == 0).sum()),
        "changepoint_rows": int(frame["changepoint"].sum()),
        "purpose": "Development exploration only; labels are annotations, not predictions.",
    }


def download_sample() -> Path:
    """Explicitly fetch the pinned public file, checking its hash before saving."""
    manifest = sample_manifest()
    path = sample_path(manifest)
    if path.exists():
        load_recording(path, manifest["sha256"])
        return path
    expected_url = (
        f"https://raw.githubusercontent.com/waico/SKAB/{manifest['revision']}/data/valve1/1.csv"
    )
    if manifest["source_url"] != expected_url:
        raise ValueError("The sample source URL does not match the pinned recording.")
    request = Request(expected_url, headers={"User-Agent": "FORGE/0.1"})
    with urlopen(request, timeout=30) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
        raise ValueError("Downloaded content differs from the source manifest; nothing saved.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(payload)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch the pinned sample if absent")
    args = parser.parse_args()
    if args.download:
        download_sample()
    try:
        frame = load_sample()
    except FileNotFoundError:
        parser.error("Sample missing. Run again with --download to fetch the pinned public file.")
    print(json.dumps(audit_recording(frame), indent=2))
    print("\nFirst five rows:")
    print(frame.head().to_string(index=False))


if __name__ == "__main__":
    main()
