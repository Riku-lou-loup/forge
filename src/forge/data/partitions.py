"""Validate the fixed SKAB split and fetch or audit source recordings explicitly."""

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from forge.config import PROJECT_ROOT
from forge.data.sample import LABEL_COLUMNS, SENSOR_UNITS, load_recording

PARTITIONS = ("train", "validation", "test")


def validate_plan(inventory: dict, plan: dict, protected_experiment: str = "valve1/1") -> None:
    """Reject incomplete allocations and known cross-partition leakage before use."""
    if inventory["revision"] != plan["dataset_revision"]:
        raise ValueError("Inventory and split revisions differ.")
    records = inventory["records"]
    ids = [r["experiment_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Inventory contains duplicate experiment IDs.")
    parts = plan["partitions"]
    if set(parts) != set(PARTITIONS) or any(not parts[p] for p in PARTITIONS):
        raise ValueError("Exactly three nonempty partitions are required.")
    assigned = [key for part in PARTITIONS for key in parts[part]]
    if Counter(assigned) != Counter(ids):
        raise ValueError("Every recording must appear exactly once across partitions.")
    protected = set(plan["development_only"]) | {protected_experiment}
    if not protected <= set(parts["train"]):
        raise ValueError("Inspected development recordings must remain in train.")
    membership = {key: part for part in PARTITIONS for key in parts[part]}
    groups = {}
    for record in records:
        groups.setdefault(record["leakage_group"], set()).add(membership[record["experiment_id"]])
        key = record["experiment_id"]
        if record["source_path"] != f"data/{key}.csv":
            raise ValueError("Unexpected source path.")
        expected = (
            f"https://raw.githubusercontent.com/waico/SKAB/{inventory['revision']}/data/{key}.csv"
        )
        if record["source_url"] != expected:
            raise ValueError("Source URL differs from the pinned recording.")
    if any(len(parts) != 1 for parts in groups.values()):
        raise ValueError("A leakage group crosses partitions.")
    for pair in inventory["overlaps"]:
        if membership[pair["left"]] != membership[pair["right"]]:
            raise ValueError("Overlapping recordings cross partitions.")


def read_plan(root: Path = PROJECT_ROOT) -> tuple[dict, dict]:
    inventory = json.loads((root / "data/skab-inventory.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "data/skab-splits.json").read_text(encoding="utf-8"))
    validate_plan(inventory, plan)
    return inventory, plan


def select_records(
    inventory: dict, plan: dict, partition: str, *, allow_test: bool = False
) -> list:
    validate_plan(inventory, plan)
    if partition not in (*PARTITIONS, "all"):
        raise ValueError("Unknown partition.")
    if partition in {"test", "all"} and not allow_test:
        raise ValueError("Test access requires --allow-test; keep model selection on validation.")
    ids = (
        {key for values in plan["partitions"].values() for key in values}
        if partition == "all"
        else set(plan["partitions"][partition])
    )
    return [record for record in inventory["records"] if record["experiment_id"] in ids]


def recording_path(record: dict, root: Path = PROJECT_ROOT) -> Path:
    path = (root / record["local_path"]).resolve()
    if (root / "data/raw/skab").resolve() not in path.parents:
        raise ValueError("Recording path must stay inside data/raw/skab.")
    return path


def verify_bytes(payload: bytes, record: dict) -> None:
    if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError(f"Checksum or size mismatch: {record['experiment_id']}")


def download_recording(record: dict, root: Path = PROJECT_ROOT) -> Path:
    """Retain existing files; validate incoming bytes before any write."""
    path = recording_path(record, root)
    if path.exists():
        verify_bytes(path.read_bytes(), record)
        return path
    request = Request(record["source_url"], headers={"User-Agent": "FORGE/0.1"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    verify_bytes(payload, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(payload)
    return path


def inspect_recording(record: dict, root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Check metadata; preserve the normal-operation file's absent annotations."""
    path = recording_path(record, root)
    verify_bytes(path.read_bytes(), record)
    if record["annotations_present"]:
        frame = load_recording(path, record["sha256"])
    else:
        frame = pd.read_csv(path, sep=";")
        if set(frame.columns) != {"datetime", *SENSOR_UNITS}:
            raise ValueError("Unexpected schema in the unannotated normal recording.")
        frame["datetime"] = pd.to_datetime(frame["datetime"], format="%Y-%m-%d %H:%M:%S")
        if (
            frame.empty
            or frame.datetime.isna().any()
            or not frame.datetime.is_monotonic_increasing
            or frame.datetime.duplicated().any()
        ):
            raise ValueError("Invalid normal-recording timestamps.")
        if not np.isfinite(frame[list(SENSOR_UNITS)].to_numpy()).all():
            raise ValueError("Invalid normal-recording measurements.")
    intervals = frame.datetime.diff().dt.total_seconds().dropna()
    observed = {
        "rows": len(frame),
        "columns": list(frame.columns),
        "annotations_present": set(LABEL_COLUMNS) <= set(frame.columns),
        "first_timestamp": str(frame.datetime.iloc[0]),
        "last_timestamp": str(frame.datetime.iloc[-1]),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_timestamps": int(frame.datetime.duplicated().sum()),
        "interval_counts_seconds": {
            str(k): int(v) for k, v in intervals.value_counts().sort_index().items()
        },
    }
    for key, value in observed.items():
        if record[key] != value:
            raise ValueError(f"Inventory metadata differs: {record['experiment_id']} / {key}")
    return frame


def find_overlaps(frames: dict[str, pd.DataFrame]) -> list[dict]:
    """Detect exact repeated measurements, including transitive group edges."""
    times = {key: set(frame.datetime) for key, frame in frames.items()}
    sensors = {
        key: set(frame[list(SENSOR_UNITS)].itertuples(index=False, name=None))
        for key, frame in frames.items()
    }
    rows = {
        key: set(frame[["datetime", *SENSOR_UNITS]].itertuples(index=False, name=None))
        for key, frame in frames.items()
    }
    overlaps = []
    for left, right in combinations(sorted(frames), 2):
        time_count = len(times[left] & times[right])
        sensor_count = len(sensors[left] & sensors[right])
        if time_count or sensor_count:
            overlaps.append(
                {
                    "left": left,
                    "right": right,
                    "shared_timestamps": time_count,
                    "identical_sensor_vectors": sensor_count,
                    "identical_timestamp_sensor_rows": len(rows[left] & rows[right]),
                }
            )
    return overlaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=(*PARTITIONS, "all"), default="train")
    parser.add_argument("--download", action="store_true", help="Fetch selected missing files")
    parser.add_argument(
        "--audit", action="store_true", help="Verify local source integrity and metadata"
    )
    parser.add_argument(
        "--allow-test", action="store_true", help="Explicitly permit test-file access"
    )
    args = parser.parse_args()
    inventory, plan = read_plan()
    try:
        records = select_records(inventory, plan, args.split, allow_test=args.allow_test)
        frames = {}
        for record in records:
            if args.download:
                download_recording(record)
            if args.audit:
                frames[record["experiment_id"]] = inspect_recording(record)
        if args.audit:
            ids = set(frames)
            expected = [p for p in inventory["overlaps"] if p["left"] in ids and p["right"] in ids]
            if find_overlaps(frames) != expected:
                raise ValueError("Observed overlap differs from the frozen inventory.")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "split_id": plan["split_id"],
                "partition": args.split,
                "recordings": len(records),
                "rows_before_filtering_or_deduplication": sum(r["rows"] for r in records),
                "files_audited": len(frames),
                "experiment_ids": [r["experiment_id"] for r in records],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
