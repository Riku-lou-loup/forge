"""Catch split leakage and accidental test access before model fitting exists."""

import copy
import hashlib
import json

import pandas as pd
import pytest

from forge.config import PROJECT_ROOT
from forge.data.partitions import (
    find_overlaps,
    inspect_recording,
    recording_path,
    select_records,
    validate_plan,
    verify_bytes,
)
from forge.data.sample import SENSOR_UNITS


@pytest.fixture
def manifests():
    return tuple(
        json.loads((PROJECT_ROOT / "data" / name).read_text(encoding="utf-8"))
        for name in ["skab-inventory.json", "skab-splits.json"]
    )


def test_inventory_matches_the_pinned_sample_and_covers_all_files(manifests):
    inventory, plan = manifests
    validate_plan(inventory, plan)
    sample = json.loads((PROJECT_ROOT / "data/sample-manifest.json").read_text())
    record = next(r for r in inventory["records"] if r["experiment_id"] == sample["experiment_id"])
    assert (record["sha256"], record["local_path"]) == (sample["sha256"], sample["local_path"])
    assert len(inventory["records"]) == 35
    assert len(inventory["overlaps"]) == 5


def test_overlap_cannot_cross_partitions_even_if_group_metadata_changes(manifests):
    inventory, plan = copy.deepcopy(manifests)
    plan["partitions"]["train"].remove("other/5")
    plan["partitions"]["test"].append("other/5")
    for record in inventory["records"]:
        record["leakage_group"] = record["experiment_id"]
    with pytest.raises(ValueError, match="Overlapping"):
        validate_plan(inventory, plan)


def test_inspected_sample_cannot_be_moved_to_test(manifests):
    inventory, plan = copy.deepcopy(manifests)
    plan["development_only"] = []
    plan["partitions"]["train"].remove("valve1/1")
    plan["partitions"]["test"].append("valve1/1")
    with pytest.raises(ValueError, match="Inspected"):
        validate_plan(inventory, plan)


@pytest.mark.parametrize("partition", ["test", "all"])
def test_test_access_requires_explicit_opt_in(manifests, partition):
    inventory, plan = manifests
    with pytest.raises(ValueError, match="allow-test"):
        select_records(inventory, plan, partition)
    assert select_records(inventory, plan, partition, allow_test=True)


@pytest.mark.parametrize("operation", ["duplicate", "omit", "unknown", "revision"])
def test_invalid_allocation_is_rejected(manifests, operation):
    inventory, plan = copy.deepcopy(manifests)
    if operation == "duplicate":
        plan["partitions"]["test"].append("valve1/1")
    elif operation == "omit":
        plan["partitions"]["test"].pop()
    elif operation == "unknown":
        plan["partitions"]["train"].append("unknown/1")
    else:
        plan["dataset_revision"] = "different"
    with pytest.raises(ValueError):
        validate_plan(inventory, plan)


def test_tampering_and_path_escape_are_rejected(tmp_path):
    record = {
        "experiment_id": "synthetic/1",
        "bytes": 3,
        "sha256": hashlib.sha256(b"abc").hexdigest(),
        "local_path": "../outside.csv",
    }
    with pytest.raises(ValueError, match="Checksum"):
        verify_bytes(b"xyz", record)
    with pytest.raises(ValueError, match="inside"):
        recording_path(record, tmp_path)


def test_overlap_detection_ignores_labels_and_detects_retimestamped_copies():
    first = pd.DataFrame({name: [1.0, 2.0] for name in SENSOR_UNITS})
    first["datetime"] = pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:00:01"])
    second = first.copy()
    second["datetime"] += pd.Timedelta(days=1)
    second["anomaly"] = [0, 1]
    result = find_overlaps({"first": first, "second": second})
    assert result[0]["shared_timestamps"] == 0
    assert result[0]["identical_sensor_vectors"] == 2


def test_normal_source_stays_unannotated(tmp_path):
    frame = pd.DataFrame({name: [1.0, 2.0] for name in SENSOR_UNITS})
    frame.insert(0, "datetime", ["2020-01-01 00:00:00", "2020-01-01 00:00:02"])
    path = tmp_path / "data/raw/skab/normal.csv"
    path.parent.mkdir(parents=True)
    frame.to_csv(path, sep=";", index=False)
    payload = path.read_bytes()
    record = {
        "experiment_id": "synthetic/normal",
        "local_path": "data/raw/skab/normal.csv",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "annotations_present": False,
        "rows": 2,
        "columns": list(frame.columns),
        "first_timestamp": frame.datetime.iloc[0],
        "last_timestamp": frame.datetime.iloc[-1],
        "missing_cells": 0,
        "duplicate_timestamps": 0,
        "interval_counts_seconds": {"2.0": 1},
    }
    loaded = inspect_recording(record, tmp_path)
    assert "anomaly" not in loaded and "changepoint" not in loaded
    assert len(loaded) == 2
