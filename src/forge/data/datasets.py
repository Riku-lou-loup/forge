"""Assemble normal training rows and deduplicated evaluation groups."""

import numpy as np
import pandas as pd

from forge.config import PROJECT_ROOT
from forge.data.partitions import inspect_recording, read_plan, select_records
from forge.data.sample import SENSOR_UNITS

FEATURES = list(SENSOR_UNITS)
OBSERVATION_KEY = ["datetime", *FEATURES]


def normal_training(root=PROJECT_ROOT):
    inventory, plan = read_plan(root)
    frames = []
    raw = excluded = 0
    for record in select_records(inventory, plan, "train"):
        frame = inspect_recording(record, root)
        raw += len(frame)
        if record["annotations_present"]:
            excluded += int(frame.anomaly.sum())
            frame = frame.loc[frame.anomaly.eq(0)].copy()
        frame["experiment_id"] = record["experiment_id"]
        frame["normal_basis"] = "annotation" if record["annotations_present"] else "source_catalog"
        frames.append(frame)
    eligible = pd.concat(frames, ignore_index=True)
    unique = eligible.drop_duplicates(OBSERVATION_KEY).reset_index(drop=True)
    return unique[FEATURES], {
        "raw_rows": raw,
        "excluded_anomalous_rows": excluded,
        "duplicate_normal_rows": len(eligible) - len(unique),
        "training_rows": len(unique),
        "source_catalog_normal_rows": int(unique.normal_basis.eq("source_catalog").sum()),
    }


def merge_group(frames):
    """One observation per group; ambiguous labels stay as unknown (-1).

    Conflicts remain in the timeline to break event/alert continuity, but do not
    enter classification metrics or threshold candidates. No majority voting.
    """
    combined = pd.concat(frames, ignore_index=True)
    conflicts = combined.groupby(OBSERVATION_KEY, dropna=False)["anomaly"].transform("nunique") > 1
    combined.loc[conflicts, "anomaly"] = -1
    merged = (
        combined.drop_duplicates(OBSERVATION_KEY).sort_values("datetime").reset_index(drop=True)
    )
    if merged.datetime.duplicated().any():
        raise ValueError(
            "Different sensor observations share a timestamp within one leakage group."
        )
    return merged, {
        "raw_rows": len(combined),
        "duplicate_rows": len(combined) - len(merged),
        "ambiguous_rows": int(merged.anomaly.eq(-1).sum()),
        "unique_rows": len(merged),
    }


def evaluation_groups(partition, *, allow_test=False, root=PROJECT_ROOT):
    if partition not in {"validation", "test"}:
        raise ValueError("Evaluation requires validation or explicitly enabled test data.")
    inventory, plan = read_plan(root)
    grouped = {}
    for record in select_records(inventory, plan, partition, allow_test=allow_test):
        grouped.setdefault(record["leakage_group"], []).append(inspect_recording(record, root))
    groups, audits = {}, {}
    for name, frames in grouped.items():
        groups[name], audits[name] = merge_group(frames)
    return groups, audits


def sensor_matrix(frame):
    """Select the fixed schema; labels and identifiers cannot become features."""
    values = frame.loc[:, FEATURES].to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Sensor observations must be nonempty and finite.")
    return values
