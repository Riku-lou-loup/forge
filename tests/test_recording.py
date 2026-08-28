"""Protect the distinction between sensor inputs, labels, and invalid recordings."""

import hashlib

import pandas as pd
import pytest

from forge.data.sample import LABEL_COLUMNS, SENSOR_UNITS, audit_recording, load_recording


@pytest.fixture
def recording(tmp_path):
    # Small synthetic fixture, never used as research or performance evidence.
    frame = pd.DataFrame({name: [1.0, 2.0, 3.0] for name in SENSOR_UNITS})
    frame["datetime"] = ["2020-01-01 00:00:00", "2020-01-01 00:00:01", "2020-01-01 00:00:03"]
    frame["anomaly"] = [0, 0, 1]
    frame["changepoint"] = [0, 0, 1]
    path = tmp_path / "recording.csv"
    frame.to_csv(path, sep=";", index=False)
    return path


def test_gaps_are_reported_without_resampling_or_treating_labels_as_sensors(recording):
    checksum = hashlib.sha256(recording.read_bytes()).hexdigest()
    frame = load_recording(recording, checksum)
    audit = audit_recording(frame)
    assert len(frame) == 3
    assert audit["interval_counts_seconds"] == {"1.0": 1, "2.0": 1}
    assert audit["anomalous_rows"] == 1
    assert set(audit["sensor_columns"]).isdisjoint(LABEL_COLUMNS)


def test_modified_recording_rejected_before_use(recording):
    checksum = hashlib.sha256(recording.read_bytes()).hexdigest()
    with recording.open("a") as output:
        output.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        load_recording(recording, checksum)


@pytest.mark.parametrize("bad_time", ["2020-01-01 00:00:00", "2019-12-31 23:59:59"])
def test_duplicate_or_reversed_time_is_not_silently_reordered(recording, bad_time):
    frame = pd.read_csv(recording, sep=";")
    frame.loc[1, "datetime"] = bad_time
    frame.to_csv(recording, sep=";", index=False)
    with pytest.raises(ValueError, match="unique and increasing"):
        load_recording(recording)


@pytest.mark.parametrize("column,value", [("anomaly", 2), ("Pressure", float("inf"))])
def test_invalid_label_or_measurement_rejected(recording, column, value):
    frame = pd.read_csv(recording, sep=";")
    frame.loc[0, column] = value
    frame.to_csv(recording, sep=";", index=False)
    with pytest.raises(ValueError):
        load_recording(recording)
