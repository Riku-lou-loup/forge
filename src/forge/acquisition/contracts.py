"""Strict simulator wire format and independently checked capture provenance."""

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge.agents.service import load_development_recording
from forge.config import PROJECT_ROOT
from forge.data.datasets import FEATURES


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def utc_time(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("Replay and receipt timestamps must explicitly be UTC.")
    return parsed


class Frame(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    schema_version: Literal[1] = 1
    recording_id: str = Field(min_length=3, max_length=100)
    recording_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sequence: int = Field(ge=0, lt=20000)
    source_timestamp: str
    source_timezone: Literal["unspecified"] = "unspecified"
    replay_utc: str
    values: dict[str, float]
    quality: dict[str, Literal["Good"]]

    @model_validator(mode="after")
    def validate_frame(self):
        if set(self.values) != set(FEATURES) or set(self.quality) != set(FEATURES):
            raise ValueError(
                "Every frame requires exactly the eight sensor channels and qualities."
            )
        source = datetime.fromisoformat(self.source_timestamp)
        if source.tzinfo is not None:
            raise ValueError("SKAB source timezone is unspecified; do not assign UTC.")
        utc_time(self.replay_utc)
        return self


class Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    frame: Frame
    received_utc: str
    ua_status: Literal["Good"]


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    schema_version: Literal[1]
    status: Literal["complete"]
    recording_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_rows: int = Field(ge=1, le=20000)
    received_rows: int = Field(ge=1, le=20000)
    stale_seconds: float = Field(ge=0.1, le=30)
    started_utc: str
    finished_utc: str
    elapsed_seconds: float = Field(ge=0, le=1810)
    endpoint: str
    simulated: Literal[True]
    source_timezone: Literal["unspecified"]
    capture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_manifest(self):
        if self.received_rows != self.expected_rows:
            raise ValueError("Expected and received frame counts differ.")
        if utc_time(self.finished_utc) < utc_time(self.started_utc):
            raise ValueError("Capture clock regressed.")
        return self


class CaptureError(ValueError):
    """An acquisition failure prevents this capture from being scored."""


class Validator:
    def __init__(self, recording_id, sha256, *, stale_seconds=2.0):
        if not math.isfinite(stale_seconds) or not 0.1 <= stale_seconds <= 30:
            raise ValueError("Freshness budget must be finite and in [0.1, 30] seconds.")
        self.recording_id = recording_id
        self.sha256 = sha256
        self.stale_seconds = stale_seconds
        self.last = None
        self.last_receipt = None

    def accept(self, raw, received_utc, ua_good=True):
        if not ua_good:
            raise CaptureError("bad_quality: OPC UA DataValue status is not Good")
        try:
            frame = Frame.model_validate_json(raw)
            received = utc_time(received_utc)
        except ValueError as error:
            raise CaptureError(f"invalid_frame: {error}") from error
        if (frame.recording_id, frame.recording_sha256) != (self.recording_id, self.sha256):
            raise CaptureError("identity_mismatch")
        age = (received - utc_time(frame.replay_utc)).total_seconds()
        if age < -0.1 or age > self.stale_seconds:
            raise CaptureError("stale_data: replay timestamp is outside the freshness budget")
        if self.last_receipt is not None and received < self.last_receipt:
            raise CaptureError("receipt_clock_regressed")
        if self.last is not None and frame.sequence == self.last.sequence:
            if frame != self.last:
                raise CaptureError("changed_duplicate: same frame identity has different contents")
            return None
        expected = 0 if self.last is None else self.last.sequence + 1
        if frame.sequence != expected:
            raise CaptureError(
                f"incomplete_capture: expected frame {expected}, received {frame.sequence}"
            )
        if self.last and datetime.fromisoformat(frame.source_timestamp) <= datetime.fromisoformat(
            self.last.source_timestamp
        ):
            raise CaptureError("source_timestamp_order")
        if self.last and utc_time(frame.replay_utc) < utc_time(self.last.replay_utc):
            raise CaptureError("replay_clock_regressed")
        self.last, self.last_receipt = frame, received
        return Receipt(frame=frame, received_utc=received_utc, ua_status="Good")


def source_frames(recording_id, root=PROJECT_ROOT):
    source, record = load_development_recording(recording_id, root)
    if not 1 <= len(source) <= 20000:
        raise ValueError("Recording outside the 1–20000 row budget.")
    return source.loc[:, ["datetime", *FEATURES]].copy(), record


def wire_frame(source, record, sequence):
    row = source.iloc[sequence]
    return Frame(
        recording_id=record["experiment_id"],
        recording_sha256=record["sha256"],
        sequence=sequence,
        source_timestamp=row.datetime.isoformat(),
        replay_utc=utc_now(),
        values={key: float(row[key]) for key in FEATURES},
        quality={key: "Good" for key in FEATURES},
    )


def read_capture(folder, root=PROJECT_ROOT):
    """Verify disk bytes, all frames and their exact pinned development-source values.

    A checksum is corruption evidence, not authentication. Comparing against the
    pinned source also rejects an edited capture with a recomputed checksum.
    """
    folder = Path(folder)
    manifest_path = folder / "manifest.json"
    if manifest_path.stat().st_size > 10000:
        raise CaptureError("Oversized manifest.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise CaptureError("Capture is incomplete or failed; offline scoring refused.")
    manifest = Manifest.model_validate(manifest).model_dump()
    capture_path = folder / "frames.jsonl"
    if capture_path.stat().st_size > 40_000_000:
        raise CaptureError("Oversized capture.")
    payload = capture_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["capture_sha256"]:
        raise CaptureError("Capture checksum mismatch.")
    source, record = source_frames(manifest["recording_id"], root)
    if manifest["source_sha256"] != record["sha256"]:
        raise CaptureError("Source checksum mismatch.")
    validator = Validator(
        record["experiment_id"], record["sha256"], stale_seconds=manifest["stale_seconds"]
    )
    rows = []
    for line in payload.decode("utf-8").splitlines():
        receipt = Receipt.model_validate_json(line)
        accepted = validator.accept(receipt.frame.model_dump_json(), receipt.received_utc)
        if accepted is None:
            raise CaptureError("Duplicate frame in persisted capture.")
        rows.append({"datetime": receipt.frame.source_timestamp, **receipt.frame.values})
    if len(rows) != len(source) or len(rows) != manifest["expected_rows"]:
        raise CaptureError("Full recording required; missing rows cannot be bridged.")
    frame = pd.DataFrame(rows, columns=["datetime", *FEATURES])
    frame["datetime"] = pd.to_datetime(frame.datetime)
    if not frame.datetime.equals(source.datetime.reset_index(drop=True)) or not np.array_equal(
        frame[FEATURES].to_numpy(), source[FEATURES].to_numpy()
    ):
        raise CaptureError("Captured measurements differ from the verified recording.")
    return frame, manifest
