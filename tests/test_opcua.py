"""Wire, persistence and real loopback transport failure tests (no test CSVs)."""

import asyncio
import hashlib
import json
import socket
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from forge.acquisition import contracts
from forge.acquisition.contracts import CaptureError, Validator, read_capture, utc_now, wire_frame
from forge.acquisition.transport import capture, check_endpoint, make_server
from forge.data.datasets import FEATURES

RECORD = {"experiment_id": "valve1/1", "sha256": "a" * 64}


def source():
    # Preserve a native sampling gap, distinct from a dropped transport frame.
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2020-03-09 10:00:00", "2020-03-09 10:00:01", "2020-03-09 10:00:05"]
            ),
            **{key: [1.0, 2.0, 3.0] for key in FEATURES},
        }
    )


def raw(sequence=0):
    return wire_frame(source(), RECORD, sequence).model_dump()


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data["values"].pop(FEATURES[0]),
        lambda data: data["quality"].update({FEATURES[0]: "Bad"}),
        lambda data: data["values"].update({FEATURES[0]: float("nan")}),
        lambda data: data["values"].update(anomaly=1.0),
        lambda data: data.update(anomaly=1),
        lambda data: data.update(source_timestamp="2020-03-09T10:00:00+00:00"),
        lambda data: data.update(recording_sha256="b" * 64),
        lambda data: data.update(sequence=1),
    ],
)
def test_invalid_frame_rejected(change):
    data = raw()
    change(data)
    with pytest.raises(CaptureError):
        Validator(RECORD["experiment_id"], RECORD["sha256"]).accept(json.dumps(data), utc_now())


def test_duplicate_and_missing_frames():
    validator = Validator(RECORD["experiment_id"], RECORD["sha256"])
    first = json.dumps(raw())
    assert validator.accept(first, utc_now()) is not None
    assert validator.accept(first, utc_now()) is None
    with pytest.raises(CaptureError, match="incomplete_capture"):
        validator.accept(json.dumps(raw(2)), utc_now())
    changed = json.loads(first)
    changed["values"][FEATURES[0]] = 99.0
    with pytest.raises(CaptureError, match="changed_duplicate"):
        validator.accept(json.dumps(changed), utc_now())


def test_stale_replay_and_bad_ua_quality():
    data = raw()
    data["replay_utc"] = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    validator = Validator(RECORD["experiment_id"], RECORD["sha256"])
    with pytest.raises(CaptureError, match="stale_data"):
        validator.accept(json.dumps(data), utc_now())
    with pytest.raises(CaptureError, match="bad_quality"):
        validator.accept(json.dumps(raw()), utc_now(), ua_good=False)


@pytest.mark.parametrize(
    "endpoint", ["opc.tcp://0.0.0.0:4840", "opc.tcp://example.com:4840", "http://127.0.0.1:4840"]
)
def test_only_loopback(endpoint):
    with pytest.raises(ValueError):
        check_endpoint(endpoint)


async def transport_case(folder, fault):
    pytest.importorskip("asyncua")
    from asyncua import Client, ua

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    endpoint = f"opc.tcp://127.0.0.1:{port}/forge/"
    server, node = await make_server(endpoint)
    connected = asyncio.Event()
    await server.start()
    stopped = False
    task = asyncio.create_task(
        capture(
            endpoint,
            RECORD["experiment_id"],
            RECORD["sha256"],
            3,
            folder,
            stale_seconds=0.4,
            timeout_seconds=5,
            connected=connected,
        )
    )
    try:
        await asyncio.wait_for(connected.wait(), 4)
        # Verify the public node cannot be written by an acquisition client.
        if fault == "clean":
            async with Client(endpoint) as client:
                idx = await client.get_namespace_index("urn:forge:skab:replay:v1")
                with pytest.raises(ua.UaStatusCodeError):
                    await client.get_node(f"ns={idx};s=Frame").write_value("tamper")
        for sequence in range(3):
            data = raw(sequence)
            if sequence == 1 and fault == "quality":
                value = ua.DataValue(
                    ua.Variant(json.dumps(data)),
                    StatusCode_=ua.StatusCode(ua.StatusCodes.BadSensorFailure),
                )
                await node.write_value(value)
                break
            if sequence == 1 and fault == "incomplete":
                data["values"].pop(FEATURES[0])
            if sequence == 1 and fault == "stale":
                break
            if sequence == 1 and fault == "disconnect":
                await server.stop()
                stopped = True
                break
            await node.write_value(json.dumps(data))
            await asyncio.sleep(0.07)
        if fault == "clean":
            return await task
        with pytest.raises(CaptureError):
            await task
        assert json.loads((folder / "manifest.json").read_text())["status"] == "failed"
        with pytest.raises(CaptureError, match="incomplete or failed"):
            read_capture(folder)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if not stopped:
            await server.stop()


@pytest.mark.parametrize("fault", ["quality", "incomplete", "stale", "disconnect"])
def test_real_loopback_faults(tmp_path, fault):
    asyncio.run(transport_case(tmp_path / fault, fault))


def test_real_clean_capture_and_tampering(tmp_path, monkeypatch):
    folder = tmp_path / "clean"
    result = asyncio.run(transport_case(folder, "clean"))
    assert result["received_rows"] == 3
    monkeypatch.setattr(contracts, "source_frames", lambda *_: (source(), RECORD))
    frame, _ = read_capture(folder)
    pd.testing.assert_frame_equal(frame, source())
    manifest_path = folder / "manifest.json"
    capture_path = folder / "frames.jsonl"
    lines = capture_path.read_text().splitlines()
    item = json.loads(lines[0])
    item["frame"]["values"][FEATURES[0]] = 99.0
    lines[0] = json.dumps(item)
    payload = ("\n".join(lines) + "\n").encode()
    capture_path.write_bytes(payload)
    with pytest.raises(CaptureError, match="checksum"):
        read_capture(folder)
    manifest = json.loads(manifest_path.read_text())
    manifest["capture_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CaptureError, match="differ from"):
        read_capture(folder)


@pytest.mark.parametrize("budget", [float("nan"), float("inf"), -1.0, 31.0])
def test_invalid_freshness_budget(budget):
    with pytest.raises(ValueError, match="Freshness"):
        Validator(RECORD["experiment_id"], RECORD["sha256"], stale_seconds=budget)


@pytest.mark.parametrize(
    "change",
    [
        {"stale_seconds": float("nan")},
        {"stale_seconds": float("inf")},
        {"schema_version": 2},
        {"simulated": False},
        {"source_timezone": "UTC"},
        {"received_rows": 1},
    ],
)
def test_manifest_tampering(change):
    manifest = dict(
        schema_version=1,
        status="complete",
        recording_id="valve1/1",
        source_sha256="a" * 64,
        expected_rows=3,
        received_rows=3,
        stale_seconds=2.0,
        started_utc=utc_now(),
        finished_utc=utc_now(),
        elapsed_seconds=1.0,
        endpoint="opc.tcp://127.0.0.1:4840",
        simulated=True,
        source_timezone="unspecified",
        capture_sha256="b" * 64,
    )
    manifest.update(change)
    with pytest.raises(ValueError):
        contracts.Manifest.model_validate(manifest)


def test_test_recording_denied_before_csv_access(monkeypatch):
    from forge.agents import service

    monkeypatch.setattr(service, "inspect_recording", lambda *_: pytest.fail("CSV accessed"))
    with pytest.raises(ValueError, match="training or validation"):
        contracts.source_frames("valve1/12")


def test_source_adapter_strips_annotations(monkeypatch):
    annotated = source().assign(anomaly=1, changepoint=0)
    monkeypatch.setattr(contracts, "load_development_recording", lambda *_: (annotated, RECORD))
    frame, _ = contracts.source_frames("valve1/1")
    assert list(frame.columns) == ["datetime", *FEATURES]


def test_scoring_respects_unavailable_initialization_and_gap():
    import numpy as np

    from forge.acquisition.demo import score_frame
    from forge.ml.operating import OperatingDetector

    class Estimator:
        classes_ = [0, 1]

        def predict_proba(self, values):
            return np.tile([0.1, 0.9], (len(values), 1))

    detector = OperatingDetector(
        name="test",
        center=np.zeros(8),
        scale_methods=["test"] * 8,
        estimator=Estimator(),
        scale=np.ones(8),
        representation="relative",
        reference_readings=2,
    )
    frame = source()
    scores, ready, alerts = score_frame(
        frame, detector, {"threshold": 0.5, "policy": {"persistence": 2, "max_gap_seconds": 2}}
    )
    assert ready.tolist() == [False, False, True]
    assert alerts.tolist() == [False, False, False]
    assert scores[-1] == 0.9
    _, ready, alerts = score_frame(
        frame.iloc[:2],
        detector,
        {"threshold": 0.5, "policy": {"persistence": 2, "max_gap_seconds": 2}},
    )
    assert not ready.any() and not alerts.any()
