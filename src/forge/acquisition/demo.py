"""Replay a permitted recording, audit capture, and compare offline detector output."""

import argparse
import asyncio
import json
import socket
from pathlib import Path

import numpy as np

from forge.acquisition.contracts import read_capture, source_frames, utc_now, wire_frame
from forge.acquisition.transport import capture, make_server
from forge.config import PROJECT_ROOT
from forge.ml.metrics import causal_alerts
from forge.ml.training import load_model


def score_frame(frame, detector, metadata):
    scores = np.asarray(detector.score(frame))
    ready = np.asarray(detector.readiness(frame))
    if scores.shape != (len(frame),) or ready.shape != scores.shape or ready.dtype != bool:
        raise ValueError("Invalid detector scores/readiness.")
    if not np.isfinite(scores[ready]).all():
        raise ValueError("Nonfinite ready scores.")
    alerts = causal_alerts(
        np.where(ready, scores, -np.inf),
        frame.datetime,
        metadata["threshold"],
        **metadata["policy"],
    )
    return scores, ready, alerts


def analyze(output, root=PROJECT_ROOT):
    frame, manifest = read_capture(output, root)
    reference, _ = source_frames(manifest["recording_id"], root)
    detector, metadata, _ = load_model(root, active=True)
    observed = score_frame(frame, detector, metadata)
    expected = score_frame(reference, detector, metadata)
    equal = {
        key: bool(np.array_equal(a, b))
        for key, a, b in zip(
            ["scores_equal", "readiness_equal", "persistent_alerts_equal"],
            observed,
            expected,
            strict=True,
        )
    }
    if not all(equal.values()):
        raise ValueError("Replay/reference detector parity failed.")
    result = dict(
        **equal,
        rows=len(frame),
        scored_rows=int(observed[1].sum()),
        alerted_rows=int(observed[2].sum()),
        unavailable_rows=int((~observed[1]).sum()),
        threshold=metadata["threshold"],
        policy=metadata["policy"],
        source_sha256=manifest["source_sha256"],
        policy_gap_count=int(
            (frame.datetime.diff().dt.total_seconds() > metadata["policy"]["max_gap_seconds"]).sum()
        ),
        source_sampling_gap_count=int((frame.datetime.diff().dt.total_seconds() > 1).sum()),
        model_run_id=metadata["run_id"],
        model_sha256=metadata["model_sha256"],
        created_utc=utc_now(),
        capture_sha256=manifest["capture_sha256"],
        initialization=detector.describe().get("initialization_requirement"),
        interpretation="Transport parity on recorded data, not detection accuracy or a physical diagnosis.",
    )
    (Path(output) / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


async def replay(recording_id, output, *, root=PROJECT_ROOT, interval=0.03, port=0):
    if not 0.01 <= interval <= 1.0:
        raise ValueError("Replay interval must be between 0.01 and 1 seconds.")
    source, record = source_frames(recording_id, root)
    if len(source) * interval + 15 > 1800:
        raise ValueError("Replay exceeds the 30-minute bound.")
    if port == 0:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
    endpoint = f"opc.tcp://127.0.0.1:{port}/forge/"
    server, node = await make_server(endpoint)
    connected = asyncio.Event()
    async with server:
        async with asyncio.TaskGroup() as group:
            task = group.create_task(
                capture(
                    endpoint,
                    recording_id,
                    record["sha256"],
                    len(source),
                    output,
                    connected=connected,
                    timeout_seconds=len(source) * interval + 15,
                )
            )
            await asyncio.wait_for(connected.wait(), timeout=10)
            for sequence in range(len(source)):
                await node.write_value(wire_frame(source, record, sequence).model_dump_json())
                await asyncio.sleep(interval)
        return task.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", default="valve1/1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.03)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Verify and score an existing capture without starting OPC UA.",
    )
    args = parser.parse_args()
    if not args.analyze_only:
        try:
            import asyncua  # noqa: F401
        except ModuleNotFoundError:
            parser.error("OPC UA replay requires: python -m pip install -r requirements-opcua.txt")
        asyncio.run(replay(args.recording, args.output, interval=args.interval, port=args.port))
    print(json.dumps(analyze(args.output), indent=2))


if __name__ == "__main__":
    main()
