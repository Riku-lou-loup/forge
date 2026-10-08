"""Loopback-only OPC UA replay and read-only acquisition, with bounded failure."""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from forge.acquisition.contracts import CaptureError, Validator, utc_now

NAMESPACE = "urn:forge:skab:replay:v1"


def check_endpoint(endpoint):
    parsed = urlparse(endpoint)
    if parsed.scheme != "opc.tcp" or parsed.hostname != "127.0.0.1" or not parsed.port:
        raise ValueError("This unsecured simulator accepts only opc.tcp://127.0.0.1:<port>.")


async def make_server(endpoint):
    from asyncua import Server, ua

    check_endpoint(endpoint)
    server = Server()
    await server.init()
    server.set_endpoint(endpoint)
    server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
    server.set_server_name("FORGE recorded SKAB replay (simulated)")
    idx = await server.register_namespace(NAMESPACE)
    obj = await server.nodes.objects.add_object(idx, "Replay")
    node = await obj.add_variable(ua.NodeId("Frame", idx), "Frame", "")
    # No set_writable: acquisition clients have read access only.
    return server, node


async def capture(
    endpoint,
    recording_id,
    sha256,
    expected_rows,
    output,
    *,
    stale_seconds=2.0,
    timeout_seconds=180.0,
    connected=None,
):
    from asyncua import Client

    check_endpoint(endpoint)
    if not 1 <= expected_rows <= 20000 or not 0.1 <= stale_seconds <= 30:
        raise ValueError("Invalid capture row/freshness budget.")
    if not 0 < timeout_seconds <= 1800:
        raise ValueError("Capture timeout must be in (0, 1800] seconds.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema_version=1,
        status="running",
        recording_id=recording_id,
        source_sha256=sha256,
        expected_rows=expected_rows,
        stale_seconds=stale_seconds,
        started_utc=utc_now(),
        endpoint=endpoint,
        simulated=True,
        source_timezone="unspecified",
    )
    validator = Validator(recording_id, sha256, stale_seconds=stale_seconds)
    started = last_new = time.monotonic()
    count = 0
    failure = None
    try:
        with (output / "frames.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            async with asyncio.timeout(timeout_seconds):
                async with Client(endpoint, timeout=min(2.0, stale_seconds)) as client:
                    idx = await client.get_namespace_index(NAMESPACE)
                    node = client.get_node(f"ns={idx};s=Frame")
                    if connected is not None:
                        connected.set()
                    while count < expected_rows:
                        value = await node.read_data_value(raise_on_bad_status=False)
                        raw = value.Value.Value
                        if not value.StatusCode.is_good():
                            raise CaptureError("bad_quality: OPC UA DataValue is not Good")
                        if raw:
                            if not isinstance(raw, str) or len(raw) > 10000:
                                raise CaptureError("invalid_frame: wrong type or oversized payload")
                            receipt = validator.accept(raw, utc_now())
                            if receipt is not None:
                                stream.write(receipt.model_dump_json() + "\n")
                                stream.flush()
                                count += 1
                                last_new = time.monotonic()
                        if time.monotonic() - last_new > stale_seconds:
                            raise CaptureError("stale_data: no new complete frame within budget")
                        await asyncio.sleep(0.001)
        manifest["status"] = "complete"
    except asyncio.CancelledError:
        manifest.update(status="failed", error="cancelled: acquisition did not complete")
        raise
    except Exception as error:
        failure = CaptureError(f"{type(error).__name__}: {error}")
        manifest.update(status="failed", error=str(failure))
    finally:
        manifest.update(
            received_rows=count,
            finished_utc=utc_now(),
            elapsed_seconds=time.monotonic() - started,
            capture_sha256=hashlib.sha256((output / "frames.jsonl").read_bytes()).hexdigest(),
        )
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    if failure is not None:
        raise failure
    return manifest
