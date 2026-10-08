# OPC UA acquisition increment

FORGE can replay a permitted SKAB recording through a local OPC UA server,
capture the measurements with a read-only client, and run the existing detector
against the verified capture. This demonstrates an acquisition boundary before
offline investigation. It does not connect to a PLC or a production installation,
and it does not send a live stream to the LLM.

## Reproduce the capture

Install the optional acquisition dependencies in the project environment. The
normal dataset setup and a trusted active model are prerequisites, as described
in [getting started](getting-started.md). Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-opcua.txt
.\.venv\Scripts\python.exe -m forge.acquisition.demo --recording valve1/1 --output artifacts/opcua/my-capture
```

The output directory must not already exist. The command binds an available
loopback port, replays at a nominal 30 ms per frame, closes the server, then
validates and scores the capture. A busy host that misses a frame produces a
failed capture rather than an interpolated record. A slower replay can be selected
with `--interval 0.1`. The source timestamps remain unchanged at either speed.

An existing capture can be checked and scored again without starting OPC UA:

```powershell
.\.venv\Scripts\python.exe -m forge.acquisition.demo --analyze-only --output artifacts/opcua/my-capture
```

This uses the currently selected trusted detector and records its run ID and
checksum. It overwrites `comparison.json`, so retain that file separately if
comparing different selected models. It does not change model pointers.

## Frame and integrity contract

The simulator exposes one read-only String variable, `Frame`, in namespace
`urn:forge:skab:replay:v1`. Its JSON payload contains the eight sensor values as
one complete measurement. This FORGE-specific schema avoids combining channels
from different updates. It is not an implementation of a vendor information
model, and a real machine would need a separate mapping and security review.
The server/client lifecycle follows the [asyncua server](https://opcua-asyncio.readthedocs.io/en/latest/usage/get-started/minimal-server.html)
and [client](https://opcua-asyncio.readthedocs.io/en/latest/usage/get-started/minimal-client.html)
interfaces. The client reads DataValues and never writes a node or invokes a
control method.

Each payload identifies the recording, its pinned SHA-256, the zero-based frame
sequence, the original measurement timestamp, and a quality flag for every
channel. The source timezone remains explicitly unspecified. Replay and receipt
times are separate UTC timestamps. Receipt timing measures this local replay,
not the delay of the original physical measurement.

Both OPC UA DataValue quality and all channel qualities must be Good. Extra
channels, annotations, missing channels, nonfinite values, changed duplicate
frames and sequence gaps are rejected. Repeated identical polling results are
ignored only within the freshness budget. Stale data and disconnection stop the
capture. The default freshness budget is two seconds, the row limit is 20,000,
and total acquisition is bounded to at most 30 minutes. This unsecured simulator
accepts only `127.0.0.1` endpoints.

`frames.jsonl` preserves every accepted frame and its receipt. `manifest.json`
records completion or failure, counts, timing and the capture checksum. The
offline reader revalidates the manifest and every frame, verifies the source via
the existing development-only loader, and compares all timestamps and values
against that source. A modified capture remains invalid even if its checksum is
recomputed. These hashes detect corruption and inconsistency. They do not
authenticate an industrial device or defend a compromised project checkout.

Analysis requires the entire recording. Missing transport frames are never
filled with zero or joined across a dropped interval. Native source gaps remain
in the timeline and the detector's existing gap policy applies. Relative-feature
models still require their initial operating reference. Their unavailable rows
remain unavailable, and a startup fault can still contaminate that reference.
The adapter supplies only `datetime` and the eight sensor columns to the model.

## Measured loopback result

On 10 October 2026, `valve1/1` was replayed with asyncua 1.1.8 on Windows and
captured in `artifacts/opcua/20261010-valve1-1`. Acquisition took 36.62 seconds at
the nominal 30 ms interval. This is one local run, not a throughput benchmark.

| Check | Result |
| --- | --- |
| Source and captured rows | 1,145 / 1,145 |
| Exact detector score equality | Passed for all rows |
| Exact readiness equality | Passed for all rows |
| Exact persistent-alert equality | Passed for all rows |
| Scored / unavailable rows | 1,085 / 60 |
| Alerted rows | 498 |
| Native intervals longer than one second | 56 |
| Intervals exceeding the detector's two-second gap policy | 0 |

The selected model was `development-20261007T091715Z-670bcc69`, with threshold
0.19899640796797832 and five-reading persistence. Its SHA-256 was
`4f05931ad8e84981e95b97ae346ec902c354dbe425e8b2ac0574ca9445d863ad`.
The source SHA-256 was
`fe4493bf805baef4e6dfb864094275d8cc2ea80a16b3735e2752b18b6aefd812`.
The comparison file includes these identifiers and the capture checksum.

Tests exercise real loopback reads, denied client writes, bad DataValue status,
incomplete frames, stalled replay and server disconnection. Contract tests cover
channel validation, duplicate and missing sequences, invalid freshness budgets,
manifest tampering, source mismatch, annotation stripping and test-partition
denial before CSV access. The acquisition dependency is optional and transport
tests skip when it is absent.

The result establishes transport parity for a simulated replay of one development
recording. It does not improve detector accuracy, establish physical failures,
or demonstrate plant interoperability. Device authentication, certificates,
vendor tag mapping, reconnect/recovery policy and a separately evaluated live
investigation path remain outside this increment.
