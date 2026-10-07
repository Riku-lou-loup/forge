# Setup and verification

Detailed commands for the local development environment. Return to the
[project overview](../README.md) for the app and results.

## Quick start

The documented environment is Windows with Python 3.13. From your cloned
repository folder, run these commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m forge.data.partitions --split train --download --audit
.\.venv\Scripts\python.exe -m forge.data.partitions --split validation --download --audit
.\.venv\Scripts\python.exe -m forge train
.\.venv\Scripts\python.exe -m forge app
```

Open [the local application](http://127.0.0.1:8511). Stop it with Ctrl+C.
The quick start creates the original baseline. To reproduce and activate the
precision-focused model, follow [the development comparison](improvement-results.md#reproduce-and-inspect).
After setup, `start.cmd` also launches the app. Use `-m forge app --port 8512`
if the default port is occupied. Always use the repository's `.venv` interpreter.

In **Investigate**, select `valve1/1`, choose **Investigate alert**, inspect the
measurements and citations, and download the draft. To record a review, enter
your name and acknowledge the evidence and limitations before selecting
**Mark reviewed**. This does not authorize equipment actions.

Data is downloaded from the pinned upstream source. Raw data and generated models
stay outside Git. An API key, database server, and hardware are not required.
The dependency snapshot records the development environment. Other platforms
have not been validated.

For a command-line draft:

```powershell
.\.venv\Scripts\python.exe -m forge investigate --recording valve1/1 --export
```

This writes an unreviewed report under `reports/incidents/`. Without `--export`,
the Markdown is printed. Only training and validation recordings are available
for interactive investigation.

To reproduce held-out evaluation after freezing selection:

```powershell
.\.venv\Scripts\python.exe -m forge.data.partitions --split test --download --audit --allow-test
.\.venv\Scripts\python.exe -m forge evaluate --allow-test
```

The result is saved beside the model and reused on subsequent evaluation calls.
Do not tune on this test set. The tracked report describes the recorded benchmark
run; later training runs do not silently replace it. Model artifacts are trusted
local pickle files: load only artifacts produced by your own training command.


## Verification

```powershell
.\scripts\check.ps1
```

If PowerShell blocks scripts, run the checks directly:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m forge doctor
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\smoke.py
```

Tests use small synthetic fixtures to verify software behavior, including actual
Streamlit review controls. The smoke check renders the app and, when a trained
artifact is present, runs an investigation on the development recording. These
checks make no model API requests. The saved [ML benchmark](evaluation.md)
uses real held-out data. Run `python -m forge.rag.evaluate` for the separately
disclosed, nine-query retrieval regression set.


## Local configuration

The Streamlit app and `forge investigate` use extractive guidance without LLM
calls. Setting provider variables in `.env` does not enable generation. The
separate [Qwen CLI](local-llm.md) requires `--enable-llm` and can use local Ollama
or the [EnsiCompute connection](ensicompute.md). Hosted API support is not yet
implemented. Keep credentials in the ignored `.env` file.

Environments, raw data, generated models and reports, local assistant configuration,
and private development notes are excluded from Git. Portable editor settings,
source manifests, tests, and public technical documentation remain tracked.

For intentional dependency updates, use the local interpreter to install
`-e ".[dev]"`, verify the environment, and run `scripts/lock.ps1` to refresh the
version snapshot.
