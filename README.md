# FORGE

Equipment anomaly investigation using sensor measurements and technical documents.

FORGE is being built to connect time-series anomaly detection, document retrieval,
and a reviewed maintenance report. The current application explores real pump
measurements from the Skoltech Anomaly Benchmark (SKAB).

**Status: early development.** Data loading, validation, and interactive exploration
are implemented. Anomaly detection, RAG, and agent orchestration are planned;
there are no model-performance or operational-impact results yet.

## What works today

- Download one SKAB experiment from a pinned revision and verify its SHA-256 hash.
- Validate timestamps, sensor values, and annotation columns without filling gaps.
- Explore eight sensor channels and their source-provided anomaly annotations.
- Inspect the recording's data-quality audit, provenance, and sampling intervals.
- Run loader tests and application smoke checks without a model API key.

The sample contains 1,145 observations over 20 minutes. Chart highlights are dataset
annotations, not detector predictions. This recording is reserved for development
and will be excluded from the final test set.

## Quick start

The documented environment is **Windows with Python 3.13**. From your cloned
repository folder, run these commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m forge.data.sample --download
.\.venv\Scripts\python.exe -m forge app
```

Open [the local application](http://127.0.0.1:8511). Stop it with Ctrl+C.
After setup, `start.cmd` also launches the app. Use `-m forge app --port 8512`
if the default port is occupied. Always use the repository's `.venv` interpreter.

The sample is downloaded from the upstream source; raw data is not bundled in Git.
An API key, database server, and hardware are not required for the current app.
The dependency snapshot records the development environment; other platforms
have not been validated.

## Stack

Python, NumPy, pandas, Plotly, and Streamlit support the implemented data explorer.
scikit-learn, pypdf, LangGraph, and langchain-openai are installed for planned
modeling and retrieval work. Installing these libraries does not implement those
features. pytest and Ruff cover software checks.

## Repository layout

```text
src/forge/data/       Recording loading, provenance, and validation
src/forge/ui/         Streamlit recording explorer
src/forge/ml/         Anomaly detection (planned)
src/forge/rag/        Document retrieval (planned)
src/forge/agents/     Investigation workflow (planned)
src/forge/reports/    Report export (planned)
data/                Tracked manifest and ignored local data directories
docs/                Public architecture and evaluation documentation
tests/               Recording validation tests
scripts/             Environment setup and verification
```

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

Tests use small synthetic fixtures to verify software behavior. The smoke check
exercises dependencies and renders the Streamlit app. Neither is an ML benchmark.
Checks make no model API requests.

## Data provenance and limitations

The recording comes from [SKAB](https://github.com/waico/SKAB), by Iurii D. Katser
and Vyacheslav O. Kozitsin. [The sample manifest](data/sample-manifest.json) records
the experiment, exact source revision, checksum, attribution, and upstream license
reference. See [data notes](data/README.md) for sensor and sampling details.

SKAB contains laboratory measurements. Results on this data would not establish
factory reliability, exact fault diagnosis, remaining useful life, or saved downtime.
Anomaly and change-point labels are evaluation annotations and are excluded from
model inputs. Source attribution remains part of the public repository.

## License and commercial use

Copyright (c) 2026 **Dang Duong Dang Khoa**.

FORGE's original code and documentation are licensed under
[Apache License 2.0](LICENSE). Commercial and noncommercial use are allowed
without asking for permission or paying a royalty, subject to the license terms.

Redistributions must include the license and preserve applicable copyright and
attribution notices, including the relevant author credit from [NOTICE](NOTICE),
as required by section 4. See [attribution guidance](ATTRIBUTION.md) for details,
including the distinction between redistribution and hosted-only use.

Third-party dependencies, SKAB data, and external documents retain their own licenses.

## Local configuration

`.env.example` documents future model settings with calls disabled and no key.
Keep credentials in the ignored `.env` file. Model-provider integration is pending.

Environments, raw data, generated models and reports, local assistant configuration,
and private development notes are excluded from Git. Portable editor settings,
source manifests, tests, and public technical documentation remain tracked.

For intentional dependency updates, use the local interpreter to install
`-e ".[dev]"`, verify the environment, and run `scripts/lock.ps1` to refresh the
version snapshot.
