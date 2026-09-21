# FORGE

Equipment anomaly investigation using sensor measurements and technical documents.

FORGE turns recorded pump measurements into an investigation draft: a trained
detector finds a persistent anomaly, a bounded workflow retrieves relevant
analytical guidance, and a reviewer can inspect the citations and export the result.
The application uses real measurements from the Skoltech Anomaly Benchmark (SKAB).

The precision-focused development detector reaches **95.1% precision** at
**65.8% recall**, reducing false-positive readings from 2,193 to 118 on validation.
It uses gradient-boosted trees and changes relative to each recording's initial
reference. These are selection results, not independent test performance; see the
[comparison and remaining limitations](docs/improvement-results.md).

The [grouped generalization audit](docs/generalization-results.md) refits detectors
across excluded recording groups and compares the current configuration with a
fixed hyperparameter search. Read it alongside the original precision figure:
that figure came from model selection on one validation allocation and does not
establish consistent performance across operating conditions.

The original frozen Isolation Forest reached **0.616 F1** and detected **18 of 23 events** on
held-out recordings. Its **145.4 false alert onsets per normal hour** make it a
research baseline, not a deployment-ready warning system. The
[evaluation report](docs/evaluation.md) explains the comparison and limitations.

The investigation runs locally with policy agents and extractive retrieval.
There are no LLM calls or generated fault diagnoses. LLM-assisted drafting and
equipment-specific documentation remain future work.

## What works today

- Validate pinned source files and keep overlapping recordings in the same split.
- Train on 18,306 unique normal observations, select on validation, and evaluate
  the original baseline on explicitly enabled test data.
- Compare supervised causal detectors on development data and activate a
  precision-focused model with an explicit 60-reading initialization requirement.
- Audit the current configuration and tuned variants with nested grouped
  cross-validation, keeping excluded groups out of fitting and threshold selection.
- Inspect sensor traces, anomaly scores, and causal persistent alerts in Streamlit.
- Retrieve versioned passages using TF-IDF and verify copied checks against citations.
- Trace the LangGraph workflow, including abstention and exhausted-budget outcomes.
- Record a named human review and export Markdown or JSON with provenance.

The 1,145-row development sample remains excluded from the held-out test set.
Source annotations are optional overlays, clearly separate from model predictions.

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
precision-focused model, follow [the development comparison](docs/improvement-results.md#reproduce-and-inspect).
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

## Stack

Python, NumPy, pandas, scikit-learn, Plotly, Streamlit, LangGraph and Pydantic
support the implemented workflow. pytest and Ruff provide software checks.
pypdf and langchain-openai remain installed for later document ingestion and
provider integration; neither is part of the current investigation path.

## Repository layout

```text
src/forge/data/       Recording loading, provenance, and validation
src/forge/ui/         Streamlit investigation, explorer and benchmark
src/forge/ml/         Detectors, selection, metrics and local artifacts
src/forge/rag/        Lexical retrieval and regression evaluation
src/forge/agents/     Bounded investigation graph and application boundary
src/forge/reports/    Structured reports, human review and export
knowledge/           Versioned original analytical passages and query fixtures
configs/             Fixed model-selection protocol
data/                Tracked manifest and ignored local data directories
docs/                Public architecture and evaluation documentation
tests/               Data, model, evidence, artifact and UI behavior
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

Tests use small synthetic fixtures to verify software behavior, including actual
Streamlit review controls. The smoke check renders the app and, when a trained
artifact is present, runs an investigation on the development recording. These
checks make no model API requests. The saved [ML benchmark](docs/evaluation.md)
uses real held-out data. Run `python -m forge.rag.evaluate` for the separately
disclosed, nine-query retrieval regression set.

## Data provenance and limitations

The recording comes from [SKAB](https://github.com/waico/SKAB), by Iurii D. Katser
and Vyacheslav O. Kozitsin. [The sample manifest](data/sample-manifest.json) records
the experiment, exact source revision, checksum, attribution, and upstream license
reference. See [data notes](data/README.md) for sensor and sampling details.

SKAB contains laboratory measurements. Results on this data do not establish
factory reliability, exact fault diagnosis, remaining useful life, or saved downtime.
Anomaly and change-point labels are source annotations and are excluded from
model inputs. Supervised development uses training anomaly annotations as targets;
the original baseline fits normal examples only. Source attribution remains part
of the public repository.

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

`.env.example` reserves future provider settings. The current pipeline makes no
LLM calls, even if those variables are set. Keep credentials in the ignored
`.env` file. Provider integration requires a separate implementation and budget.

Environments, raw data, generated models and reports, local assistant configuration,
and private development notes are excluded from Git. Portable editor settings,
source manifests, tests, and public technical documentation remain tracked.

For intentional dependency updates, use the local interpreter to install
`-e ".[dev]"`, verify the environment, and run `scripts/lock.ps1` to refresh the
version snapshot.
