# Local LLM investigation

FORGE can use a local Qwen model to request recording inspection, search the
analytical notes with BM25, and draft an explanation. The model chooses the tool
calls and search query. Python validates and executes those requests, then checks
the report's structured measurements and cited checks before accepting a draft.

This is a separate command-line demo. The Streamlit application and its existing
LangGraph policy workflow keep their current behavior. The local LLM does not
replace the anomaly detector or change the active model.

## Run an investigation

Use the project virtual environment and the existing pinned SKAB data, trained
detector and playbook. The [increment demo](increment-demo.md) explains those
prerequisites. No paid API key or additional Python package is needed here.

Install [Ollama for Windows](https://docs.ollama.com/windows), or unpack the official
portable Windows release into `.cache/ollama/runtime/` so that it contains
`ollama.exe`. From the repository root, start the local server:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-ollama.ps1
```

This command permits this script for the launched PowerShell process without
changing the system execution policy. The script finds the portable runtime
first, then tries `ollama` on PATH. It binds the server to `127.0.0.1:11434`, disables Ollama cloud features, and keeps model
files under the ignored `.cache/ollama/models/` directory. Paths are resolved from
the script's location. An existing server is left running with its existing
settings. Use a separate port if you need an isolated server.

For the portable runtime, download the model once:

```powershell
.\.cache\ollama\runtime\ollama.exe pull qwen3.5:4b
```

For an installation on PATH, use `ollama pull qwen3.5:4b` instead. These commands
contact the model registry to download weights. Investigation requests go only
to the local server, and the adapter refuses remote-backed models.

Run the demo after the download finishes:

```powershell
.\.venv\Scripts\python.exe -m forge.agents.local_demo --enable-llm --recording valve1/1 --export
```

The terminal shows a Markdown draft. `--export` saves that draft and a JSON trace
under `reports/incidents/`, which is ignored by Git. The trace includes the local
model digest and Ollama version, model replies and tool requests, full tool
results with data/model/corpus provenance, elapsed time, and validation outcome.
Exit code 0 means a draft passed the structural checks. It does not mean a human
has reviewed the explanation. A blocked investigation exits with code 2.

Use `--question "..."` to change the question, `--backend tfidf` to change the
retriever, or `--torch-artifact models/<your-run>` for an explicit research model.
`--root` supports another checkout and `--port` selects another local Ollama port.
The selected recording remains fixed for the run. Only training and validation
recordings are accepted by the inspection tool.

The command requires `--enable-llm` every time. It does not change `.env`, use
`OPENAI_API_KEY`, download a model automatically, or enable LLM calls in the app.

## What the model controls

The model receives the two existing [tool schemas](agent-tools.md). It can request
`inspect_recording` for the selected recording and `search_evidence` with a query
of up to 2,000 characters. An alert must have a completed evidence search before
the application attempts a report. An empty search result remains an evidence
gap. For no-alert or insufficient-data results, the application ends tool selection
and proceeds directly to a draft with no alert-specific checks.

After tool selection ends, a separate request asks the model for a JSON report.
This keeps the report schema separate from native tool calling and avoids carrying
the entire conversation into synthesis. The final request receives measured
findings and retrieved passages. The detector produces the scores and alerts.
The language model supplies the explanation and selects cited analytical checks.

The dispatcher exposes no shell, web search, arbitrary file access or equipment
control. Tool arguments cannot change thresholds, select artifact paths or load
another detector. The application supplies those settings.

## Validation and limits

The report validator checks the recording ID, alert status, alerted-row count and
unavailable-row count against the actual inspection. Every cited check must match
the text of a retrieved passage's check exactly. Invented citations, changed check
text, duplicate citations and malformed reports block the draft. Alert-specific
checks are rejected for no-alert and insufficient-data results.

The explanation is free-form model text. Its factual support is not established
by these checks, so it appears under **Model explanation (unreviewed)**. A valid
citation establishes where a check came from. It does not establish that the
passage is relevant, that every sentence is supported, or that the detector found
a physical failure. Retrieved material consists of project-authored analytical
notes, not manufacturer instructions. Prompt instructions and an allowlist do not
prove general resistance to prompt injection.

The default model is `qwen3.5:4b`, using a 4,096-token context, temperature 0,
seed 42, presence penalty 0 and thinking output disabled. Tool-selection requests
allow 512 output tokens and synthesis allows 768. The loop permits up to four
tool-selection requests, three tool executions (one per reply), and one synthesis
request. A 600-second deadline is checked between requests and bounds subsequent HTTP timeouts, with a maximum
of 180 seconds per chat request. This is a cooperative deadline, not a hard kill
of a running Python tool or an Ollama generation after a client disconnect.

Requests have a 14,000-character ceiling and responses a 256 KiB ceiling. The
adapter also checks the reported prompt token count for context headroom. The
character limit is not an exact tokenizer calculation and cannot prove that a
server never truncated input. Larger documents need a separate context-budget
design. The current small playbook keeps this demo bounded.

The command unloads Qwen in a `finally` block after success or failure. Requests
also use a 30-second idle keep-alive as a fallback if the client exits unexpectedly.
The command checks Windows memory before starting: it requires at least 6 GiB
of available physical RAM and 8 GiB of remaining system commit capacity. These
are conservative demo guardrails, not a guarantee against memory exhaustion.
Other processes can allocate memory after the check. The Windows check is skipped
on other operating systems. Direct users of the Python client must manage their
own memory preflight and model lifecycle.

Keep model inference separate from full regression tests and model training.
During development, running the full suite while Qwen remained loaded coincided
with a system-wide freeze and Windows error 1455 (paging-file capacity exhausted).
The exact cause of the black screen was not established. That interrupted test
run is not counted as a pass. Ollama was left stopped after recovery. Automatic
unloading and the headroom check were added afterward and tested offline.

To unload the portable model manually without stopping the server:

```powershell
.\.cache\ollama\runtime\ollama.exe stop qwen3.5:4b
```

## Verification

Offline tests exercise model-selected calls against the real inspection and BM25
tools using a scripted provider. They cover bad counts and citations, altered
checks, malformed calls, recording restrictions, budgets, abstention, no-alert
routing, timeouts, local-only transport and incomplete model responses. These are
software behavior tests, not evidence of Qwen's reasoning quality. The latest
focused run passed 64 tests covering local integration and the existing tools,
including low-memory refusal and cleanup after success and failure. An earlier
full-suite run passed 214 tests before the final routing and memory changes.
The later full-suite run was interrupted by the freeze and has not been repeated.

Two final development smoke runs on 7 October 2026 returned structurally valid
drafts with policy `local-agent-v4`. Both ran on the RTX 4050 laptop GPU through
Ollama 0.40.0, using Qwen3.5 4B Q4_K_M and the same default question.

| Detector and recording | Tool calls | Measured outcome | Elapsed time |
| --- | --- | --- | --- |
| Active gradient boosting, `valve1/1` | Inspection, BM25 search | 498 alerted rows, 60 unavailable; three cited checks | 13.657 s |
| Research PyTorch CNN, `valve2/1` | Inspection only | No alerts, 62 unavailable; no alert-specific checks | 5.337 s |

The local traces are `local-llm-5b552b4f9bd846fabf0e0d18b21422fb.json` and
`local-llm-7a4cda16ae6f4f649e1a3462ad176bb2.json` under `reports/incidents/`.
They are generated files and are not included in a clone. The PyTorch run used
`models/pytorch-20261007T111233Z-b8da62e6`; substitute your own trained artifact
when reproducing that path. Both traces record model digest
`d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04`.

Earlier development attempts exposed a truncated tool-selection reply, an
over-budget search request, wording that implied sensor attribution, and
alert-specific guidance in a no-alert draft. The prompt was revised, and the
application now ends tool selection after a no-alert or insufficient-data
inspection and rejects alert-specific checks for those results. One intermediate
run also failed when the Ollama inference worker closed its connection. It
returned a blocked result with no accepted draft. No automatic retry hides such
failures.

These two smoke runs helped shape the implementation and are not an independent
evaluation set. Their timings are individual warm-run observations, not a
latency benchmark. The model's prose still needs review for implied causality
and relevance. The no-alert recording is a known detector miss, so this result
demonstrates routing rather than successful failure detection.

## Implementation references

- [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Ollama chat API](https://docs.ollama.com/api/chat)
- [Qwen3.5 4B model listing](https://ollama.com/library/qwen3.5:4b)

The implementation follows the native Ollama HTTP protocol and uses the Python
standard library for transport. `ollama_client.py` handles local requests,
`local_agent.py` manages the tool loop and report validation, and `local_demo.py`
provides the runnable command. `local_memory.py` checks Windows headroom.
