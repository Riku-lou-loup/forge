# Typed investigation tools

FORGE exposes two local tools through typed request and response schemas. A
dispatcher validates requests and limits how many tools a caller can execute.
The demo on this page uses a Python script to choose the calls, so it runs without
a language model, natural-language interpretation or a generated explanation.
The Streamlit application has its own LangGraph policy workflow.

## Tools and results

`inspect_recording` accepts a recording ID such as `valve1/1`. It uses the existing
pinned-data loader, accepts training and validation recordings only, and strips
annotations and identifiers before invoking the detector. The response includes
available and unavailable row counts, alert counts, the threshold, model and data
hashes, and three sensor deviations at a selected score peak.

If alerts exist, the peak is selected among alerted readings. Otherwise, the
tool can describe the highest-scoring available reading without declaring an
alert. An entirely unavailable recording returns `insufficient_data`, no peak
and no sensor findings. Deviations refer to the pooled training reference and
are not model feature attribution or a physical diagnosis.

`search_evidence` accepts a nonblank query of at most 2,000 characters and a limit
of one to three passages. It returns verified passage text, suggested checks,
citations, corpus provenance and an explicit abstention flag. BM25 is the default
for this tool bundle. TF-IDF remains selectable, and the existing app retains
its own TF-IDF default. Retrieval does not require a loaded detector.

The application chooses the repository, retrieval backend and optional PyTorch
artifact when constructing the tools. Those choices are not tool arguments.
The input schemas forbid extra fields and coercion, so a caller cannot set an
artifact path, threshold or shell command through these tool requests.

## Run the offline demo

From the repository root, use the existing project environment:

```powershell
.\.venv\Scripts\python.exe -m forge.agents.tool_demo --schemas
.\.venv\Scripts\python.exe -m forge.agents.tool_demo --recording valve1/1
```

The schema command needs no recordings or model artifact. Inspection needs the
pinned training/validation data and an existing trained model, as described in
the project setup. To use the saved PyTorch research run:

```powershell
.\.venv\Scripts\python.exe -m forge.agents.tool_demo --recording valve1/1 --torch-artifact models/pytorch-20261007T111233Z-b8da62e6
```

That artifact name identifies the verified local run and is not bundled in Git.
A fresh clone must [train its own run](increment-demo.md) and substitute its path.
All commands accept `--root` for another project location. Relative artifact paths
are resolved from that root.

The printed JSON contains `mode: scripted_offline_tool_demo` and
`llm_called: false`. The fixed demo policy requests inspection, then searches
using the returned sensor names only if an alert exists. This routing is Python
code. A model adapter can also call the search tool independently.

## Dispatch limits and verification

A `ToolSession` accepts typed `ToolCall` objects with a call ID, a tool name and
arguments. It dispatches only the two registered names, validates their arguments,
and returns a typed result or a structured error. Unknown names, invalid tool
arguments, duplicate call IDs and tool failures consume the call budget.
Duplicate IDs do not execute the tool again.

The default budget is three calls and the supported range is one to four.
A call after the budget is exhausted raises `ToolBudgetExceeded` so the caller
can end its loop. Successful payloads are limited to 24,000 serialized characters,
and inspection is limited to 20,000 rows. These are resource bounds, not an LLM
token budget or a hard wall-clock timeout.

The trace records call IDs, tool names, results and errors. Tool failures return
a general setup message instead of copying internal exception paths into the
result. Search checks every returned passage against the retriever's corpus
before exposing it. Documents remain text; the dispatcher offers no shell,
network search, equipment control or arbitrary file-access capability.

The 25 tool tests cover measured scoring, annotation exclusion, unavailable
readings, pinned test-partition rejection, retrieval abstention and citation
tampering, input schemas, call budgets, duplicate IDs, result size, error
sanitization and the scripted demo. At that milestone, the complete suite passed 182 tests.

Real local runs also confirmed:

| Detector and recording | Inspection status | Scripted tool calls |
| --- | --- | --- |
| Active detector, valve1/1 | alert | inspection, retrieval (three passages) |
| PyTorch research run, valve1/1 | alert | inspection, retrieval (three passages) |
| PyTorch research run, valve2/1 | no_alert | inspection only |

These checks verify software behavior, not detector generalization or retrieval
quality. Model pointers, training data and existing notebooks were not changed.

## Local model integration

The separate [local LLM demo](local-llm.md) connects these tools to Ollama.
It adapts the schemas to native tool calling, bounds model requests, and checks
the generated report's structured facts and exact citation/check pairs. Its
free-form explanation remains unreviewed model text. This does not establish
complete factual grounding or resistance to prompt injection.

The scripted command above remains useful for testing tools without a language
model. The local model command requires explicit enablement on each run and does
not change the existing Streamlit or LangGraph policy workflow.
