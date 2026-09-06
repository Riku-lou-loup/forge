# Behavioral tests

Run `.\.venv\Scripts\python.exe -m pytest -q` from the project root.
Synthetic fixtures verify software behavior; they are not benchmark results.

The suite covers recording integrity, chronological order, finite sensor values,
split completeness, overlap leakage, explicit test access, and preserved source
annotations. Modeling tests cover label exclusion, zero-MAD fallback, causal
persistence, event matching, conflicting annotations, timestamp resolution,
reproducible forest scores, and artifact integrity before deserialization.

Workflow checks cover missing evidence, one retry, no-alert outcomes, tool and
step budgets, forged citations, draft status, named review, and duplicate exports.
Streamlit interaction tests generate a report, require acknowledgment before
review, and clear a report when its recording changes.

`scripts/check.ps1` adds dependency, environment, formatting, and application
smoke checks. The smoke test also runs the real recording through the investigation
when a local model exists. These checks make no language-model API requests.

ML performance is reported separately in [evaluation](../docs/evaluation.md).
`python -m forge.rag.evaluate` checks nine disclosed development queries; its
results do not measure general NLP or free-form generation quality.
