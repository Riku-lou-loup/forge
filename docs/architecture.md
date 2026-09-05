# Architecture

FORGE connects a frozen anomaly detector to a local evidence investigation. The
implemented path uses real SKAB recordings, scikit-learn, TF-IDF retrieval,
LangGraph, and Streamlit. It can run without a provider account or API key.

## Data and model boundary

The inventory pins each source recording to a revision and checksum. Loading
checks numeric values, timestamp order, schema, and metadata. The fixed split
keeps overlap-connected recordings together. Only normal training rows enter
fitting; validation chooses the detector and threshold. Explicit test evaluation
is separate from the application and returns the cached result on subsequent calls.

`data/datasets.py` constructs the eight-column feature table and deduplicates
evaluation groups. `ml/detectors.py` implements the statistical reference and
Isolation Forest. `ml/metrics.py` applies causal persistence and computes point
and event metrics. `ml/training.py` saves a local model, configuration, hashes,
library version, and validation results in a unique run directory.

Pickled models are trusted local artifacts created by the training command.
They must never be supplied as uploads or downloaded from an untrusted source.
Checksums detect accidental corruption; replacing both a model and its manifest
would bypass them. Loading checks the scikit-learn version and manifest hashes.

## Investigation graph

```mermaid
flowchart TD
    A[Validated recording and frozen detector] --> B[Observe: scores and persistent alerts]
    B --> C[Triage: decide whether evidence is needed]
    C -->|No persistent alert| N[No-alert record]
    C -->|Alert found| D[Retrieve: sensor-based query]
    K[Versioned project-authored corpus] --> D
    D -->|Evidence found| E[Draft: extract supported checks]
    D -->|No evidence, first attempt| R[Refine query once]
    R --> D
    D -->|Still no evidence| U[Insufficient-evidence record]
    E --> F[Verify: passage, version, hash and check text]
    F -->|Supported| G[Unreviewed investigation]
    F -->|Invalid citation| X[Remove unsupported guidance]
    G --> H[Human review in Streamlit]
    H --> I[Reviewed Markdown and JSON export]
    G --> J[Clearly labeled draft export]
    B -. Budget checks between nodes .-> Q[Stop with partial observations only]
```

The roles are **local policy agents**: Python makes the conditional decisions.
They do not use a language model or autonomous free-form planning. This is a
bounded agent workflow with extractive retrieval, not an LLM RAG demonstration.
LangGraph carries typed state, branches, a single permitted query refinement,
and a terminal review outcome. No benefit over a simpler implementation is
claimed; the graph makes the decision path inspectable and provides a boundary
for later model-assisted drafting.

The observation tool selects the highest-scoring persistent alert episode
without seeing labels. It returns the episode bounds and the three largest
robust sensor deviations at the score peak. These are reference comparisons,
not causal explanations or Isolation Forest feature attributions. The evidence
query uses sensor names, never the experiment's known condition or annotations.

## Retrieval and grounding

`knowledge/playbook.json` contains seven versioned analytical passages. They are
original project guidance, explicitly separate from manufacturer instructions.
The source reference documents sensor and annotation definitions; it does not
validate a fault diagnosis. Each passage is indexed with its title, sensor terms,
text and proposed analytical check using word unigrams/bigrams and TF-IDF.
Cosine similarity selects up to three passages above 0.12 similarity.

This lexical method has no embeddings, reranker, PDF ingestion, or language-model
generation. It can miss paraphrases and retrieve weak lexical matches. The
small [query set](../knowledge/retrieval-cases.json) is a disclosed development
regression set, not an independent measure of NLP generalization.

Drafting copies a supported check from each retrieved passage. Verification
requires the passage ID, version, corpus hash, title, excerpt and check text to
match the loaded corpus. A check must be paired with that exact citation.
This guarantees provenance for copied guidance; it does not prove that the
guidance diagnoses the observed event. Root-cause claims are not generated.
Documents are data: they cannot select tools, change budgets or execute commands.

## Execution and review limits

Default budgets are eight role steps, three local tool calls, 20 seconds checked
between tools, and 20,000 input rows. LangGraph also caps recursion. The time
budget is cooperative, not a hard process interrupt. Scoring counts as one tool
call, and retrieval gets at most two attempts. Budget exhaustion removes partial
guidance and preserves only measured observations. No evidence results in an
explicit abstention; invalid citations remove the draft's guidance.

No network tools, shell, equipment control, or maintenance-system writes are
available to the graph. Ambient LangSmith tracing is explicitly disabled.
Setting the reserved provider environment variables does not enable LLM calls.
Future provider integration needs an explicit opt-in and spending limit.

`reports/incident.py` defines the report schema. Every result starts unreviewed.
The UI requires a reviewer name and an explicit acknowledgment before marking a
grounded report reviewed. Review is an annotation, not approval to operate
equipment. Both draft and reviewed exports keep their status and provenance.
CLI exports remain drafts; existing files are not silently overwritten.

## Storage and next extensions

Raw data, model binaries, generated reports, credentials and local assistant
notes are ignored. Source manifests, the small original corpus, configuration,
aggregate benchmark results, and code remain tracked. The app has no server-side
account system and is intended for local use.

The next additions are a new modeling protocol focused on alert burden, applicable
equipment documentation, and a separately evaluated LLM drafting adapter. Computer
vision remains optional: nameplate recognition could identify an asset, followed
by human confirmation before selecting its documentation.

Implementation references: [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api),
[scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html),
and [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).
