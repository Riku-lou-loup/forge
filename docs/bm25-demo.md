# BM25 retrieval demo

FORGE can now retrieve its analytical notes with either TF-IDF or positive-IDF
Okapi BM25. TF-IDF remains the default. Both backends return the same `Evidence`
objects and use the existing citation verification and investigation workflow.
No model, model pointer, recording, annotation, or corpus passage changes for this
demo.

The corpus contains seven project-authored passages about reviewing recorded pump
measurements. It is not a collection of manufacturer manuals, and its contents do
not establish a physical fault diagnosis. The nine queries in
`knowledge/retrieval-cases.json` are the original development regression fixture,
authored with the corpus. They are not an independent NLP benchmark.

## Run the comparison

From the repository root, using the project environment:

```powershell
.\.venv\Scripts\python.exe -m forge.rag.evaluate --compare
.\.venv\Scripts\python.exe -m forge.rag.evaluate --backend bm25 --query "flow pressure"
.\.venv\Scripts\python.exe -m forge.rag.evaluate --compare --query "renaissance sonnet planetary astronomy"
```

The module also accepts `--root PATH`, so the same commands work outside the
repository when the project interpreter can import FORGE. For an isolated
worktree, point `PYTHONPATH` at that worktree's `src` directory before using the
shared project interpreter. No model artifact, raw recording, API credential, or
network access is needed for these retrieval commands.

The JSON includes the backend, threshold, BM25 parameters, corpus hash, rankings,
and verification results. The query form includes the original passage text,
check, citation, and an explicit abstention flag. Running without arguments still
evaluates TF-IDF. Use `--k1` and `--b` to explore BM25 parameters. A
`--minimum-score` override applies to one selected backend and cannot be combined
with `--compare`, because their score scales differ.

The Python API exposes the same choice:

```python
from forge.agents.service import investigate_recording
from forge.rag.retrieval import Retriever

retriever = Retriever(backend="bm25", k1=1.2, b=0.75)
evidence = retriever.search("flow pressure")
assert all(retriever.verifies(item) for item in evidence)

# Requires the existing local model and pinned development recording.
report = investigate_recording("valve1/1", retrieval_backend="bm25")
```

Selecting BM25 changes passage ranking only. The detector, bounded graph, human
review requirement, and removal of unsupported guidance stay in place. The
service still uses TF-IDF when `retrieval_backend` is omitted.

## Scoring and abstention

For background on BM25, see Robertson and Zaragoza (2009),
[The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019).
FORGE uses the explicit positive-IDF variant below. For each unique query
token, the score adds

```text
IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
contribution(t, d) = IDF(t) * tf(t, d) * (k1 + 1)
                    / (tf(t, d) + k1 * (1 - b + b * length(d) / average_length))
```

`N` is the passage count, `df` counts passages containing the token, and `tf`
counts occurrences within one passage. The positive IDF keeps even common terms
nonnegative. With the defaults `k1=1.2` and `b=0.75`, repeated document terms help
but saturate, and longer documents receive a length adjustment. Setting `b=0`
disables that adjustment. `k1` must be finite and positive, and `b` must be finite
and between zero and one.

Each document joins the title, sensor names, text, and suggested check, matching
the fields used by TF-IDF. BM25 uses lowercase unigrams with the existing English
stop-word and word-token rules: tokens have at least two word characters.
Document lengths count those retained tokens. There is no stemming, semantic
embedding, or synonym expansion. Repeating a query token does not raise its
weight. Equal scores preserve corpus order, and query terms are summed in sorted
order for reproducibility. TF-IDF retains its original unigrams, bigrams,
sublinear term frequency, and cosine scores.

BM25's default minimum score is `0.5`, a separate development policy chosen to
filter weak lexical matches in this small corpus. It is not calibrated confidence
and has not been established for a larger document collection. TF-IDF keeps its
original `0.12` threshold. Scores and thresholds must not be compared numerically
between backends. Search returns at most three passages by default, permits a
limit from one to five, and preserves the 2,000-character query budget.

Empty queries, stop-word-only queries, and queries with no exact token overlap
abstain. Even an explicit threshold of zero cannot emit zero-score evidence.
An all-empty token corpus also abstains with BM25. Citation verification checks
passage identity, version, corpus hash, text, title, and suggested check. It
establishes correspondence with this local corpus, not the relevance or physical
correctness of a passage.

## Observed comparison

The run on 7 October 2026 used the unchanged corpus with SHA-256
`dfc2c6f41aa1fcb54501dc66876ec7421026155192f1fa58ab75b4c4b6776421`.
The defaults above were used for both backends.

| Measure | TF-IDF | BM25 |
| --- | ---: | ---: |
| Recall at 3, seven answerable cases | 1.000 | 1.000 |
| Mean reciprocal rank within the returned top 3, seven answerable cases | 1.000 | 1.000 |
| Correct abstentions | 2 / 2 | 2 / 2 |
| Verified citations / returned citations | 7 / 7 | 15 / 15 |

Recall measures the fraction of a case's expected passage IDs retrieved. Reciprocal
rank is the inverse rank of its first expected passage, or zero when none is
returned. Their averages exclude cases that expect abstention. Ranking uses the
thresholded results, so MRR here is limited to the returned top three. The JSON
also records these metrics and verification for each case.

| Original query | TF-IDF ranking | BM25 ranking |
| --- | --- | --- |
| flow pressure operating setpoint | flow-context | flow-context, electrical-context, vibration-context |
| vibration accelerometer channels | vibration-context | vibration-context |
| current voltage operating regime | electrical-context | electrical-context, baseline-scope, thermal-context |
| motor temperature circulating fluid thermocouple | thermal-context | thermal-context |
| missing timestamps sampling gaps | sampling-gaps | sampling-gaps |
| anomaly score failure probability diagnosis | baseline-scope | baseline-scope, vibration-context, flow-context |
| anomaly changepoint source labels evaluation | annotation-scope | annotation-scope, sampling-gaps, electrical-context |
| renaissance sonnet planetary astronomy | Abstain | Abstain |
| sourdough pastry baking recipe | Abstain | Abstain |

Every answerable case has recall at 3 and reciprocal rank of 1. Both unrelated
queries abstain correctly. The fixture was already perfect under TF-IDF, and
BM25 does not improve those metrics. It returns eight additional lower-ranked
passages at its default threshold. All are authentic corpus excerpts, but
verification does not show that these extra matches are useful. The expected
IDs identify target passages and are not exhaustive relevance judgments. A larger,
separately authored relevance set, including difficult queries with incidental
word overlap, would be needed to judge a change in retrieval quality or choose
production thresholds. No such evaluation is claimed here.

## Behavioral checks

`tests/test_bm25.py` checks a hand-calculated score, positive IDF for common terms,
term-frequency saturation, document-length normalization, query repetition,
empty and out-of-vocabulary input, ties, invalid parameters, the existing query
budget, and rejection of forged evidence. It also compares unchanged TF-IDF
scores directly with the original calculation, checks the nontrivial reciprocal
rank of a synthetic second-place result, exercises the portable module CLI, and
runs BM25 through the real investigation graph with synthetic sensor inputs.
The service test replaces only model and recording disk loading.

Run the focused checks with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_bm25.py tests/test_investigation.py -q
```
