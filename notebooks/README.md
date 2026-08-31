# Exploratory analyses

[Flow variation in a pump recording](01_recording_exploration.ipynb) examines
SKAB experiment `valve1/1`. It compares flow distributions by source annotation
and examines temporal variation with a trailing 30-second rolling median.

The anomalous group has a lower median flow, but its observed range overlaps
with the normal group. These are descriptive findings from one development
recording; no trained detector or held-out evaluation is reported.

## Reproduction

Install the environment using the project's [quick start](../README.md#quick-start).
Download the pinned sample from the repository root if it is not already present:

```powershell
.\.venv\Scripts\python.exe -m forge.data.sample --download
```

Open the notebook with the repository's `.venv` interpreter and run all cells
in order. The loader verifies the source checksum before analysis. Plotly charts
are generated locally; the Markdown results table remains readable on GitHub.

## Publication and data scope

Raw recordings remain outside Git. Clear notebook outputs and execution counts
before committing, because tables and interactive charts embed source data.
Keep methods, aggregate findings, provenance, and limitations in Markdown.
SKAB retains its source license; see the [data notes](../data/README.md).

The inspected recording is reserved for development and excluded from the
future final test set.
