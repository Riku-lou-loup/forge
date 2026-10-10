# Exploratory analyses

[Flow variation in a pump recording](01_recording_exploration.ipynb) examines
SKAB experiment `valve1/1`, comparing flow distributions by source annotation
and temporal variation with a trailing 30-second rolling median. The observed
ranges overlap, so these descriptive findings alone are not a detector.

[Baseline modeling preparation](02_baseline_modeling.ipynb) constructs the normal
training feature table from 14 assigned recordings, keeps provenance separate,
and examines the pressure channel's zero MAD. The notebook records the initial
modeling calculations. Reusable training lives in `src/forge/ml/`, with its
results in [the benchmark report](../docs/evaluation.md).

[Validation score separation](03_validation_diagnostics.ipynb) compares the frozen
detector's normal and anomalous score distributions on validation groups. It
reports raw ranking and persistent alert rates separately, and exports local
interactive and static figures without refitting or changing the threshold.
See [the findings](../docs/validation-diagnostics.md).

## Reproduction

Install the environment using the project's [quick start](../README.md#quick-start).
The training download includes the exploration sample:

```powershell
.\.venv\Scripts\python.exe -m forge.data.partitions --split train --download --audit
```

Open the notebooks with the repository's `.venv` interpreter and run cells in
order. The loader verifies source checksums. Training preparation reads no
validation or test recordings. Plotly charts are generated locally.

## Publication and data scope

Raw recordings stay outside Git. Clear notebook outputs and execution counts
before committing because tables and charts can embed source data. Keep methods,
aggregate findings, provenance, and limitations in Markdown. SKAB retains its
source license, as described in the [data notes](../data/README.md).

The explored recording belongs to development and is excluded from the held-out
test set. The notebook does not silently refit or replace the frozen artifact.
