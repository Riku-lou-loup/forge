# SKAB inventory and evaluation split

FORGE uses a fixed allocation of complete recordings before baseline training.
The inventory contains 35 CSV files, comprising 34 annotated recordings and one
separate normal-operation recording. This allocation was fixed before fitting;
the resulting benchmark is documented in [evaluation](evaluation.md).

## Source and scope

The [inventory](../data/skab-inventory.json) records every CSV under `data/` at
SKAB revision `b2c0d46c2971dcbfe71e26087b6d231998bb91c2`, including source URLs,
Git blob identities, SHA-256 checksums, schemas, chronology, and sampling intervals.
Source files were checked against that revision's Git tree before inventorying.
Raw files remain outside Git. The inventory includes source attribution and
license links for SKAB, developed by Iurii D. Katser and Vyacheslav O. Kozitsin.

The pinned [source catalog](https://github.com/waico/SKAB/blob/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/data/README.md)
describes inlet-valve, outlet-valve, and other testbed experiments. Actual filenames
are authoritative: `valve1/0` through `valve1/15` and `valve2/0` through `valve2/3`.
The catalog's illustrated numbering differs from the repository files.

The audit examined integrity, schema, timestamps, and exact repeated measurements.
All files have complete recorded values and unique, ordered timestamps within each
file, although sampling intervals vary. Test sensor distributions,
label summaries, plots, and detector outcomes were not used to choose the split.
Schema validation reads annotation values to verify their allowed domain.

## Allocation

The allocation is stored in [skab-splits.json](../data/skab-splits.json) as
`skab-experiment-v1`. It assigns complete recordings to partitions without randomly
splitting individual rows.

| Partition | Files | Recorded rows, before filtering or deduplication | Purpose |
|---|---:|---:|---|
| Training | 14 | 23,893 | Fit preprocessing and detectors using eligible normal observations |
| Validation | 9 | 9,994 | Select features, model settings, thresholds, and alert rules |
| Test | 12 | 12,919 | Evaluate the frozen system |

| Family | Training IDs | Validation IDs | Test IDs |
|---|---|---|---|
| `valve1` | 0 to 7 | 8 to 11 | 12 to 15 |
| `valve2` | 0 | 1 | 2 to 3 |
| `other` | 5, 6, 9, 14 | 1, 7, 10, 11 | 2, 3, 4, 8, 12, 13 |
| `anomaly-free` | `anomaly-free` | None | None |

Valve runs use consecutive chronological blocks within each family. The heterogeneous
`other` files have an explicit allocation that keeps overlap groups together.
FORGE defines this allocation independently of the official SKAB benchmark.
Fault mechanisms need not appear in every partition, and the allocation does not
require equal row counts. The previously inspected `valve1/1` stays in training.

## Overlap findings

Holding out files alone would leak identical observations across partitions.
Pairwise checks compared exact sensor tuples, timestamps, and combined
timestamp and sensor tuples. All five detected pairs repeat both time and measurements.

| Recordings | Shared timestamp and sensor rows | Assigned partition |
|---|---:|---|
| `anomaly-free/anomaly-free`, `other/5` | 572 | Training |
| `other/10`, `other/11` | 21 | Validation |
| `other/12`, `other/13` | 349 | Test |
| `other/2`, `other/3` | 120 | Test |
| `other/3`, `other/4` | 228 | Test |

The overlap between `other/2`, `other/3`, and `other/4` connects them into one
group. They must stay together even though the first and last files do not
directly overlap. The split validator checks both group membership and every
recorded overlap edge. The audit
found no exact repeated sensor vectors or shared timestamps across partitions.
These checks cannot establish that all temporal or operating-condition dependence
has been removed.

## Baseline rules

- Fit on normal-labeled observations from training files plus the source-designated
  normal-operation file. Keep the normal-operation file without annotation columns.
- Deduplicate training observations using timestamp and all eight sensor values
  before fitting. The normal-operation file and `other/5` otherwise double-count
  572 observations. Do not resample or fill gaps implicitly.
- Exclude `anomaly`, `changepoint`, recording identifiers, and absolute timestamps
  from detector features. Timestamp values may order observations and define windows.
- Fit transformations on training data only. Derive temporal features causally
  within recordings, resetting state at each file boundary.
- Select model settings, thresholds, and alert aggregation on validation data.
  Freeze these choices and event-matching rules before final test scoring.
- Account for overlapping observations when calculating validation and test metrics.
  Count identical observations once, report annotation conflicts, and aggregate by
  leakage group. Overlapping files cannot be treated as independent replicates.

The [modeling notebook](../notebooks/02_baseline_modeling.ipynb) implements normal
training-row selection, deduplication, and feature/provenance separation. Reusable
training and scoring now live in `src/forge/ml/`; the precise alert and metric
definitions are in the [evaluation protocol](evaluation-protocol.md).

## Reproduction

From the repository root, inspect the training allocation without reading raw files:

```powershell
.\.venv\Scripts\python.exe -m forge.data.partitions
```

Download and audit the development partitions against their recorded checksums:

```powershell
.\.venv\Scripts\python.exe -m forge.data.partitions --split train --download --audit
.\.venv\Scripts\python.exe -m forge.data.partitions --split validation --download --audit
```

For a deliberate integrity and overlap audit of the complete frozen inventory:

```powershell
.\.venv\Scripts\python.exe -m forge.data.partitions --split all --download --audit --allow-test
```

The all-file audit reads test values to verify metadata and overlap. It performs no
fitting, plotting, or performance measurement. Routine development should use the
first two partitions. The explicit flag prevents accidental test access through this
command. Local CSV files can still be opened manually.

## Limits and next step

All recordings come from one testbed. Training, validation, and test partitions
include recordings from the same acquisition dates, and nearby valve runs may
share operating conditions. Evaluation is therefore limited to held-out recordings
from this testbed. Generalization to unseen machines, independent acquisition days,
and factory deployment would require separate evidence. Published leaderboard
numbers from other protocols are not directly comparable.

Normal-training selection and deduplication are implemented in the notebook.
The statistical baseline and Isolation Forest have since been compared on
validation data, followed by test evaluation with the selected model and rules
frozen. The benchmark linked above records those results.
