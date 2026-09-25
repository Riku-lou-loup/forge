# PyTorch temporal CNN demo

The first frozen run detected 31.4% of annotated anomalous readings at 92.7%
precision on reused development validation data. It missed 2,407 of 3,507
anomalous readings, and two validation groups had zero recall. This increment
provides a working CPU training, checkpoint and investigation path. It does not
establish an improvement over the active model or the grouped generalization
audit, which used a different evaluation procedure.

## Run the demo

Use the project virtual environment with the CPU PyTorch dependency installed.
The recorded run used PyTorch 2.14.1+cpu. Keep the pinned training and validation
CSVs at the paths in `data/skab-inventory.json`. From the repository root:

```powershell
.\.venv\Scripts\python.exe -m forge.ml.torch_training --root .
```

The command reads `configs/pytorch-v1.json`, trains once, prints validation
metrics and epoch losses, and creates an ignored `models/pytorch-<UTC>-<id>/`
folder. It never updates `models/active.json` or `models/latest.json`. The
repository-relative default also works when the command runs from another
working directory. `--root` explicitly chooses the data, config and output root.

Reload the returned artifact path and investigate an original training or
validation recording:

```powershell
.\.venv\Scripts\python.exe -m forge.ml.torch_training --artifact models/pytorch-<UTC>-<id>
.\.venv\Scripts\python.exe -m forge.ml.torch_training --artifact models/pytorch-<UTC>-<id> --recording valve1/1
```

The second command runs the existing local investigation graph with its TF-IDF
retriever and prints a JSON report. No network model call is needed. Original
test recordings are rejected by this module's investigation command. The
Python adapter can also be passed to `forge.agents.workflow.investigate`:

```python
from forge.ml.torch_artifacts import load_torch_artifact

detector, metadata = load_torch_artifact(artifact_folder, root=repository_root)
# detector.score(frame), detector.readiness(frame), detector.deviations(frame)
```

Each frame must contain one source recording in timestamp order. The adapter
accepts the same eight-sensor schema as the existing workflow. Sensor deviations
are differences from the pooled training reference, not CNN feature attribution.

## What is fitted

The frozen configuration uses seed 42, 32-reading windows, two convolution layers
with 16 channels and kernel size 5, ReLU activations, temporal mean pooling and
one output logit. Adam runs for eight epochs with batches of 256, learning rate
0.001 and weight decay 0.0001 on two CPU threads. Binary cross entropy is
unweighted. There is no architecture search, epoch selection or hyperparameter
sweep.

Scaling uses the existing robust median/MAD reference, with its IQR fallback,
fitted only on 18,306 deduplicated normal rows from the original training split.
The normal catalog recording receives an inferred normal training target through
the existing data helper. The supervised fit uses 22,825 available window
endpoints, including 4,953 anomalous targets and 17,872 normal targets.

Windows are built from each complete source recording before endpoint
deduplication. They include the current reading and the previous 31 readings,
never a later reading. A gap over two seconds restarts the window. The first 31
rows of every recording or new segment are unavailable. Pooling only normal rows
would erase anomalous intervals and manufacture temporal adjacency, so that
pooled normal table is used only for scaling.

For overlapping source recordings, the first source in inventory order supplies
the window context for a shared endpoint, including its availability status.
Deduplication occurs within the existing leakage groups. Conflicting endpoint
labels are excluded from fitting and classification metrics. Only the eight
sensor measurements enter the network. Anomaly is the target. Changepoint,
recording identifiers and absolute timestamps are not model inputs. Timestamps
only enforce ordering and continuity.

## Development threshold and observed result

Weights and scaling are fitted before validation is opened. Validation chooses
the threshold from 201 score quantiles and one lower boundary, with a fixed
three-reading persistence rule. The selection maximizes precision subject to at
least 30% recall across all labeled validation readings, including unavailable
anomalies as misses. Equal precision is resolved by recall, then the higher
threshold.

A candidate must detect at least one positive and leave at least one available
labeled endpoint unalerted. If no candidate meets the recall floor, the declared
fallback maximizes recall and then precision among these nondegenerate
candidates. If none exist, training reports an error rather than saving a model
with no usable operating point. The first run met the floor without fallback at
threshold `0.5169919431209564`.

The first run was `pytorch-20261007T103456Z-2b35ccb7`. Its validation confusion
matrix includes unavailable readings as unalerted:

| Actual annotation | Alert | No alert |
| --- | ---: | ---: |
| Anomaly | 1,100 | 2,407 |
| Normal | 86 | 6,380 |

Precision was **92.7%** and recall was **31.4%**. False positive rate was **1.39%**
among 6,208 available normal readings, or **1.33%** among all 6,466 normal readings.
There were seven false alert onsets. Seven of ten annotated events had an alert
onset under the existing event metric. These event counts do not change the
large number of missed anomalous readings.

Of 9,973 unique validation readings, 9,684 were available and 289 were unavailable
at startup or after gaps. The unavailable rows included 31 anomalies. The
`other/1` and `valve2/1` groups both had zero point recall. A high pooled precision
therefore does not imply useful coverage of every recording.

| Epoch | Mean training loss |
| --- | ---: |
| 1 | 0.543285 |
| 2 | 0.336972 |
| 3 | 0.312047 |
| 4 | 0.304860 |
| 5 | 0.296289 |
| 6 | 0.291104 |
| 7 | 0.285157 |
| 8 | 0.281439 |

This validation partition has already informed earlier development and selects
this threshold. Its metrics are not independent held-out performance. No original
test CSV was opened for this increment. The supervised output predicts SKAB
anomaly annotations. Neither those annotations nor the sigmoid score establish
physical failure or a calibrated failure probability.

## Artifacts and checks

A run saves the tensor `state_dict` in `weights.pt`, a frozen `config.json`,
`metadata.json`, `history.json`, a threshold scan and checksums for weights,
config, metadata and history. Metadata records the split and inventory hashes,
source recording hashes, source configuration hash, code revision and file hashes,
Python and package versions, scaling audit, fitting counts and validation metrics.
Code hashes identify the actual source even when training precedes a commit.

The loader explicitly uses `torch.load(..., weights_only=True)` and checks tensor
keys, shapes, dtypes and finite values against the declared architecture. It
also checks configuration consistency, scaling values, the PyTorch version and
pinned manifest hashes. PyTorch documents that `weights_only=True` restricts
unpickling but does not remove every risk from untrusted files. Use local
artifacts from this training command. Checksums detect accidental corruption,
not authenticity when a whole artifact can be replaced. See the
[official PyTorch serialization documentation](https://docs.pytorch.org/docs/2.14/notes/serialization.html#torch-load-with-weights-only-true).

Training verifies identical scores before and after reloading on every validation
source. Behavioral tests also cover future-data isolation, gap and source resets,
annotation independence, deduplicated source context, conflicting labels,
train-only scaling, deterministic tiny fitting, independence of fitted weights
from validation changes, corrupted checkpoints and tensor shapes, pointer
preservation, and the existing investigation interface.
