# First anomaly-detection benchmark

The frozen Isolation Forest reached **0.616 point F1** on the held-out recordings
and detected **18 of 23 annotated events**. It also produced **145.4 false alert
onsets per normal hour**. This is a working research baseline, not a detector
ready for unattended maintenance alerts.

## Selection and results

All candidates used the same 18,306 unique normal training observations and eight
instantaneous sensor inputs. The statistical reference used medians and scaled
MAD, with an IQR fallback for pressure. Isolation Forest used 256 trees, seed 42,
and either 256 or 1,024 samples per tree. Validation F1 selected the latter.

| Candidate | Validation F1 | Validation average precision | False onsets / normal hour |
|---|---:|---:|---:|
| Robust maximum deviation | 0.697 | 0.724 | 3.3 |
| Isolation Forest, 256 samples | 0.677 | 0.493 | 147.0 |
| Isolation Forest, 1,024 samples | 0.713 | 0.508 | 111.9 |

The learned detector's small F1 advantage came with a large alert burden.
The statistical baseline had stronger ranking quality on validation. Selecting
only for point F1 was therefore a poor proxy for operational usefulness. The
chosen operating point remains frozen so this limitation is visible rather than
being hidden by tuning on the test set.

| Held-out metric | Result |
|---|---:|
| Evaluated observations | 11,987 |
| Anomalous observations | 4,310 |
| Precision | 0.488 |
| Recall | 0.834 |
| Point F1 | 0.616 |
| Average precision, pooled | 0.451 |
| Average precision, mean over groups | 0.608 |
| Event recall | 18 / 23 (0.783) |
| Median delay among detected events | 2 seconds |
| False alert onsets | 310 |
| Normal observation exposure | 2.1325 hours |

The two-second delay excludes missed events. An event receives detection credit
only for a new alert onset inside it. A warning already active before an event
does not count as detection. No point adjustment is applied. The pooled anomaly
prevalence is approximately 0.360, providing context for average precision.

## Data accounting

Training started with 23,893 rows: 5,015 anomalous rows and 572 repeated normal
observations were excluded. Of the 18,306 retained rows, 9,405 came from the
source-designated normal recording, which has no row-level annotations.

Validation contained 9,994 raw rows, reduced to 9,973 unique observations after
removing 21 duplicates. No conflicting anomaly annotations remained.

Test contained 12,919 raw rows, with 697 repeated observations removed.
Among the 12,222 unique observations, 235 had conflicting anomaly annotations
across overlapping source files: 178 in group `other/12` and 57 in group `other/2`.
These were excluded from metrics and broke evaluation continuity, leaving 11,987
scored observations. They were not treated as normal or assigned a majority label.

## What this establishes

The code can reproduce training, selection, causal alerting, and evaluation on a
fixed split with recorded provenance. The resulting alerts can enter the local
investigation workflow, which retrieves review guidance and exports cited drafts.

The data comes from one laboratory testbed and a small number of acquisition
dates. Adjacent experiments are not independent machines. The normal reference
mixes operating regimes, and the long normal recording contributes about half
the training rows. A low-flow regime can be far from the pooled median while
still being normal for that regime. Reference deviations must not be interpreted
as a causal diagnosis or an Isolation Forest feature explanation.

Next modeling work should define an alert-burden objective before selection,
compare operating-regime-aware references and causal temporal features on
development data, and obtain a fresh holdout for any new performance claim.
PyTorch sequence models should be compared against these baselines rather than
assumed to improve them.

## Reproduction and audit

The [protocol](evaluation-protocol.md) and
[machine-readable results](evaluation-results.json) contain the fixed selection
rule, thresholds, per-group counts, hashes, and source revision. Raw data, pickled
models, and generated reports remain outside Git. Run `python -m forge train`
in the project environment to create a fresh local artifact; the saved public
summary describes the recorded benchmark run, not automatically every new run.

The first evaluation exposed a pandas timestamp-resolution error in the normal
exposure calculation. That arithmetic was corrected and covered by a regression
test. The original local metadata and result were retained. The model, threshold,
predictions, classification counts, average precision, event counts, and delays
did not change; no model was selected again after viewing the test results.
