# Generalization and tuning audit

This audit refits FORGE’s detectors while excluding different whole recording
groups. It tests whether the current gradient-boosting configuration remains
effective beyond its original training allocation and whether a fixed, small
hyperparameter search improves its precision.

The tuned procedure reaches **33.7% precision** and
**89.5% recall** on pooled outer-fold predictions.
The current configuration reaches 32.5% precision
and 91.0% recall under the same fold protocol.
Tuning changes precision by **+1.2 percentage points** and
recall by **-1.5 percentage points**. These are development
audit results, not a new-machine reliability claim.

In 2 of the five outer folds, no gradient-boosting candidate met
all inner recall requirements, so the declared fallback supplied the alert
settings. The per-fold results and unmet alert targets below are part of the
outcome, not exceptions removed from the comparison.

## What the comparison measures

The pool contains 23 previously inspected source recordings in 21 overlap
groups, drawn from the original training and validation partitions. No old
test CSV was read. Five outer folds assess every group once. Within each
outer-training pool, three inner folds select the threshold and persistence.
The tuned procedure also chooses among four declared gradient-boosting
configurations. Whole overlap groups stay together at both levels.

“Current configuration” means a new fit using the existing hyperparameters,
with its alert settings selected inside each outer fold. It is not the app’s
saved model scored on observations it already fitted. Isolation Forest is
also refitted in each fold and receives the same inner alert-selection rule.
Global robust scales are recomputed from each fit’s normal observations.
The relative models still initialize their unlabeled reference from the first
60 readings of each assessed source file, matching the application.

The original 95.1% precision used a different, smaller validation allocation
and thresholds chosen on that allocation. Its difference from the figures
below cannot by itself quantify overfitting. The paired outer-fold comparison
is the relevant measure of the declared tuning procedure.

## Pooled outer-fold results

| Procedure | Precision | Recall | False-positive readings | Missed anomalous readings | False alert onsets | Onsets / available normal hour | Events detected |
|---|---:|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 31.0% | 83.6% | 15,846 | 1,397 | 381 | 55.37 | 17 / 24 |
| Current gradient-boosting configuration | 32.5% | 91.0% | 16,079 | 771 | 137 | 21.01 | 14 / 24 |
| Tuned gradient-boosting procedure | 33.7% | 89.5% | 14,975 | 898 | 220 | 33.74 | 16 / 24 |

Relative to the current configuration, tuning changes false-positive readings
by -1,104, missed anomalous readings by
+127, and separate false alert episodes by
+83.
A small precision gain should be assessed alongside these changes rather
than taken as a sufficient reason to replace the active model.

| Target for the tuned procedure | Achieved? |
|---|---|
| Precision at least 95% | No |
| Available-normal FPR at most 1% | No |
| At most two false onsets per available normal hour | No |

Precision is the share of alerted readings annotated anomalous. Recall is
the share of anomalous readings that trigger an alert. Precision is reported
as zero when no alerts occur, and the JSON retains the alert count. A 100%
precision value means there were no false-positive readings among the alerts.
It does not mean that all anomalies were detected or that accuracy is 100%.

Initialization readings receive no assessment. Their normal exposure is
excluded from FPR and false alert frequency, while startup anomalies still
count as misses. The full confusion counts retain unscored normal readings
as normal readings without an alert. An event counts as detected only when
a new alert onset falls inside its annotated interval. No point adjustment
expands an alert across an entire anomalous interval.

| Procedure | Available-normal FPR | Macro group precision | Macro group recall | Initialization readings | Median detected-event delay |
|---|---:|---:|---:|---:|---:|
| Isolation Forest | 64.0% | 62.8% | 82.4% | 0 | 9 s |
| Current gradient-boosting configuration | 68.5% | 62.3% | 90.5% | 1,299 | 6.5 s |
| Tuned gradient-boosting procedure | 63.8% | 64.0% | 89.2% | 1,299 | 6.5 s |

Macro metrics give each recording group equal weight. Pooled metrics count
all readings together, so the long source-normal overlap group contributes
much more normal exposure. Delay medians cover detected events only and may
refer to different event sets for different procedures.

## Variation across folds

| Outer fold | Current precision | Tuned precision | Current recall | Tuned recall | Chosen tuned candidate | Inner recall requirements met? |
|---|---:|---:|---:|---:|---|---|
| 1 | 13.1% | 13.5% | 100.0% | 100.0% | larger_hgb | No; declared fallback |
| 2 | 70.1% | 76.4% | 89.8% | 84.7% | regularized_hgb | Yes |
| 3 | 39.5% | 43.2% | 99.9% | 97.3% | larger_hgb | No; declared fallback |
| 4 | 87.3% | 87.3% | 84.8% | 84.8% | current_hgb | Yes |
| 5 | 95.0% | 95.0% | 75.4% | 75.4% | current_hgb | Yes |

Inner selection requires 60% point recall, 60% event recall and 20% point
recall in every inner group. If no setting qualifies, the predeclared
fallback maximizes the weakest normalized recall requirement before applying
the precision tie-breaker. Such folds remain in the comparison and are
marked above. Inner feasibility does not guarantee outer feasibility.
These requirements can select a low threshold that retains anomaly readings
but also alerts on many normal readings. The audit evaluates that complete
selection procedure, including its fallback, rather than tree capacity alone.

| Procedure | Mean training precision | Mean outer precision | Mean training recall | Mean outer recall | Outer precision range |
|---|---:|---:|---:|---:|---:|
| Isolation Forest | 55.7% | 44.5% | 78.9% | 83.9% | 12.5%–60.0% |
| Current gradient-boosting configuration | 62.7% | 61.0% | 96.9% | 90.0% | 13.1%–95.0% |
| Tuned gradient-boosting procedure | 68.0% | 63.1% | 96.6% | 88.4% | 13.5%–95.0% |

Each training score uses the same fitted model and alert settings as its
outer assessment. Training readings were used for fitting, so these are
resubstitution scores, not another validation result. Large gaps warrant
concern about fit and operating-condition sensitivity, but they do not
separate memorization from distribution differences by themselves. Means
and ranges above describe five folds, not confidence intervals.

The largest false-positive count belongs to `anomaly-free/anomaly-free` with
9,102 readings, or 60.8% of
the tuned procedure's false-positive readings. This group contains
`anomaly-free/anomaly-free`, `other/5`. This concentration helps locate
the weakness, but it does not establish which sensor or mechanism caused it.

## Final tuning artifact

After the nested assessment, five-fold grouped selection across the full
development pool selected `larger_hgb`. It was refitted on
31,995 available observations, including
8,522 anomalous training targets.
The selected threshold is `0.14579881621623614` with
3 consecutive above-threshold readings.
Its full-pool inner selection status is `recall_feasible`.

This is a research artifact. Its full-pool selection scores are not
independent performance, and it has **not been activated**. The app’s prior
model, original baseline and historical result files retain their recorded
hashes. The JSON records the final candidate settings and model checksum.

## Limits and next evidence

The data and earlier results influenced the model family and feature design
before this audit. Nested grouping protects the current fitting and tuning
steps from their outer assessment groups, but it cannot remove that prior
development history. Overlap groups are also not independent factories or
assets. Source-file order is preserved, but groups are not evaluated in a
strictly forward-time deployment simulation.

A reference built from an already faulty startup may miss that fault.
SKAB annotations identify anomalous readings rather than confirmed physical
failures. Further claims need fresh recordings, with the model and alert
settings frozen before scoring. The next model change should follow the
group-level failures shown here, rather than repeated tuning against these
outer scores until they look favorable.

## Reproduction and provenance

From the repository, using its environment:

```powershell
.\.venv\Scripts\python.exe -m forge.ml.generalization --plan
.\.venv\Scripts\python.exe -m forge.ml.generalization
.\.venv\Scripts\python.exe scripts/report_generalization.py <run_id>
```

Reported run: `generalization-20261007T095051Z-01a19f79`. The [JSON summary](generalization-results.json)
contains the exact group allocation, configuration, per-group outcomes,
training audits and source hashes. Ignored local run folders retain inner
selection ledgers, outer prediction arrays and fitted models. The publisher
builds a source-only [notebook 05](../notebooks/05_generalization_audit.ipynb).
Run its cells to display the saved results inline. This notebook does not
retrain a model or read test recordings.

See the [protocol](generalization-protocol.md) for the fixed search and
selection rules. Software tests include changing only excluded-group labels
and checking that the corresponding model and selected settings do not move.
Those checks establish specific implementation properties, not the absence
of statistical overfitting.
