# Precision-focused development results

FORGE's original Isolation Forest produced too many false alerts on the SKAB
validation recordings. The next comparison prioritized precision, accepting more
missed anomalies in exchange for fewer alerts. The selected detector is now a
supervised gradient-boosting classifier. Isolation Forest remains the baseline.

On development validation, false-positive readings fall from 2,193 to 118, a
94.6% reduction. Precision rises from 59.0% to 95.1%, while point recall falls
from 90.0% to 65.8%. Precision measures how many alerted readings are annotated
anomalous. Recall measures how many annotated anomaly readings trigger an alert.

These recordings were used to select the model and threshold. The results
describe development performance and do not establish reliability on new
equipment. The previously inspected test partition was neither rescored nor
used for this selection. SKAB anomaly annotations do not establish physical
equipment failures.

## Why gradient boosting replaced the active detector

The earlier analysis found differences between some normal validation readings
and the normal operating conditions represented in training. The comparison
therefore tested features that describe changes from each recording's starting
condition, alongside raw measurements and rolling features. The relative
gradient-boosting model achieved the highest precision among the candidates
meeting the minimum recall requirements.

This changes the learning approach as well as the algorithm. Isolation Forest
was fitted on 18,306 unique normal observations without learning a classifier
from normal and anomalous targets. The new classifier uses both classes in
training. Its 22,541 available training observations include 5,015 anomalous
targets. Source annotations provide those targets but never enter the model's
input features or the investigation evidence.

The feature representation and alert persistence also changed. These results
therefore compare complete detector configurations. They do not isolate the
contribution of gradient boosting or establish that it is generally superior
to Isolation Forest. The original model and its published benchmark remain
available for comparison.

## Results on the same validation observations

The table compares each model with its selected alert settings. A reading-level
error and an alert episode measure different things: several consecutive
false-positive readings can belong to a single false alert episode. An onset
marks the start of an alert episode under the persistence and gap rules.

| Measure | Original Isolation Forest | Relative gradient boosting |
|---|---:|---:|
| True positive readings | 3,156 | 2,309 |
| False positive readings | 2,193 | 118 |
| Missed anomalous readings | 351 | 1,198 |
| Normal readings without alert | 4,273 | 6,348 |
| Precision | 59.0% | 95.1% |
| Point recall | 90.0% | 65.8% |
| Point F1 | 0.713 | 0.778 |
| New alert onsets in annotated events | 8 / 10 | 10 / 10 |
| False alert onsets | 201 | 26 |
| Available normal readings | 6,466 | 5,947 |
| FPR over available normal readings | 33.92% | 1.98% |
| False onsets / available normal hour | 111.91 | 15.74 |
| Median detected-event delay | 5 seconds | 41.5 seconds |
| Unavailable initialization readings | 0 | 519 |

Both full confusion matrices cover 9,973 unique observations, including 3,507
anomalous and 6,466 normal readings. The new model makes no assessment during
initialization. Its 519 normal initialization readings appear in the full matrix
as normal readings without an alert, but they are excluded from available normal
exposure. No startup anomaly occurs in this validation set. If one did, it would
count as missed in recall.

The false-positive rate (FPR) and false alert frequency use the normal readings
for which an assessment is available. Sampling is at least one second apart, so
each such reading contributes one second of capped exposure. Sampling gaps do
not count as observed normal operation.

The 95% precision target is met. The 1% available-normal FPR and two false onsets
per available normal hour targets are **not met**. Detecting all ten events means
that at least one new alert onset falls inside each annotated event. It does not
mean every anomalous reading was detected. The other/10 overlap group has only
21.7% point recall. Median alert delay is also higher, although the two delay
medians cover different sets of detected events.

## Features, fitting and the alert rule

The eight sensor channels supply 32 features. For each sensor, the model receives
the deviation from the initial 60-reading median, the one-step change, the
deviation from the preceding 20-reading median and the trailing 20-reading
standard deviation. These features use only current or earlier measurements.
Normal training data supplies global robust scales. Gaps over two seconds reset
rolling history and alert persistence, while the initial reference is retained.

The implementation uses scikit-learn's `HistGradientBoostingClassifier` with
200 iterations, seven maximum leaves, a learning rate of 0.05, a minimum leaf
size of 30, L2 regularization of 10 and seed 42. Internal early stopping is
disabled to avoid random row validation. Recordings that share observations are
kept in the same leakage group, and each training group receives equal total
weight after duplicate observations are removed.

The selected threshold is **0.19899640796797832**. An alert requires five
consecutive readings above that threshold, compared with three for the original
baseline. Scores are not calibrated physical failure probabilities. The first
60 readings establish an unlabeled operating reference and receive no assessment.
A fault already present throughout initialization may become part of that
reference and be missed. The app and exported reports disclose this requirement.

## Selection and the recording-context correction

The [fixed protocol](improvement-protocol.md) requires at least 60% point recall,
60% event recall and 20% point recall in every validation group containing
anomalies. These requirements prevent a nearly silent detector from winning on
precision alone. Qualifying candidates are ranked by precision, then fewer false
onsets per normal hour, then recall.

The first stage compares Extra Trees with raw, causal and relative features,
alongside the frozen Isolation Forest, robust deviation and relative Isolation
Forest. A follow-up compares two additional Extra Trees representations and four
gradient-boosting variants. Its design was informed by the first development
results, so the twelve-candidate comparison is an iterative development exercise.

During verification, the first implementation was found to initialize overlapping
recordings as one merged stream. The app initializes each source file separately.
That mismatch produced an overly optimistic provisional result of 98.7%
precision. Both comparison stages were rerun after correcting the boundary.
The published **95.1%** result uses separate source-file initialization before
deduplication. For a shared observation, the first inventory occurrence supplies
the prediction context. Conflicting annotations remain unknown. A regression
test checks this behavior.

The corrected follow-up was rerun to record the library version and reproduced
the same threshold, policy and confusion counts. The [JSON report](improvement-results.json)
retains all twelve candidates, fit counts, settings, per-group metrics and
provenance. Ranking metrics in its legacy full-metrics block include zero startup
scores, so they do not measure ranking on available readings alone.

## Reproduce and inspect

After the README setup and baseline training, use the project interpreter:

```powershell
.\.venv\Scripts\python.exe -m forge.ml.development
.\.venv\Scripts\python.exe -m forge.ml.development --followup
.\.venv\Scripts\python.exe -m forge.ml.activation <run_id printed by the selected comparison>
```

The comparison commands write unique local folders without activating a
candidate. Activation checks configuration, split, inventory and model hashes,
the library version, the recall requirements and improvement over baseline
precision. It writes `models/active.json`. The original `models/latest.json`
and test benchmark remain intact.

Investigations use the active model. They fall back to the baseline only when
there is no active pointer. An invalid active artifact produces an error rather
than silently changing the model. `forge evaluate --allow-test` remains the
original baseline evaluation path.

[Notebook 04](../notebooks/04_precision_comparison.ipynb) displays inline confusion
matrices, the candidate comparison and per-group counts from saved aggregates.
Running it does not retrain a model or open test recordings. Software checks
cover feature causality, recording context, gaps, startup misses, artifact round
trips, workflow abstention and Streamlit review controls.

## What would establish stronger evidence

Reproducing these results checks that the implementation behaves consistently.
It does not rule out overfitting. Both the candidate and its threshold were
selected on these validation recordings, and the follow-up was informed by
earlier results on the same data. The reported precision may therefore be
optimistic for unseen recordings.

A stronger generalization assessment needs separately collected recordings,
preferably from another operating period or asset. Freeze the detector and alert
settings before scoring them, then examine precision, recall, false alert
frequency and delay together. Further tuning on the current validation set can
improve development behavior but cannot supply that independent evidence.
