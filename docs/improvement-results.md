# Precision-focused development results

Relative gradient boosting reduces false-positive readings from 2,193 to 118 on
development validation, a 94.6% reduction. Precision rises from 59.0% to 95.1%,
while point recall falls from 90.0% to 65.8%. This implements the chosen preference
for fewer alerts even when more anomalies are missed.

These recordings were used for model and threshold selection. The results are
development evidence, not a fresh held-out benchmark or a factory reliability
claim. The previously inspected test partition was neither rescored nor used for
selection. Source anomaly annotations do not establish physical failures.

## Same validation observations, different operating points

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

Both full confusion matrices cover 9,973 unique observations: 3,507 anomalous and
6,466 normal. The new model suppresses startup alerts; 519 normal observations are
counted in its full no-alert cell but excluded from available normal exposure.
No startup anomaly occurs in this validation set. Startup anomalies would count
as misses, not disappear from recall. Sampling is at least one second apart;
each available normal reading contributes one second of capped exposure. Gaps
contribute no invented normal operation.

The 95% precision target is met. The 1% available-normal FPR and two false onsets
per available normal hour targets are **not met**. Detecting every event means at
least one new alert onset falls inside each event; it does not mean all anomalous
readings are detected. The other/10 overlap group has only 21.7% point recall.
Delay medians cover detected events, whose membership differs between models.
The new operating point is more selective and typically slower.

## What changed

The eight sensor channels remain inputs. Each source recording supplies 32
features: deviation from the initial 60-reading median, one-step change,
deviation from the preceding 20-reading median, and trailing 20-reading standard
deviation. Features use only current or earlier measurements. Normal training
data supplies global robust scales. Gaps over two seconds reset rolling history
and alert persistence; the fixed initial reference is retained.

HistGradientBoostingClassifier uses 200 iterations, seven maximum leaves,
learning rate 0.05, minimum leaf size 30, L2 regularization 10, and seed 42.
Internal early stopping is disabled to avoid random row validation. It fits
22,541 unique available training observations, including 5,015 anomalous targets.
Training leakage groups receive equal total weight. Annotations are training
targets, never input features or investigation evidence. This is supervised
learning; the original Isolation Forest fits normal examples only.

The selected threshold is **0.19899640796797832**, with five consecutive readings
above threshold required for an alert. Scores are not calibrated physical failure
probabilities. The starting reference must represent an appropriate operating
condition; a fault already present throughout initialization may be missed.
Startup readings have no assessment, which the app and exported reports disclose.

## Comparison and correction

The [fixed protocol](improvement-protocol.md) requires point recall of at least
60%, event recall of at least 60%, and point recall of at least 20% in every
validation group. Feasible candidates are ranked by precision, then fewer false
onsets per normal hour, then recall. Three raw/causal/relative Extra Trees models,
the frozen Isolation Forest, robust deviation, and relative Isolation Forest form
the first stage. A declared follow-up compares two additional Extra Trees
representations and four gradient-boosting variants. The follow-up is informed by
initial development outcomes; it is not an independent statistical experiment.

The first implementation initialized overlapping recordings as one merged stream.
That differed from the application's per-file reference and produced an overly
optimistic provisional result of 98.7% precision. Both stages were rerun after
fixing this boundary. The published **95.1%** result initializes every source file
separately before deduplicating observations. The first inventory occurrence
supplies prediction context for shared observations. Conflicting annotations
remain unknown. A regression test protects this behavior.

The corrected follow-up was rerun to record the library version and reproduced
the same threshold, policy and confusion counts. The aggregate
[JSON report](improvement-results.json) retains all twelve candidates, fit counts,
settings, per-group metrics and provenance. Ranking metrics in its legacy
full-metrics block include zero startup scores; they do not measure ranking on
available readings alone.

## Reproduce and inspect

After the README setup and baseline training, use the project interpreter:

```powershell
.\.venv\Scripts\python.exe -m forge.ml.development
.\.venv\Scripts\python.exe -m forge.ml.development --followup
.\.venv\Scripts\python.exe -m forge.ml.activation <run_id printed by the selected comparison>
```

Comparison commands write unique local folders without activating candidates.
Activation checks configuration, split, inventory, model hashes, library version,
recall floors and improvement over baseline precision. It writes `models/active.json`;
the original `models/latest.json` and test benchmark remain intact. Investigation
uses the active model, falling back to the baseline only when no active pointer
exists. A broken active artifact fails visibly rather than silently changing model.
`forge evaluate --allow-test` remains the original baseline evaluation path.

Read [notebook 04](../notebooks/04_precision_comparison.ipynb) for inline confusion
matrices, the candidate comparison and per-group counts. It reads saved aggregates
and does not train or open test recordings. Checks cover feature causality,
recording context, gaps, startup misses, artifact round trips, workflow abstention
and Streamlit review controls.

The next evidence needed is a fresh evaluation collection, preferably another
operating period or asset, with the detector and thresholds frozen before scoring.
Further tuning on these validation recordings can improve development behavior
but cannot establish that generalization claim.
