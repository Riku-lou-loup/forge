# Why the frozen detector raises false alerts

The validation distributions show two distinct problems: normal scores shift
between groups, and some groups have overlapping or reversed normal/anomaly
score ordering. Changing one shared threshold cannot resolve both problems.

This analysis uses the frozen `isolation_forest_1024` artifact from run
`20261007T003523Z-e78e7b17`, threshold **0.504721**, and the original three-reading
persistence rule. It reads only the nine validation recordings, deduplicated
into eight groups and 9,973 observations. No unknown anomaly annotations occur
in this partition. No model fitting, threshold selection, or test scoring is
performed.

## Group results

| Validation group | Normal median score | Anomaly median score | ROC AUC | Persistent FPR | Persistent recall |
|---|---:|---:|---:|---:|---:|
| `other/1` | 0.6447 | 0.6301 | 0.175 | 99.6% | 100.0% |
| `other/10` | 0.5672 | 0.6099 | 0.786 | 65.6% | 95.3% |
| `other/7` | 0.4603 | 0.5942 | 0.996 | 0.5% | 99.1% |
| `valve1/10` | 0.4947 | 0.5667 | 0.882 | 10.1% | 86.0% |
| `valve1/11` | 0.4997 | 0.5421 | 0.719 | 21.6% | 73.9% |
| `valve1/8` | 0.4703 | 0.5424 | 0.925 | 1.2% | 95.0% |
| `valve1/9` | 0.4903 | 0.5358 | 0.794 | 6.2% | 79.4% |
| `valve2/1` | 0.5259 | 0.5632 | 0.782 | 53.0% | 89.2% |

`other/10` combines overlapping recordings 10 and 11. The rates count individual
observations after persistence, not alert onsets. ROC AUC describes raw score
ranking: 1 is perfect ordering, 0.5 is chance-level ordering, and below 0.5 is
predominantly reversed ordering. These descriptive estimates do not include
uncertainty intervals for temporally dependent data.

## What the distributions show

**`other/1` has a ranking failure as well as a threshold problem.** Every normal
score exceeds the threshold; persistence leaves an FPR of 99.6%. Normal scores
have a higher median than anomaly scores, and AUC is only 0.175. Raising a
one-sided threshold would still favor the wrong ordering through much of this
group. Lower alert volume alone would not establish better discrimination.

**`other/10` and `valve2/1` show better ordering but a poor shared operating point.**
Their normal score medians sit above the threshold, and roughly 79.2% of normal
scores exceed it before persistence. Anomalous medians are higher, but the
distributions overlap. A higher threshold could reduce false positives at a
recall cost. This analysis does not select that tradeoff or install per-group
thresholds. A recording-specific threshold fitted to its labels would not be a
demonstration of generalization to a new recording.

**`other/7` separates well.** AUC is 0.996, with 0.5% persistent FPR and 99.1%
recall. Its raw normal exceedance rate is 13.5%, so persistence suppresses many
isolated exceedances here. In `other/1`, almost all normal scores remain high,
so waiting for three observations offers little help.

The score distribution of normal-labeled readings differs substantially across
groups. Different operating conditions or incomplete coverage of normal behavior
are plausible causes, but scores alone cannot identify the physical cause.
The analysis does not establish sensor drift or incorrect source labels.

## Reproduce and inspect

Open [the diagnostics notebook](../notebooks/03_validation_diagnostics.ipynb), or
run this command in the project's installed environment:

```powershell
.\.venv\Scripts\python.exe -m forge.ml.diagnostics
```

The command uses portable paths derived from the package location. It exports
`score-distributions.html`, `score-distributions.png`, `score-summary.csv`, and
`summary.json` under the ignored `reports/figures/validation-<run>/` directory.
The HTML embeds Plotly for offline use. Histograms use identical bin edges across
groups, with each class normalized separately to 100%. Their bars describe raw
scores; the FPR in each panel title includes persistence. The static figure uses
the same aggregated bins as the interactive plot.

The model SHA-256 and evaluation provenance accompany the generated summary.
The notebook keeps outputs clear for publication and preserves the separate
modeling notebook. If a different model is trained later, regenerated plots
describe that new run; the table above remains the recorded baseline analysis.

## Next development question

Compare the sensor distributions of normal training observations with normal
validation observations, starting with `other/1` and `other/7`. This can locate
which channels distinguish well-covered from poorly covered operating regimes.
Use those findings to design a development experiment before changing features,
normal-reference construction, or alert objectives. The previously viewed test
set cannot support an independent new claim after this development cycle; a fresh
holdout is needed for that claim.
