# Causal alert-policy refinement

A fixed 15-reading median of the gradient-boosting scores reduced separate false
alert onsets from 137 to 41 in the grouped development comparison. Precision
changed from 32.53% to 32.79%, while recall fell from 90.95% to 90.66%.
This is an alert-policy improvement in episode fragmentation, with a substantial
remaining false-positive burden. It is not a new neural model or LLM fine-tuning.
The experimental policy has not replaced the active detector.

## Why this experiment

The [grouped audit](generalization-results.md) found that the current relative
gradient-boosting model produced 16,079 false-positive readings and 137 separate
false alert onsets. Hyperparameter tuning raised onsets to 220. The optional
[PyTorch model](pytorch-demo.md) reported 92.7% precision at only 31.4% recall on a
different, reused validation allocation. Its [missed-anomaly diagnosis](pytorch-missed-anomalies.md)
found large score-level differences between recordings. Those neural results do
not establish superiority over the grouped baseline and were not reused as a
matched comparator here.

The bounded hypothesis was that a causal median could suppress short score
fluctuations and reduce fragmented alerts. The window was fixed at 15 ready
readings before evaluation. There was no search over smoothing windows and no
revision after seeing outer-fold outcomes. Smoothing cannot correct a model that
assigns persistently high scores to normal operation. That limitation is visible
in the results below.

## Matched development protocol

The [frozen configuration](../configs/refinement-v1.json) uses the same 23 source
recordings, 21 overlap groups, five outer folds and three inner folds as the
previous audit, with seed 42. Only original training and validation recordings
are opened through the existing hash-checked loader. Original test CSVs are not
read. These development recordings have already informed the project, so this is
a retrospective comparison, not an untouched evaluation or evidence from new
machines.

Both procedures share each fitted HGB model. Training-only normal observations
supply robust scales, and supervised fitting uses only that fold's training
groups. Threshold and three- or five-reading persistence are selected separately
for each procedure from inner out-of-fold predictions. The original recall floors
remain 60% point recall, 60% event recall and 20% in every positive group. Both
procedures use the declared fallback in outer folds 1 and 3 because no inner
operating point satisfies every floor. Outer labels never enter fitting or
threshold selection. The unchanged baseline reproduces every pooled and per-group
metric in the earlier grouped audit exactly.

Scores are smoothed separately within each source file before overlap
deduplication. A median uses only current and past scores, requires 15 consecutive
available readings and resets after gaps over two seconds, unavailable readings
or nonfinite scores. The original 60-reading reference requirement remains, so a
clean uninterrupted recording first supplies a smoothed score at its 75th
reading. Initialization and reset periods remain explicitly unavailable, and
unavailable anomalous readings count as misses. Source sensor values are never
filled with zero. Zero placeholders for unavailable scores are masked by their
readiness flags.

## Results and tradeoffs

| Measure | Current HGB | HGB with causal median |
|---|---:|---:|
| Precision | 32.53% | 32.79% |
| Recall | 90.95% | 90.66% |
| F1 | 0.4792 | 0.4816 |
| True-positive readings | 7751 | 7726 |
| False-positive readings | 16079 | 15836 |
| Missed anomalous readings | 771 | 796 |
| True-negative readings | 8693 | 8936 |
| False alert onsets | 137 | 41 |
| Unavailable readings | 1299 | 1663 |
| Unavailable anomalous readings | 0 | 42 |
| Available normal readings | 23473 | 23151 |
| Available-normal FPR | 68.50% | 68.40% |
| False onsets / available normal hour | 21.01 | 6.38 |
| Detected events | 14 / 24 | 14 / 24 |
| Median detected-event delay | 6.5 s | 20.5 s |

False onsets fall by 70.1%, but false-positive readings fall by only 1.5%. The
available-normal false-positive rate is nearly unchanged, at about 68.4%. There
are also 322 fewer available normal readings. Fewer false-positive readings alone
therefore cannot be interpreted as better discrimination. The policy produces
fewer separate alert episodes while retaining long false alerts.

The extra initialization loses assessment of 42 anomalous readings. Total missed
anomalous readings increase by 25 after inner threshold selection, while the
median detected-event delay increases by 14 seconds. Delay medians describe only
detected events and need not represent the same event set. The existing event
metric counts a new alert onset within an annotated interval, without point
adjustment. Both procedures detect 14 of 24 events. The 95% precision, 1% FPR and
two-false-onsets-per-hour targets remain unmet.

The worst pooled burden remains the `anomaly-free/anomaly-free` overlap group:
its false-positive readings increase from 9,401 to 9,410, despite fewer separate
onsets. The group name is an inventory identifier, not a statement that every
merged reading is normal. `valve2/0` also becomes worse, with precision falling
from 89.22% to 81.25%, recall from 46.19% to 42.89%, and false-positive readings
rising from 22 to 39. The paired table includes all groups rather than selecting
only favorable cases.

| Group | Precision, before → after | Recall, before → after | False readings, before → after | False onsets, before → after |
|---|---:|---:|---:|---:|
| anomaly-free/anomaly-free | 4.18% → 4.18% | 100.00% → 100.00% | 9401 → 9410 | 18 → 6 |
| valve1/1 | 37.12% → 37.68% | 100.00% → 100.00% | 681 → 665 | 1 → 1 |
| valve1/6 | 37.09% → 37.64% | 100.00% → 100.00% | 687 → 671 | 1 → 1 |
| other/1 | 27.53% → 28.19% | 100.00% → 100.00% | 495 → 479 | 1 → 1 |
| valve1/8 | 36.97% → 37.52% | 100.00% → 100.00% | 682 → 666 | 1 → 1 |
| other/14 | 84.46% → 82.16% | 95.36% → 93.05% | 53 → 61 | 7 → 1 |
| valve1/2 | 59.91% → 65.45% | 79.82% → 85.46% | 180 → 152 | 20 → 6 |
| other/7 | 74.62% → 72.73% | 99.14% → 96.83% | 117 → 126 | 3 → 1 |
| valve1/9 | 65.46% → 69.35% | 85.82% → 85.57% | 182 → 152 | 17 → 4 |
| valve1/4 | 33.66% → 33.13% | 99.43% → 94.84% | 684 → 668 | 1 → 1 |
| valve1/7 | 39.32% → 40.58% | 100.00% → 100.00% | 625 → 593 | 2 → 2 |
| other/10 | 43.33% → 43.92% | 100.00% → 100.00% | 1356 → 1324 | 2 → 2 |
| valve1/10 | 36.99% → 37.55% | 100.00% → 100.00% | 683 → 667 | 1 → 1 |
| other/6 | 100.00% → 98.74% | 98.76% → 97.51% | 0 → 5 | 0 → 0 |
| valve1/0 | 79.81% → 87.12% | 63.09% → 70.82% | 64 → 42 | 10 → 2 |
| valve1/3 | 88.73% → 96.86% | 91.58% → 91.58% | 47 → 12 | 16 → 2 |
| valve2/1 | 78.36% → 80.48% | 85.89% → 80.48% | 79 → 65 | 16 → 2 |
| other/9 | 90.62% → 90.89% | 98.75% → 97.01% | 41 → 39 | 13 → 5 |
| valve1/5 | 100.00% → 100.00% | 81.39% → 81.39% | 0 → 0 | 0 → 0 |
| valve2/0 | 89.22% → 81.25% | 46.19% → 42.89% | 22 → 39 | 7 → 2 |
| valve1/11 | 100.00% → 100.00% | 74.69% → 74.69% | 0 → 0 | 0 → 0 |

## Reproduction and verification

Run the isolated experiment with the existing project environment and pinned SKAB
files available:

```powershell
.\.venv\Scripts\python.exe -m forge.ml.refinement
.\.venv\Scripts\python.exe -m pytest tests/test_refinement.py -q
```

The command freezes its configuration and writes a unique ignored
`artifacts/refinement/<run-id>/` folder. It saves fold allocations, fit groups,
training scales, threshold scans, selected policies, outer prediction arrays,
source/configuration hashes and measured outcomes. It neither refits a final
production model nor updates either model pointer. The public
[JSON result](refinement-results.json) contains aggregate and per-group evidence,
not raw sensor data.

Run `20261010T184310Z-8fbbcc72` took 106.2 seconds with one CPU thread. The five
shared outer fits took 17.47 seconds in total, raw scoring and readiness took
0.52 seconds, and the additional median calculation took 0.18 seconds. The full
elapsed time also includes 15 inner fits, threshold selection and artifact writes.
These are one-run measurements on the development machine, not a latency SLA or
a controlled hardware benchmark.

Eleven targeted tests cover future-score invariance, readiness and gap resets,
nonfinite scores, invalid configuration and timestamps, per-source boundaries,
inner group separation and complete out-of-fold coverage. A synthetic nested
rerun flips only the first outer fold's labels and verifies that its selected
policies, training reference and saved scores remain identical, while its
assessment changes. The test also rejects any attempted partition access beyond
training and validation. This substantiates implementation boundaries, but cannot
remove the retrospective nature or limited diversity of the dataset.

Smoothing remains an inactive experiment. The evidence supports fewer fragmented
alerts, not reliable physical-failure detection. Better operating-condition
coverage and independently collected evaluation recordings are still needed
before making that stronger claim.
