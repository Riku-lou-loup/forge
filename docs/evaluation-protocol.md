# Baseline evaluation protocol

This protocol is fixed before looking at held-out model results. The configuration
is `configs/baseline.json`; the recording allocation is `skab-experiment-v1`.

Fit on the 18,306 unique normal training observations. Use all eight instantaneous
sensor readings. Exclude timestamps, source identifiers, anomaly labels and
change-point labels from model inputs. The unannotated normal recording is eligible
because of its source designation, not an invented row annotation.

Compare maximum absolute robust deviation with two Isolation Forests, each with
256 trees and random seed 42, sampling 256 or 1,024 training observations per tree.
The robust reference uses median and 1.4826 times MAD, falling back to IQR/1.349
when MAD is zero. Pressure needs this fallback. A channel with both spreads zero
raises an error and needs an explicit sensor-specific policy. Isolation Forest
uses the raw eight inputs; its scores are not calibrated probabilities. Reference
deviations explain unusual readings, not the forest's feature attribution.

For each candidate, consider 101 quantiles of validation scores plus a threshold
below the minimum. An alert begins on the third consecutive score strictly above
the threshold. No earlier timestamp is relabeled. Gaps over two seconds reset the
counter. Select the highest pooled validation point F1; ties prefer fewer false
alarm onsets, then a higher threshold, then the earlier candidate. This is a
benchmark operating point, not a maintenance service-level agreement.

Merge recordings within each overlap-connected group and count a shared
timestamp plus sensor vector once. If duplicate anomaly annotations disagree,
mark that observation unknown, exclude it from metrics and threshold candidates,
and break evaluation alert/event continuity there. Never resolve it by voting.
Conflicting sensor vectors at one timestamp are an error. Report all exclusions.

Freeze the selected model, threshold, alert policy and validation results before
explicit test evaluation. Do not refit on validation. Run the frozen candidate
on test once; changes motivated by test performance would require a new holdout.
Test metadata audits do not count as model evaluation.

Report pooled point precision, recall, F1, average precision on continuous scores,
and per-group results. Macro average precision omits single-class groups. No
point adjustment is used. An event is a contiguous run of anomalous annotations
within a group, split at unknown observations or gaps over two seconds. Detection
requires a new alert onset inside the event. An alert already active before the
event earns no detection credit. Report event recall and delays among detected
events only; missed events have no finite delay.

False alarms count new alert onsets at normal observations. Normal exposure gives
each normal observation at most one second (the last gets one second), without
counting recording gaps. This is a sample-exposure approximation. False alarms
per normal hour and delay figures describe this alert policy, not deployment
reliability. SKAB's shared testbed and acquisition dates limit generalization;
these are not results on independent machines or SKAB's official split.
