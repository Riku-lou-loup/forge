# Reducing false alerts: development protocol

The first detector flags unfamiliar normal operating conditions and produces too
many alerts. This experiment uses the existing training/validation allocation to
develop a lower-alert operating point. The already inspected test partition is
not used for selection or rescored. Any new generalization claim requires a new
evaluation plan and fresh holdout.

The user chose precision priority before any new candidate was fitted: fewer
alerts even if more anomalies are missed. Require pooled point recall of at least
0.60, event recall of at least 0.60, and point recall of at least 0.20 in every
validation group with anomalies to prevent a nearly silent detector from winning.
Among feasible choices, maximize precision, then minimize false alert onsets per
normal hour, then maximize point recall. Targets of at least 95% precision, at
most 1% point FPR and two false onsets per normal hour are development targets, not industry safety
requirements. Show per-group results, alert delay and coverage alongside aggregates.
These criteria are in `configs/improvement-v2.json` before fitting new candidates.

## Candidate comparison

Retain the frozen Isolation Forest and robust maximum-deviation baseline as
comparators. Check whether threshold/persistence changes alone can help. Compare
Extra Trees classifiers using three fixed representations: eight raw readings;
raw readings plus causal changes and rolling variation; and relative changes
against a recording's initial reference. Also compare an Isolation Forest trained
on normal relative features. Extra Trees uses source annotations as training
targets, while Isolation Forest still fits normal examples only. No annotation,
experiment identifier, absolute clock time or elapsed time becomes an input.

All learned transforms and estimator fitting use training recordings only.
Overlapping training observations are deduplicated before they contribute to
fitting. Supervised fitting may use anomalous training targets, a deliberate
change from the unsupervised first baseline. Weight each training leakage group
equally so the long source-normal file cannot dominate by length. Use the same
validation groups and deduplication rules for every candidate.

Causal features use the current and preceding observations, never future rows.
Rolling changes reset after gaps over two seconds; stable recording references
are preserved across gaps rather than relearned from a potentially faulty segment.
No window crosses a recording/group boundary. Global scaling is fitted on normal
training rows only.

### Serving-boundary correction

The first implementation initialized relative features on merged overlap groups,
which did not match the application's per-CSV initialization. Those initial
development results are retained as provisional artifacts and cannot be used to
promote a model. Schema version 2 initializes and computes features separately
for each source recording, then deduplicates observations before fitting or
evaluation. For a shared observation, the first inventory occurrence supplies
its prediction context; annotation conflicts remain unknown. Group weights and
reported metrics still use leakage groups. This correction changes neither the
selection objective nor the allowed training/validation partitions. Both declared
comparison stages are rerun with the corrected boundary.

The relative representation initializes a reference from the first 60 sensor
readings, with no label lookup. Scoring after reference initialization assumes
those initial readings represent an appropriate operating reference. This is an
explicit startup requirement, not a guaranteed property of new equipment. The
first 60 readings are unavailable for that candidate; startup anomalies must be
reported as missed, not removed from recall. Report coverage and false-positive
rates over available normal observations so startup suppression does not create
an apparent reduction by padding true negatives. The other representations do
not require this reference initialization.

## Selection and reporting

Consider 101 validation score quantiles and persistence of three or five readings.
The time-gap rule remains two seconds. Candidate scores and the selected policy
are development results, not newly held-out results. If recall floors cannot be
met, retain the previous default and report the tradeoff instead of calling a
lower-recall model an improvement. Do not silently change the goal after seeing
the comparison.

Before integrating a replacement, test feature causality, label exclusion,
recording boundaries, gap behavior, startup coverage, artifact round trips, and
compatibility with the investigation workflow. Preserve the original model and
published benchmark. An annotated SKAB anomaly is not a confirmed physical
failure, and a classifier score is not a calibrated failure probability.
