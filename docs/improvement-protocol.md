# Reducing false alerts: development protocol

The first detector flags unfamiliar normal operating conditions and produces too
many alerts. This experiment uses the existing training/validation allocation to
develop a lower-alert operating point. The already inspected test partition is
not used for selection or rescored. Any new generalization claim requires a new
evaluation plan and fresh holdout.

The balanced objective requires pooled point recall of at least 0.85, event recall
of at least 0.80, and point recall of at least 0.50 in every validation group with
anomalies. Among feasible choices, minimize point false-positive rate, then false
alert onsets per normal hour, then maximize F1. Targets of at most 5% point FPR
and six false onsets per normal hour are development targets, not industry safety
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
