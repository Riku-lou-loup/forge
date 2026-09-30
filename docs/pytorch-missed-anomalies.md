# Missed-anomaly diagnosis for the PyTorch demo

This read-only diagnosis inspects the two validation groups with zero point
recall in `pytorch-20261007T111233Z-b8da62e6`. It uses the existing checkpoint,
threshold 0.5169919431209564 and three-reading persistence rule. No fitting,
threshold selection, source changes or original test CSV access was performed.
It covers these two groups, not every missed anomaly in validation.

## What the scores show

Score summaries below include only available readings. Labels are used after
inference to inspect the result and are not model inputs.

| Recording | Normal median | Anomaly median | Maximum anomaly score | Missed anomalous readings | Unavailable anomalies |
| --- | ---: | ---: | ---: | ---: | ---: |
| valve2/1 | 0.337772 | 0.418414 | 0.513064 | 333 | 31 |
| other/1 | 0.000002190 | 0.000080795 | 0.000429062 | 188 | 0 |

Neither recording has a single available reading above the selected threshold.
Consequently, removing the persistence requirement would not recover any alerts
at this threshold. Unavailable windows account for 31 of the 333 missed anomalies
in valve2/1, but cannot explain the remaining misses.

The network responds to the annotated intervals in both recordings. The score
levels differ substantially between recordings, however. Within-recording ROC AUC
is 0.89854 for valve2/1 and 0.99458 for other/1 on available readings. These are
post-hoc ranking diagnostics, not accuracy or independent performance estimates.
They show that an absence of alerts need not mean an absence of score response.

This supports investigating the consistency of score levels across operating
conditions. It does not establish why the levels differ or prove that a single
lower threshold would be useful. Lowering the shared threshold based on these
recordings could increase false alerts elsewhere. A threshold chosen separately
from each recording's anomaly labels would leak the answer into the decision.

## Consequence for the investigation graph

A real PyTorch/BM25 investigation of valve1/1 followed observe, triage, retrieve,
draft, verify and human review. Its trace counted two tool calls: observation
through the detector and evidence retrieval. The report required review.

For valve2/1, the trace followed observe, triage and finalization with status
`no_alert` and one tool call. Retrieval was never invoked. The current workflow
therefore cannot use retrieved documents to rescue a missed sensor alert.
Its `human_review` trace entry records an awaiting-user state, not a completed
review or an external human task notification.

The graph calls ordinary local Python capabilities. Rules control routing and
one narrower retrieval retry when evidence is missing. Drafting copies verified
checks from the corpus. No LLM chooses a tool, rewrites a query or generates an
explanation. Citation verification is a graph step, not an additional counted
tool call. Default budgets allow eight steps and three tool calls; elapsed time
is checked between local calls rather than enforced as a process timeout.

## Next controlled experiment

The proposed experiment compares the frozen CNN's current globally normalized
inputs with inputs expressed relative to each recording's first 60 readings.
The hypothesis is that removing an operating-level offset may make scores more
consistent across recordings. The observations above motivate this hypothesis
but do not establish that it will work.

Keep the architecture, seed, eight-epoch budget, window size and persistence
fixed. Run both representations under the same existing five outer and three
inner grouped development allocations. Fit global normalization only on each
training fold. Select the precision-first threshold inside the inner folds with
the same declared recall floor. Exclude each outer group's labels from fitting
and selection, and preserve known overlap groups.

A relative representation must wait until its startup reference and causal
window are available. Report unavailable coverage and count missed unavailable
anomalies consistently, alongside pooled and per-group precision, recall, false
positive rate and false alert onsets. The first-60 reference assumes a usable
startup period and needs explicit qualification for recordings that begin in an
abnormal state.

This experiment has been specified, not run. It uses already-inspected development
data and would remain a retrospective comparison. It would not replace the active
model or establish performance on new equipment.
