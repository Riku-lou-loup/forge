# Generalization audit protocol

The precision-focused model was selected on the same validation recordings used
to report its 95.1% precision. This audit asks whether its training procedure
continues to work when different whole recordings are excluded. It also tests
whether a small, fixed hyperparameter search improves precision over the current
gradient-boosting configuration.

The audit combines the original training and validation partitions, giving 23
source recordings in 21 overlap groups. It does not change the original split
manifest or read the old test CSVs. These development recordings and their
results have already been inspected. Nested cross-validation can assess the
specified procedure under new fold assignments, but it cannot undo those earlier
design choices or turn these data into an independent benchmark.

## What is held out

Five outer folds divide whole overlap groups using GroupKFold with shuffle and
seed 42. Every development group is assessed once. Three inner folds divide only
the remaining outer-training groups. The same group assignments apply to all
compared procedures. Timestamps preserve order inside each source recording;
this is a recording-group audit, not a forward-time evaluation. Source files
sharing observations always stay together. A runtime check also rejects exact
sensor-vector duplication across group boundaries.

Every inner fit recomputes robust scales from its own normal training readings.
It then fits a new estimator on its own permitted training recordings. Causal
features initialize separately for each source file before deduplication, as in
the application. No validation or outer-assessment readings enter the learned
global scales or estimator fitting. The first 60 readings of an assessed source
file still supply its unlabeled startup reference, exactly as they would at
inference. They remain unscored, and startup anomalies count as misses.

## What is compared and tuned

Three procedures receive the same outer folds. The Isolation Forest reference
is refitted on normal training observations using its original 256-tree,
1,024-sample configuration. The current gradient-boosting reference is refitted
with its existing hyperparameters. The tuned gradient-boosting procedure chooses
among that configuration and three declared alternatives, varying tree size,
regularization, learning rate and iterations. Its relative features and startup
length remain fixed. This is hyperparameter tuning with fresh fits, not continued
training of the app's saved estimator.

All three procedures select their threshold and persistence on inner
out-of-fold predictions. This keeps the comparison from giving one procedure a
threshold chosen with access to its outer assessment data. Selection uses the
existing minimum recall requirements: 60% pooled point recall, 60% event recall
and 20% point recall in each inner group. Among qualifying settings, maximize
precision, then minimize false alert onsets per available normal hour, then
maximize point recall. Threshold candidates are 101 score quantiles, with
persistence of three or five readings. These settings and the four candidate
definitions are fixed in `configs/generalization-v1.json` before fitting.

Some inner comparisons may have no setting that meets all recall requirements.
Such a fold must remain in the report. The declared fallback maximizes the
minimum capped ratio between achieved recall and each recall requirement, then
uses the same precision and alert-frequency tie-breakers. The output marks the
selection infeasible. This produces measurable predictions without silently
relaxing the requirements or discarding a difficult fold.

Once inner selection finishes, refit the chosen estimator and scales on all
outer-training groups and apply the selected threshold unchanged to the excluded
outer groups. Numerical classifier scores can shift after refitting. The outer
assessment measures the consequence of that transfer rather than recalibrating
against the excluded groups.

## How results will be interpreted

Report pooled confusion counts and precision alongside recall, false alert
frequency, available-normal FPR, event detection and delay. Also show results by
fold and group. The long source-normal overlap group contributes much more
normal exposure than other groups, so pooled results alone are insufficient.
Compare training and outer results at the same selected operating point to
expose large gaps, without treating a gap as proof of a single cause. No event
point adjustment or score-based removal of difficult groups is allowed.

The primary comparison is the nested tuned procedure against the refitted
current configuration on the same excluded groups. The original 95.1% result
covers a different selection setting and a smaller data pool, so its numerical
difference from this audit is not a direct measure of overfitting. Precision is
reported with its alert count and recall; a nearly silent model must not be
presented as successful because its few alerts happen to be correct.

After the nested assessment, a five-fold grouped comparison over the full
development pool selects final settings for a research artifact. Refit that
candidate on the full development pool and save it with hashes and provenance.
These final selection scores are not independent evaluation. The artifact is
not activated: the app's existing model, baseline and historical results remain
unchanged. The report must distinguish tuning gains, regressions and unmet
targets. A negative result still answers the audit question.

The design follows scikit-learn's explanations of
[nested model selection](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
and [grouped folds](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
Fresh recordings from another period or asset are still needed for a stronger
generalization claim.
