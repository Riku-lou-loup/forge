"""Build the generalization report and notebook from one completed local audit."""

import argparse
import json
from pathlib import Path

import nbformat

from forge.config import PROJECT_ROOT
from forge.ml.training import digest

NAMES = {
    "isolation_forest": "Isolation Forest",
    "current_hgb": "Current gradient-boosting configuration",
    "tuned_hgb": "Tuned gradient-boosting procedure",
}


def percentage(value):
    return f"{value:.1%}"


def build_markdown(result):
    current = result["procedures"]["current_hgb"]["pooled"]
    tuned = result["procedures"]["tuned_hgb"]["pooled"]
    difference = 100 * (tuned["precision"] - current["precision"])
    recall_difference = 100 * (tuned["recall"] - current["recall"])
    final = result["final_refit"]
    worst_group, worst_metrics = max(
        result["procedures"]["tuned_hgb"]["per_group"].items(), key=lambda item: item[1]["fp"]
    )
    worst_sources = next(item["sources"] for item in result["pool"] if item["group"] == worst_group)
    unmet_folds = sum(
        not fold["procedures"]["tuned_hgb"]["selection"]["feasible"]
        for fold in result["fold_results"]
    )
    lines = [
        "# Generalization and tuning audit",
        "",
        "This audit refits FORGE’s detectors while excluding different whole recording",
        "groups. It tests whether the current gradient-boosting configuration remains",
        "effective beyond its original training allocation and whether a fixed, small",
        "hyperparameter search improves its precision.",
        "",
        f"The tuned procedure reaches **{percentage(tuned['precision'])} precision** and",
        f"**{percentage(tuned['recall'])} recall** on pooled outer-fold predictions.",
        f"The current configuration reaches {percentage(current['precision'])} precision",
        f"and {percentage(current['recall'])} recall under the same fold protocol.",
        f"Tuning changes precision by **{difference:+.1f} percentage points** and",
        f"recall by **{recall_difference:+.1f} percentage points**. These are development",
        "audit results, not a new-machine reliability claim.",
        "",
        f"In {unmet_folds} of the five outer folds, no gradient-boosting candidate met",
        "all inner recall requirements, so the declared fallback supplied the alert",
        "settings. The per-fold results and unmet alert targets below are part of the",
        "outcome, not exceptions removed from the comparison.",
        "",
        "## What the comparison measures",
        "",
        "The pool contains 23 previously inspected source recordings in 21 overlap",
        "groups, drawn from the original training and validation partitions. No old",
        "test CSV was read. Five outer folds assess every group once. Within each",
        "outer-training pool, three inner folds select the threshold and persistence.",
        "The tuned procedure also chooses among four declared gradient-boosting",
        "configurations. Whole overlap groups stay together at both levels.",
        "",
        "“Current configuration” means a new fit using the existing hyperparameters,",
        "with its alert settings selected inside each outer fold. It is not the app’s",
        "saved model scored on observations it already fitted. Isolation Forest is",
        "also refitted in each fold and receives the same inner alert-selection rule.",
        "Global robust scales are recomputed from each fit’s normal observations.",
        "The relative models still initialize their unlabeled reference from the first",
        "60 readings of each assessed source file, matching the application.",
        "",
        "The original 95.1% precision used a different, smaller validation allocation",
        "and thresholds chosen on that allocation. Its difference from the figures",
        "below cannot by itself quantify overfitting. The paired outer-fold comparison",
        "is the relevant measure of the declared tuning procedure.",
        "",
        "## Pooled outer-fold results",
        "",
        "| Procedure | Precision | Recall | False-positive readings | Missed anomalous readings | False alert onsets | Onsets / available normal hour | Events detected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, entry in result["procedures"].items():
        m = entry["pooled"]
        lines.append(
            f"| {NAMES[name]} | {percentage(m['precision'])} | {percentage(m['recall'])} | {m['fp']:,} | {m['fn']:,} | {m['false_alarm_onsets']} | {m['false_onsets_per_available_normal_hour']:.2f} | {m['detected_events']} / {m['events']} |"
        )
    lines += [
        "",
        "Relative to the current configuration, tuning changes false-positive readings",
        f"by {tuned['fp'] - current['fp']:+,}, missed anomalous readings by",
        f"{tuned['fn'] - current['fn']:+,}, and separate false alert episodes by",
        f"{tuned['false_alarm_onsets'] - current['false_alarm_onsets']:+,}.",
        "A small precision gain should be assessed alongside these changes rather",
        "than taken as a sufficient reason to replace the active model.",
        "",
        "| Target for the tuned procedure | Achieved? |",
        "|---|---|",
        f"| Precision at least 95% | {'Yes' if tuned['precision'] >= result['config']['target_precision'] else 'No'} |",
        f"| Available-normal FPR at most 1% | {'Yes' if tuned['fpr_available_normal'] <= result['config']['target_false_positive_rate'] else 'No'} |",
        f"| At most two false onsets per available normal hour | {'Yes' if tuned['false_onsets_per_available_normal_hour'] <= result['config']['target_false_alarm_onsets_per_normal_hour'] else 'No'} |",
    ]
    lines += [
        "",
        "Precision is the share of alerted readings annotated anomalous. Recall is",
        "the share of anomalous readings that trigger an alert. Precision is reported",
        "as zero when no alerts occur, and the JSON retains the alert count. A 100%",
        "precision value means there were no false-positive readings among the alerts.",
        "It does not mean that all anomalies were detected or that accuracy is 100%.",
        "",
        "Initialization readings receive no assessment. Their normal exposure is",
        "excluded from FPR and false alert frequency, while startup anomalies still",
        "count as misses. The full confusion counts retain unscored normal readings",
        "as normal readings without an alert. An event counts as detected only when",
        "a new alert onset falls inside its annotated interval. No point adjustment",
        "expands an alert across an entire anomalous interval.",
        "",
        "| Procedure | Available-normal FPR | Macro group precision | Macro group recall | Initialization readings | Median detected-event delay |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, entry in result["procedures"].items():
        m = entry["pooled"]
        delay = m["median_detected_event_delay_seconds"]
        lines.append(
            f"| {NAMES[name]} | {percentage(m['fpr_available_normal'])} | {percentage(m['macro_group_precision'])} | {percentage(m['macro_group_recall'])} | {m['unavailable_rows']:,} | {delay:g} s |"
            if delay is not None
            else f"| {NAMES[name]} | {percentage(m['fpr_available_normal'])} | {percentage(m['macro_group_precision'])} | {percentage(m['macro_group_recall'])} | {m['unavailable_rows']:,} | No detected events |"
        )
    lines += [
        "",
        "Macro metrics give each recording group equal weight. Pooled metrics count",
        "all readings together, so the long source-normal overlap group contributes",
        "much more normal exposure. Delay medians cover detected events only and may",
        "refer to different event sets for different procedures.",
        "",
        "## Variation across folds",
        "",
        "| Outer fold | Current precision | Tuned precision | Current recall | Tuned recall | Chosen tuned candidate | Inner recall requirements met? |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for fold in result["fold_results"]:
        a = fold["procedures"]["current_hgb"]["assessment"]["pooled"]
        b = fold["procedures"]["tuned_hgb"]["assessment"]["pooled"]
        selected = fold["procedures"]["tuned_hgb"]
        lines.append(
            f"| {fold['fold']} | {percentage(a['precision'])} | {percentage(b['precision'])} | {percentage(a['recall'])} | {percentage(b['recall'])} | {selected['candidate']} | {'Yes' if selected['selection']['feasible'] else 'No; declared fallback'} |"
        )
    lines += [
        "",
        "Inner selection requires 60% point recall, 60% event recall and 20% point",
        "recall in every inner group. If no setting qualifies, the predeclared",
        "fallback maximizes the weakest normalized recall requirement before applying",
        "the precision tie-breaker. Such folds remain in the comparison and are",
        "marked above. Inner feasibility does not guarantee outer feasibility.",
        "These requirements can select a low threshold that retains anomaly readings",
        "but also alerts on many normal readings. The audit evaluates that complete",
        "selection procedure, including its fallback, rather than tree capacity alone.",
        "",
        "| Procedure | Mean training precision | Mean outer precision | Mean training recall | Mean outer recall | Outer precision range |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        train = [f["procedures"][name]["training"] for f in result["fold_results"]]
        outer = [f["procedures"][name]["assessment"]["pooled"] for f in result["fold_results"]]

        def avg(rows, key):
            return sum(m[key] for m in rows) / len(rows)

        lines.append(
            f"| {NAMES[name]} | {percentage(avg(train, 'precision'))} | {percentage(avg(outer, 'precision'))} | {percentage(avg(train, 'recall'))} | {percentage(avg(outer, 'recall'))} | {percentage(min(m['precision'] for m in outer))}–{percentage(max(m['precision'] for m in outer))} |"
        )
    lines += [
        "",
        "Each training score uses the same fitted model and alert settings as its",
        "outer assessment. Training readings were used for fitting, so these are",
        "resubstitution scores, not another validation result. Large gaps warrant",
        "concern about fit and operating-condition sensitivity, but they do not",
        "separate memorization from distribution differences by themselves. Means",
        "and ranges above describe five folds, not confidence intervals.",
        "",
        f"The largest false-positive count belongs to `{worst_group}` with",
        f"{worst_metrics['fp']:,} readings, or {worst_metrics['fp'] / tuned['fp']:.1%} of",
        "the tuned procedure's false-positive readings. This group contains",
        f"{', '.join(f'`{name}`' for name in worst_sources)}. This concentration helps locate",
        "the weakness, but it does not establish which sensor or mechanism caused it.",
        "",
        "## Final tuning artifact",
        "",
        "After the nested assessment, five-fold grouped selection across the full",
        f"development pool selected `{final['candidate']}`. It was refitted on",
        f"{final['fit_audit']['fit_rows']:,} available observations, including",
        f"{final['fit_audit']['positive_training_targets']:,} anomalous training targets.",
        f"The selected threshold is `{final['selection']['threshold']}` with",
        f"{final['selection']['policy']['persistence']} consecutive above-threshold readings.",
        f"Its full-pool inner selection status is `{final['selection']['status']}`.",
        "",
        "This is a research artifact. Its full-pool selection scores are not",
        "independent performance, and it has **not been activated**. The app’s prior",
        "model, original baseline and historical result files retain their recorded",
        "hashes. The JSON records the final candidate settings and model checksum.",
        "",
        "## Limits and next evidence",
        "",
        "The data and earlier results influenced the model family and feature design",
        "before this audit. Nested grouping protects the current fitting and tuning",
        "steps from their outer assessment groups, but it cannot remove that prior",
        "development history. Overlap groups are also not independent factories or",
        "assets. Source-file order is preserved, but groups are not evaluated in a",
        "strictly forward-time deployment simulation.",
        "",
        "A reference built from an already faulty startup may miss that fault.",
        "SKAB annotations identify anomalous readings rather than confirmed physical",
        "failures. Further claims need fresh recordings, with the model and alert",
        "settings frozen before scoring. The next model change should follow the",
        "group-level failures shown here, rather than repeated tuning against these",
        "outer scores until they look favorable.",
        "",
        "## Reproduction and provenance",
        "",
        "From the repository, using its environment:",
        "",
        "```powershell",
        r".\.venv\Scripts\python.exe -m forge.ml.generalization --plan",
        r".\.venv\Scripts\python.exe -m forge.ml.generalization",
        r".\.venv\Scripts\python.exe scripts/report_generalization.py <run_id>",
        "```",
        "",
        f"Reported run: `{result['run_id']}`. The [JSON summary](generalization-results.json)",
        "contains the exact group allocation, configuration, per-group outcomes,",
        "training audits and source hashes. Ignored local run folders retain inner",
        "selection ledgers, outer prediction arrays and fitted models. The publisher",
        "builds a source-only [notebook 05](../notebooks/05_generalization_audit.ipynb).",
        "Run its cells to display the saved results inline. This notebook does not",
        "retrain a model or read test recordings.",
        "",
        "See the [protocol](generalization-protocol.md) for the fixed search and",
        "selection rules. Software tests include changing only excluded-group labels",
        "and checking that the corresponding model and selected settings do not move.",
        "Those checks establish specific implementation properties, not the absence",
        "of statistical overfitting.",
        "",
    ]
    return "\n".join(lines)


def build_notebook(result):
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python (FORGE)",
        "language": "python",
        "name": "python3",
    }
    markdown = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    current = result["procedures"]["current_hgb"]["pooled"]
    tuned = result["procedures"]["tuned_hgb"]["pooled"]
    nb.cells = [
        markdown("""# Generalization audit and model tuning

The previous model was selected on one validation allocation. This audit excludes different whole recording groups and repeats fitting and selection inside each fold. It compares a refitted Isolation Forest, the current gradient-boosting configuration and a tuned gradient-boosting procedure. The app's saved model is preserved.

Five outer folds provide assessment predictions. Within each outer-training pool, three inner folds choose alert settings and, for the tuned procedure, one of four declared model configurations. Every global scale and estimator is fitted only on its permitted training groups. The first 60 readings of each assessed source recording still establish its unlabeled startup reference and receive no assessment.

These are **retrospective development results** on the original train/validation pool. The dataset and model family were already inspected before this audit. Grouped cross-validation strengthens the comparison but does not create fresh data or establish performance on another machine. The old test CSVs are not read."""),
        code("""import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from forge.config import PROJECT_ROOT

result = json.loads((PROJECT_ROOT / "docs/generalization-results.json").read_text())
names = {
    "isolation_forest": "Isolation Forest",
    "current_hgb": "Current gradient boosting",
    "tuned_hgb": "Tuned gradient boosting",
}
pool = pd.DataFrame(result["pool"])
pool[["group", "sources", "rows", "normal", "anomalous", "unknown"]]"""),
        markdown("""## Recording groups and fold boundaries

Names such as `valve2/1` identify recordings, not input features. Eight sensor channels supply the model features. Files sharing observations belong to the same overlap group, so copies cannot appear on both sides of a fold boundary.

The pool has 23 recordings in 21 groups. One group contains the much longer source-normal recording. The group table above exposes that imbalance. The allocation below is fixed before fitting, and each group is assessed once."""),
        code("""pd.DataFrame([
    {"outer_fold": fold["fold"], "fitting_groups": fold["fit_groups"], "excluded_groups": fold["assessment_groups"]}
    for fold in result["fold_results"]
])"""),
        markdown(f"""## Precision compared with recall

The tuned procedure reaches **{percentage(tuned["precision"])} precision** at **{percentage(tuned["recall"])} recall**, compared with **{percentage(current["precision"])} precision** at **{percentage(current["recall"])} recall** for the current configuration under this audit.

Relative to the current configuration, tuning changes false-positive readings by {tuned["fp"] - current["fp"]:+,}, missed anomalous readings by {tuned["fn"] - current["fn"]:+,} and separate false alert episodes by {tuned["false_alarm_onsets"] - current["false_alarm_onsets"]:+,}. The precision change must be weighed against those tradeoffs before considering activation. The report retains all three target checks alongside these results.

Precision is the share of alerted readings annotated anomalous. Recall is the share of anomalous readings that trigger alerts. A 100% precision value means no false-positive readings occurred among the alerts. It does not mean accuracy is 100% or that no anomalies were missed. Precision is displayed as zero when there are no alerts, with the alert count retained in the results.

The earlier 95.1% precision used a different allocation and selection setting. Compare the procedures below on their matched excluded groups, rather than interpreting the difference from 95.1% as a direct estimate of overfitting."""),
        code("""pooled = pd.DataFrame({names[name]: entry["pooled"] for name, entry in result["procedures"].items()}).T
pooled[["precision", "recall", "f1", "alerted_readings", "fp", "fn", "false_alarm_onsets", "false_onsets_per_available_normal_hour", "macro_group_precision", "macro_group_recall"]]"""),
        code("""fig = make_subplots(rows=1, cols=3, subplot_titles=list(names.values()))
for column, (name, entry) in enumerate(result["procedures"].items(), start=1):
    m = entry["pooled"]
    matrix = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    fig.add_trace(go.Heatmap(z=matrix, x=["No alert", "Alert"], y=["Normal", "Anomaly"],
                            text=matrix, texttemplate="%{text}", colorscale="Blues", showscale=False,
                            zmin=0, zmax=int(pooled[["tn", "fp", "fn", "tp"]].to_numpy().max())), row=1, col=column)
fig.update_yaxes(autorange="reversed")
fig.update_layout(height=370, title="Outer-fold confusion counts: each group assessed once")
fig.show()"""),
        markdown("""## Does tuning help consistently?

Pooled precision can hide difficult groups. These plots compare precision and recall in each outer fold, using the same excluded groups for all three procedures. The table also shows whether inner selection met the minimum recall requirements. A failed requirement invokes the declared fallback and remains visible in the audit.

The tuned procedure may choose different hyperparameters in different folds. That is expected when evaluating a tuning procedure. None of those choices uses the corresponding outer-assessment labels."""),
        code("""fold_rows = []
for fold in result["fold_results"]:
    for name, outcome in fold["procedures"].items():
        m = outcome["assessment"]["pooled"]
        fold_rows.append({"fold": fold["fold"], "procedure": names[name], "candidate": outcome["candidate"],
                          "precision": m["precision"], "recall": m["recall"],
                          "training_precision": outcome["training"]["precision"],
                          "training_recall": outcome["training"]["recall"],
                          "inner_status": outcome["selection"]["status"]})
folds = pd.DataFrame(fold_rows)
fig = make_subplots(rows=1, cols=2, subplot_titles=["Precision", "Recall"])
for procedure, color in zip(names.values(), ["#4777b8", "#d58036", "#35866b"], strict=True):
    selected = folds.loc[folds.procedure.eq(procedure)]
    for column, metric in enumerate(["precision", "recall"], start=1):
        fig.add_trace(go.Scatter(x=selected.fold, y=selected[metric], name=procedure,
                                legendgroup=procedure, showlegend=column == 1, mode="lines+markers",
                                line={"color": color}, marker={"color": color}), row=1, col=column)
fig.update_yaxes(range=[0, 1])
fig.update_xaxes(title_text="Outer fold", dtick=1)
fig.update_layout(height=400, legend={"orientation": "h", "y": -0.25})
fig.show()
folds"""),
        markdown("""## Training results and excluded-group results

Each training score uses the same fitted estimator and threshold as its outer assessment. Training readings were used for fitting, so these scores are expected to be optimistic. A large gap is a warning about fit or operating-condition sensitivity, but it cannot distinguish those causes by itself. The table is descriptive rather than a confidence interval."""),
        code(
            """folds.groupby("procedure")[["training_precision", "precision", "training_recall", "recall"]].mean()"""
        ),
        markdown("""## Find the groups behind the aggregate

The following table pairs both gradient-boosting procedures on every group. Precision alone is insufficient: inspect false-positive and missed-anomaly counts together with recall. Initialization readings are unscored, but startup anomalies still count as misses. Available-normal FPR and alert frequency exclude unscored normal exposure."""),
        code("""group_rows = []
for name, entry in result["procedures"].items():
    for group, m in entry["per_group"].items():
        group_rows.append({"procedure": names[name], "group": group, "precision": m["precision"],
                           "recall": m["recall"], "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                           "initialization_readings": m["unavailable_rows"], "false_onsets": m["false_alarm_onsets"]})
group_results = pd.DataFrame(group_rows)
group_results.pivot(index="group", columns="procedure", values=["precision", "recall", "fp", "fn"])"""),
        markdown("""## Final tuning artifact and remaining evidence

After the nested assessment, a separate five-fold grouped selection across the full development pool chooses final settings and refits a research artifact. Its selection scores are not independent evaluation. The artifact is saved locally and is **not activated**. The app's prior model and baseline remain unchanged.

The current data influenced the feature design before this audit. Further tuning against the outer results would make them another selection set. A stronger generalization assessment needs new recordings, with settings frozen before scoring. An unlabeled reference from an already faulty startup remains a limitation.

This notebook reads saved aggregate results only. See the [report](../docs/generalization-results.md) and [protocol](../docs/generalization-protocol.md) for commands, hashes and selection rules."""),
        code("""final = result["final_refit"]
pd.Series({"candidate": final["candidate"], "threshold": final["selection"]["threshold"],
           "persistence": final["selection"]["policy"]["persistence"], "selection_status": final["selection"]["status"],
           "fit_rows": final["fit_audit"]["fit_rows"], "activated": final["activated"],
           "model_sha256": final["model_sha256"]})"""),
    ]
    return nb


def publish(run_id, root=PROJECT_ROOT):
    root = Path(root)
    folder = (root / "models" / run_id).resolve()
    if folder.parent != (root / "models").resolve():
        raise ValueError("Invalid local audit run path.")
    result = json.loads((folder / "audit-results.json").read_text())
    if result["run_id"] != run_id or result["schema_version"] != 1:
        raise ValueError("Invalid audit result schema.")
    if digest(folder / "detector.pkl") != result["final_refit"]["model_sha256"]:
        raise ValueError("Final research artifact checksum mismatch.")
    for name, checksum in result["preserved_sha256"].items():
        if digest(root / name) != checksum:
            raise ValueError(f"Preserved artifact changed: {name}")
    (root / "docs/generalization-results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (root / "docs/generalization-results.md").write_text(build_markdown(result), encoding="utf-8")
    nbformat.write(build_notebook(result), root / "notebooks/05_generalization_audit.ipynb")
    print("Published the audit report, aggregate JSON and notebook 05 source.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    publish(args.run_id)
