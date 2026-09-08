"""Inspect validation score separation without fitting or changing the detector."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from forge.config import PROJECT_ROOT
from forge.data.datasets import evaluation_groups
from forge.ml.metrics import evaluate_group
from forge.ml.training import load_model

COLORS = {0: "#267f8c", 1: "#c35432"}
LABELS = {0: "Normal annotation", 1: "Anomaly annotation"}


def summarize_group(frame, scores, threshold, policy):
    """Keep raw score ranking separate from persistent alert classification."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError("Each observation needs one finite score.")
    metrics = evaluate_group(frame, scores, threshold, policy)
    normal_count = metrics["tn"] + metrics["fp"]
    summary = {
        "normal_rows": normal_count,
        "anomaly_rows": metrics["positive_rows"],
        "excluded_unknown_rows": int(frame.anomaly.lt(0).sum()),
        "fpr": metrics["fp"] / normal_count if normal_count else None,
        "recall": metrics["recall"] if metrics["positive_rows"] else None,
        "roc_auc": metrics["roc_auc"],
        "average_precision": metrics["average_precision"],
        "anomaly_prevalence": metrics["positive_rows"] / metrics["rows"]
        if metrics["rows"]
        else None,
    }
    for label, prefix in [(0, "normal"), (1, "anomaly")]:
        values = scores[frame.anomaly.to_numpy() == label]
        for quantile in (10, 50, 90):
            summary[f"{prefix}_p{quantile}"] = (
                float(np.percentile(values, quantile)) if len(values) else None
            )
        summary[f"{prefix}_above_threshold"] = (
            float(np.mean(values > threshold)) if len(values) else None
        )
    return summary


def analyze_validation(root=PROJECT_ROOT):
    """This entry point deliberately has no test-partition or threshold override."""
    detector, metadata, _ = load_model(root)
    groups, audit = evaluation_groups("validation", root=root)
    scores = {name: detector.score(frame) for name, frame in groups.items()}
    rows = []
    for name, frame in groups.items():
        rows.append(
            {
                "group": name,
                **summarize_group(frame, scores[name], metadata["threshold"], metadata["policy"]),
            }
        )
    table = pd.DataFrame(rows).set_index("group")
    provenance = {
        "partition": "validation",
        "run_id": metadata["run_id"],
        "model_sha256": metadata["model_sha256"],
        "split_id": metadata["split_id"],
        "threshold": metadata["threshold"],
        "policy": metadata["policy"],
        "audit": audit,
        "scope": "Descriptive diagnostics of the frozen model. No fitting, threshold selection, or test scoring.",
    }
    return groups, scores, table, provenance


def histograms(groups, scores, bins=32):
    """Common score bins; each class sums to 100% within its own group."""
    values = np.concatenate(
        [scores[name][frame.anomaly.to_numpy() >= 0] for name, frame in groups.items()]
    )
    edges = np.histogram_bin_edges(values, bins=bins)
    rows = []
    for name, frame in groups.items():
        for label in (0, 1):
            subset = scores[name][frame.anomaly.to_numpy() == label]
            if not len(subset):
                continue
            counts, _ = np.histogram(subset, bins=edges)
            rows.extend(
                {
                    "group": name,
                    "label": label,
                    "left": float(left),
                    "right": float(right),
                    "count": int(count),
                    "percent": float(100 * count / len(subset)),
                }
                for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True)
            )
    return pd.DataFrame(rows)


def distribution_figure(binned, table, provenance):
    names = list(table.index)
    rows = (len(names) + 1) // 2
    titles = [
        f"{name} · FPR {table.loc[name, 'fpr']:.1%} · AUC {table.loc[name, 'roc_auc']:.3f}"
        for name in names
    ]
    figure = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=titles,
        shared_xaxes="all",
        shared_yaxes="all",
        vertical_spacing=0.075,
    )
    for index, name in enumerate(names):
        row, col = index // 2 + 1, index % 2 + 1
        for label in (0, 1):
            selected = binned.loc[(binned.group == name) & binned.label.eq(label)]
            figure.add_trace(
                go.Bar(
                    x=(selected.left + selected.right) / 2,
                    y=selected.percent,
                    width=selected.right - selected.left,
                    name=LABELS[label],
                    legendgroup=str(label),
                    showlegend=index == 0,
                    marker_color=COLORS[label],
                    opacity=0.55,
                    hovertemplate="Score %{x:.3f}<br>%{y:.1f}% of class observations<extra>%{fullData.name}</extra>",
                ),
                row=row,
                col=col,
            )
        figure.add_vline(
            x=provenance["threshold"], line_dash="dash", line_color="#222222", row=row, col=col
        )
    figure.update_layout(
        title=f"Validation score distributions · frozen threshold {provenance['threshold']:.4f}",
        template="plotly_white",
        barmode="overlay",
        height=310 * rows + 100,
        legend={"orientation": "h", "y": 1.065},
        margin={"t": 110, "b": 80},
    )
    figure.update_xaxes(title_text="Anomaly score (higher = more unusual)")
    figure.update_yaxes(title_text="Within-class observations (%)")
    figure.add_annotation(
        text="Dashed line = frozen threshold. FPR uses three-reading persistence; histograms and AUC use raw scores.",
        x=0.5,
        y=-0.065,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    return figure


def export_diagnostics(groups, scores, table, provenance, root=PROJECT_ROOT):
    """Save an offline interactive plot, a shareable PNG, and aggregate statistics."""
    # Matplotlib is part of the development environment, not required for scoring.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output = Path(root) / "reports/figures" / f"validation-{provenance['run_id']}"
    output.mkdir(parents=True, exist_ok=True)
    binned = histograms(groups, scores)
    figure = distribution_figure(binned, table, provenance)
    figure.write_html(output / "score-distributions.html", include_plotlyjs=True, auto_open=False)
    table.to_csv(output / "score-summary.csv")
    payload = {
        **provenance,
        "groups": json.loads(table.reset_index().to_json(orient="records", double_precision=15)),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    names = list(table.index)
    fig, axes = plt.subplots(
        (len(names) + 1) // 2, 2, figsize=(12, 12), sharex=True, sharey=True, squeeze=False
    )
    for index, name in enumerate(names):
        axis = axes.flat[index]
        for label in (0, 1):
            selected = binned.loc[(binned.group == name) & binned.label.eq(label)]
            if selected.empty:
                continue
            edges = np.r_[selected.left.to_numpy(), selected.right.iloc[-1]]
            axis.stairs(selected.percent, edges, color=COLORS[label], fill=True, alpha=0.22)
            axis.stairs(
                selected.percent,
                edges,
                color=COLORS[label],
                linewidth=1.4,
                linestyle="-" if label == 0 else ":",
            )
        axis.axvline(provenance["threshold"], color="#222222", linestyle="--", linewidth=1)
        axis.set_title(
            f"{name} | FPR {table.loc[name, 'fpr']:.1%} | AUC {table.loc[name, 'roc_auc']:.3f}",
            fontsize=11,
            loc="left",
        )
        axis.grid(axis="y", alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
    for index in range(len(names), axes.size):
        axes.flat[index].set_visible(False)
    fig.suptitle("Normal and anomalous scores on validation recordings", fontsize=17, y=0.985)
    handles = [
        Line2D(
            [0], [0], color=COLORS[label], linestyle="-" if label == 0 else ":", label=LABELS[label]
        )
        for label in (0, 1)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#222222",
            linestyle="--",
            label=f"Frozen threshold {provenance['threshold']:.4f}",
        )
    )
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False
    )
    fig.supxlabel("Anomaly score (higher = more unusual)", y=0.045)
    fig.supylabel("Within-class observations (%)", x=0.018)
    fig.text(
        0.5,
        0.014,
        "Each class is normalized separately. FPR includes persistence; distributions and AUC use raw scores.\nValidation only · frozen model · no refitting or threshold changes",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0.035, 0.065, 1, 0.915), h_pad=2.2)
    fig.savefig(output / "score-distributions.png", dpi=170, facecolor="white")
    plt.close(fig)
    return output


def main():
    groups, scores, table, provenance = analyze_validation()
    output = export_diagnostics(groups, scores, table, provenance)
    print(
        table[
            ["normal_rows", "anomaly_rows", "fpr", "recall", "roc_auc", "normal_p50", "anomaly_p50"]
        ].to_string(float_format=lambda x: f"{x:.4f}")
    )
    print(f"\nSaved validation diagnostics: {output}")


if __name__ == "__main__":
    main()
