"""Plot measured FORGE comparison results without fitting or reading sensor data.

Usage:
    python scripts/plot_modern_comparison.py path/to/results.json --output path/to/modern-comparison.png
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

PROCEDURES = (
    ("historical_hgb", "Old HGB policy"),
    ("precision_hgb", "HGB F0.5"),
    ("precision_catboost", "CatBoost F0.5"),
    ("precision_tabm", "Compact TabM F0.5"),
)


def read_metrics(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("activated") is not False:
        raise ValueError("Expected an explicitly inactive comparison result.")
    rows = []
    exposures = set()
    for key, label in PROCEDURES:
        metrics = result["procedures"][key]["pooled"]
        counts = [metrics[name] for name in ("tp", "fp", "fn", "tn")]
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError(f"Invalid confusion counts for {key}.")
        tp, fp, fn, tn = counts
        if tp + fp + fn + tn == 0:
            raise ValueError(f"No assessed readings for {key}.")
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        for name, actual in (("precision", precision), ("recall", recall)):
            reported = metrics[name]
            if not isinstance(reported, (int, float)) or not math.isclose(
                reported, actual, rel_tol=1e-8, abs_tol=1e-10
            ):
                raise ValueError(f"Reported {name} disagrees with counts for {key}.")
        exposures.add((tp + fn, fp + tn))
        rows.append((label, precision * 100, recall * 100, fp))
    if len(exposures) != 1:
        raise ValueError("Procedures do not share the same annotated exposure.")
    return result, rows


def render(result_path, output_path):
    result, rows = read_metrics(result_path)
    labels = [row[0] for row in rows]
    precision = np.array([row[1] for row in rows])
    recall = np.array([row[2] for row in rows])
    false_positive = np.array([row[3] for row in rows])
    y = np.arange(len(rows))

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "text.color": "#172536",
            "axes.labelcolor": "#172536",
            "xtick.color": "#455264",
            "ytick.color": "#172536",
            "axes.edgecolor": "#BEC7D0",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    ):
        fig, (left, right) = plt.subplots(
            1,
            2,
            figsize=(13.4, 6.25),
            sharey=True,
            gridspec_kw={"width_ratios": (1.18, 1)},
        )
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.22, top=0.72, wspace=0.16)
        fig.suptitle(
            "FORGE: alert selection and model tradeoffs",
            x=0.035,
            y=0.965,
            ha="left",
            fontsize=18,
            fontweight="bold",
        )
        outer = result["config"]["outer_folds"]
        inner = result["config"]["inner_folds"]
        fig.text(
            0.035,
            0.905,
            f"Same development recordings and causal features | {outer} outer / {inner} inner grouped folds",
            fontsize=11,
            color="#455264",
        )

        bar_height = 0.27
        p_bars = left.barh(
            y - 0.16,
            precision,
            height=bar_height,
            color="#0072B2",
            label="Precision",
        )
        r_bars = left.barh(
            y + 0.16,
            recall,
            height=bar_height,
            color="#E69F00",
            edgecolor="#7B5300",
            linewidth=0.35,
            hatch="///",
            label="Recall",
        )
        left.set_title("Precision and recall", loc="left", pad=42)
        left.legend(
            loc="lower left",
            bbox_to_anchor=(-0.015, 1.015),
            ncol=2,
            frameon=False,
            handlelength=1.6,
            borderaxespad=0,
        )
        left.set_xlim(0, 113)
        left.set_xticks(np.arange(0, 101, 20))
        left.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
        left.set_xlabel("Share of alerts / share of annotated anomalies")
        left.set_yticks(y, labels)
        left.invert_yaxis()
        for bars, values in ((p_bars, precision), (r_bars, recall)):
            for bar, value in zip(bars, values, strict=True):
                left.text(
                    value + 1.1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}%",
                    va="center",
                    fontsize=10,
                )

        fp_bars = right.barh(y, false_positive, height=0.52, color="#667887")
        right.set_title("False-positive readings", loc="left", pad=42)
        right.set_xlabel("Normal readings that triggered an alert")
        upper = max(int(false_positive.max()), 1)
        right.set_xlim(0, upper * 1.23)
        right.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        if upper < 5:
            right.xaxis.set_major_locator(MultipleLocator(1))
        for bar, value in zip(fp_bars, false_positive, strict=True):
            right.text(
                value + upper * 0.018,
                bar.get_y() + bar.get_height() / 2,
                f"{int(value):,}",
                va="center",
                fontsize=11,
                fontweight="medium",
            )
        for axis in (left, right):
            axis.set_axisbelow(True)
            axis.grid(axis="x", color="#E4E8ED", linewidth=0.7)
            axis.tick_params(axis="both", length=0)
            axis.tick_params(axis="y", pad=10)
            for spine in ("top", "right", "left"):
                axis.spines[spine].set_visible(False)
        right.tick_params(axis="y", labelleft=False)

        fig.text(
            0.035,
            0.128,
            "False alert onsets, same row order: "
            + " / ".join(
                str(result["procedures"][key]["pooled"]["false_alarm_onsets"])
                for key, _ in PROCEDURES
            )
            + ". Fewer false readings need not mean fewer interruptions.",
            fontsize=10,
            color="#455264",
        )
        fig.text(
            0.035,
            0.083,
            "Retrospective grouped development; experiments inactive. Anomaly labels do not confirm physical equipment failure.",
            fontsize=10,
            color="#455264",
        )
        fig.text(
            0.035,
            0.039,
            f"Recorded run: {result['run_id']}",
            fontsize=8.5,
            color="#667887",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, metadata={"Title": "FORGE measured model comparison"})
        plt.close(fig)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Measured modern comparison results.json")
    parser.add_argument("--output", type=Path, required=True, help="Explicit chart output path")
    arguments = parser.parse_args()
    print(render(arguments.results, arguments.output))
