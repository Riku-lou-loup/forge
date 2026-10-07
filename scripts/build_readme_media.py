"""Build README figures from saved results and the pinned development sample."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forge.config import PROJECT_ROOT
from forge.data.sample import load_sample, sample_manifest

OUT = PROJECT_ROOT / "docs/assets"
OUT.mkdir(exist_ok=True)
BG, PANEL, TEXT, MUTED, GOLD, GREEN, CORAL, LINE = (
    "#101816",
    "#192521",
    "#eff3eb",
    "#b2c1b6",
    "#ddba76",
    "#91c8ad",
    "#ef937b",
    "#42564b",
)
plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "text.color": TEXT,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": TEXT,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": LINE,
        "savefig.facecolor": BG,
    }
)

result = json.loads((PROJECT_ROOT / "docs/generalization-results.json").read_text(encoding="utf-8"))
names = ["Isolation Forest", "Current gradient boosting", "Tuned gradient boosting"]
entries = [
    result["procedures"][key]["pooled"] for key in ["isolation_forest", "current_hgb", "tuned_hgb"]
]
fig, ax = plt.subplots(figsize=(13.6, 5.3))
for offset, label, key, color in [
    (-0.16, "Precision", "precision", GOLD),
    (0.16, "Recall", "recall", GREEN),
]:
    positions = [i + offset for i in range(3)]
    values = [m[key] * 100 for m in entries]
    ax.barh(positions, values, height=0.24, color=color, label=label)
    for y, value in zip(positions, values):
        ax.text(value + 1.7, y, f"{value:.1f}%", va="center", fontsize=12, color=TEXT)
ax.set_yticks(range(3), names)
ax.invert_yaxis()
ax.set_xlim(0, 110)
ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax.grid(axis="x", color=LINE, alpha=0.35)
ax.set_axisbelow(True)
ax.legend(loc="lower right", bbox_to_anchor=(1, 1.13), ncol=2, frameon=False, labelcolor=TEXT)
fig.text(0.035, 0.93, "Performance across excluded recording groups", fontsize=20, weight="bold")
fig.text(
    0.035,
    0.865,
    "Nested grouped cross-validation / 23 recordings / 21 overlap groups",
    fontsize=11,
    color=MUTED,
)
fig.text(
    0.035,
    0.045,
    "Already-inspected development data. Tuned model not activated. Source: docs/generalization-results.json",
    fontsize=10,
    color=MUTED,
)
fig.subplots_adjust(left=0.25, right=0.94, bottom=0.17, top=0.75)
fig.savefig(OUT / "grouped-audit.png", dpi=150)
plt.close(fig)

frame = load_sample()
manifest = sample_manifest()
elapsed = (frame.datetime - frame.datetime.iloc[0]).dt.total_seconds() / 60
fig, axes = plt.subplots(2, 1, figsize=(13.6, 6), sharex=True)
for ax, sensor, label in zip(
    axes, ["Volume Flow RateRMS", "Pressure"], ["Flow (L/min)", "Pressure (bar)"]
):
    ax.plot(elapsed, frame[sensor], color=GOLD, lw=1.3)
    marked = frame.anomaly.eq(1)
    ax.scatter(elapsed[marked], frame.loc[marked, sensor], s=5, color=CORAL, zorder=3)
    ax.set_ylabel(label)
    ax.grid(axis="y", color=LINE, alpha=0.35)
axes[1].set_xlabel("Elapsed minutes")
fig.text(
    0.055,
    0.935,
    f"Inside a pump recording: {manifest['experiment_id']}",
    fontsize=20,
    weight="bold",
)
fig.text(
    0.055,
    0.878,
    "Source anomaly annotations are shown in coral. These are measured signals, not model predictions.",
    fontsize=11,
    color=MUTED,
)
fig.text(
    0.055,
    0.035,
    "SKAB / Iurii D. Katser and Vyacheslav O. Kozitsin / laboratory measurements / gaps are not interpolated",
    fontsize=10,
    color=MUTED,
)
fig.subplots_adjust(left=0.095, right=0.96, top=0.79, bottom=0.13, hspace=0.33)
fig.savefig(OUT / "recorded-signals.png", dpi=150)
plt.close(fig)


print("Built grouped-audit.png and recorded-signals.png from recorded evidence.")
