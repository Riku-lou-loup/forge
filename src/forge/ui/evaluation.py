"""Read saved evaluation summaries without running an experiment."""

import json

import plotly.graph_objects as go
import streamlit as st

from forge.config import PROJECT_ROOT
from forge.ui.presentation import chart_style

NAMES = {
    "isolation_forest": "Isolation Forest",
    "current_hgb": "Current gradient boosting",
    "tuned_hgb": "Tuned gradient boosting",
}


def comparison_chart(rows):
    figure = go.Figure()
    for label, key, color in [
        ("Precision", "precision", "#ddba76"),
        ("Recall", "recall", "#91c8ad"),
    ]:
        figure.add_trace(
            go.Bar(
                name=label,
                y=[name for name, _ in rows],
                x=[m[key] for _, m in rows],
                orientation="h",
                marker_color=color,
                text=[f"{m[key]:.1%}" for _, m in rows],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}: %{x:.1%}<extra>" + label + "</extra>",
            )
        )
    chart_style(figure, 320)
    figure.update_layout(barmode="group", bargap=0.32, margin={"l": 12, "r": 44, "t": 38, "b": 30})
    figure.update_xaxes(range=[0, 1.1], tickformat=".0%", dtick=0.25, title=None)
    figure.update_yaxes(autorange="reversed")
    return figure


def render_results():
    st.subheader("Model evaluation")
    choice = st.radio(
        "Result set",
        ["Grouped audit", "Development comparison", "Original benchmark"],
        horizontal=True,
        key="evaluation_scope",
    )
    files = {
        "Grouped audit": "generalization-results.json",
        "Development comparison": "improvement-results.json",
        "Original benchmark": "evaluation-results.json",
    }
    path = PROJECT_ROOT / "docs" / files[choice]
    if not path.exists():
        st.info("This result set has not been saved in this checkout.")
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    if choice == "Grouped audit":
        st.write(
            "Performance changes substantially when whole recording groups are excluded from training."
        )
        st.caption(
            "Five outer folds · three inner folds for selection · 23 recordings in 21 overlap groups"
        )
        rows = [(NAMES[name], entry["pooled"]) for name, entry in result["procedures"].items()]
        st.plotly_chart(comparison_chart(rows), width="stretch")
        current = result["procedures"]["current_hgb"]["pooled"]
        tuned = result["procedures"]["tuned_hgb"]["pooled"]
        st.write(
            f"Tuning raises pooled precision from {current['precision']:.1%} to "
            f"{tuned['precision']:.1%}, while false alert episodes rise from "
            f"{current['false_alarm_onsets']} to {tuned['false_alarm_onsets']}. "
            "The tuned model remains a research artifact. It is not active in Investigate."
        )
        st.caption(
            "These recordings were already inspected during development. New recordings are still needed to assess generalization."
        )
        with st.expander("Error counts and event detection"):
            st.dataframe(
                [
                    {
                        "Procedure": name,
                        "False positive readings": m["fp"],
                        "Missed anomaly readings": m["fn"],
                        "False alert episodes": m["false_alarm_onsets"],
                        "Events detected": f"{m['detected_events']} / {m['events']}",
                    }
                    for name, m in rows
                ],
                hide_index=True,
                width="stretch",
            )
        with st.expander("Precision by recording group"):
            st.dataframe(
                [
                    {
                        "Group": group,
                        "Current precision": f"{m['precision']:.1%}",
                        "Tuned precision": f"{result['procedures']['tuned_hgb']['per_group'][group]['precision']:.1%}",
                        "Current recall": f"{m['recall']:.1%}",
                        "Tuned recall": f"{result['procedures']['tuned_hgb']['per_group'][group]['recall']:.1%}",
                    }
                    for group, m in result["procedures"]["current_hgb"]["per_group"].items()
                ],
                hide_index=True,
                width="stretch",
            )
        report = "generalization-results.md"
    elif choice == "Development comparison":
        st.write(
            "The saved model was selected for fewer false positives, accepting more missed anomalies."
        )
        st.caption(
            "Model and threshold selection used these validation recordings. This is development performance."
        )
        selected = result["selected"]["metrics"]
        rows = [
            ("Original Isolation Forest", result["baseline_validation"]),
            ("Relative gradient boosting", selected),
        ]
        st.plotly_chart(comparison_chart(rows), width="stretch")
        st.write(
            f"The selected model misses {selected['fn']:,} anomalous readings and produces "
            f"{selected['false_alarm_onsets']} false alert episodes. Its first 60 readings "
            "per recording establish a reference and receive no assessment."
        )
        with st.expander("Full comparison and startup coverage"):
            st.dataframe(
                [
                    {
                        "Model": name,
                        "False positive readings": m["fp"],
                        "Missed anomaly readings": m["fn"],
                        "False alert episodes": m["false_alarm_onsets"],
                    }
                    for name, m in rows
                ],
                hide_index=True,
                width="stretch",
            )
            st.write(
                f"{selected['unavailable_rows']} readings were unscored. Available-normal FPR: "
                f"{selected['fpr_available_normal']:.2%}. False onsets per available normal hour: "
                f"{selected['false_onsets_per_available_normal_hour']:.2f}. "
                "The targets of 1% FPR and two onsets per hour remain unmet."
            )
        report = "improvement-results.md"
    else:
        st.write(
            "The original Isolation Forest was evaluated once on the historical test partition."
        )
        m = result["test"]["metrics"]["pooled"]
        columns = st.columns(3)
        columns[0].metric("Point F1", f"{m['f1']:.3f}")
        columns[1].metric("Events detected", f"{m['detected_events']} / {m['events']}")
        columns[2].metric("False onsets / normal hour", f"{m['false_alarms_per_normal_hour']:.1f}")
        st.write(
            "This test partition has since been inspected. It differs from the development and audit allocations, so the scores are not a direct comparison."
        )
        with st.expander("Historical results by group"):
            st.dataframe(
                [
                    {
                        "Group": name,
                        "F1": m["f1"],
                        "Precision": m["precision"],
                        "Recall": m["recall"],
                    }
                    for name, m in result["test"]["metrics"]["per_group"].items()
                ],
                hide_index=True,
                width="stretch",
            )
        report = "evaluation.md"
    with st.expander("How to read these results"):
        st.write(
            "Precision is the share of alerted readings annotated anomalous. Recall is the share of anomalous readings that trigger an alert. A false alert episode may contain several consecutive readings."
        )
        st.write(
            "SKAB labels describe laboratory anomalies. They do not establish a physical failure or its cause. Software tests check implementation behavior; they do not prove generalization."
        )
        st.caption(f"Saved run: {result['run_id']}")
        st.caption(f"Report and reproduction: docs/{report}")
