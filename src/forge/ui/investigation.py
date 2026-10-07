"""Inspect real model scores, evidence, and the report review boundary."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from forge.agents.service import available_recordings, load_development_recording
from forge.agents.workflow import investigate
from forge.config import PROJECT_ROOT
from forge.data.partitions import recording_path
from forge.ml.metrics import causal_alerts
from forge.ml.training import load_model
from forge.rag.retrieval import Retriever
from forge.reports.incident import markdown
from forge.ui.presentation import chart_style


def render_investigation():
    st.subheader("Anomaly scores")
    try:
        detector, metadata, _ = load_model(active=True)
    except (OSError, ValueError, KeyError) as error:
        st.info("Train the local detector to enable investigations.")
        st.code(r".\.venv\Scripts\python.exe -m forge train", language="powershell")
        with st.expander("Artifact details"):
            st.text(str(error))
        return
    records = [r for r in available_recordings() if recording_path(r).exists()]
    if not records:
        st.info("Download the training and validation recordings using the README instructions.")
        return
    ids = [r["experiment_id"] for r in records]
    selection = st.selectbox(
        "Recording",
        ids,
        index=ids.index("valve1/1") if "valve1/1" in ids else 0,
        key="investigation_recording",
    )
    st.caption("Choose from downloaded training and validation recordings.")
    try:
        frame, record = load_development_recording(selection)
        scores = detector.score(frame)
        ready = detector.readiness(frame)
        displayed_scores = np.where(ready, scores, np.nan)
        alerts = causal_alerts(
            displayed_scores, frame.datetime, metadata["threshold"], **metadata["policy"]
        )
    except (ValueError, OSError) as error:
        st.error(str(error))
        return
    left, middle, right = st.columns(3)
    family = detector.describe()["family"]
    label = {
        "gradient_boosting": "Gradient boosting",
        "HistGradientBoostingClassifier": "Gradient boosting",
        "IsolationForest": "Isolation Forest",
        "isolation_forest": "Isolation Forest",
        "robust_max": "Robust deviation",
    }.get(family, family.replace("_", " ").title())
    left.metric("Detector", label)
    middle.metric("Alerted readings", f"{int(alerts.sum()):,} / {len(frame):,}")
    right.metric("Alert threshold", f"{metadata['threshold']:.4f}")
    st.caption(
        f"An alert requires {metadata['policy']['persistence']} consecutive readings above the threshold. Scores are not failure probabilities."
    )
    if (~ready).any():
        st.caption(
            f"The first {int((~ready).sum())} readings establish the reference and are unscored. A fault present at startup may be missed."
        )
    figure = go.Figure(
        go.Scatter(
            x=frame.datetime,
            y=displayed_scores,
            mode="lines",
            name="Anomaly score",
            line={"color": "#ddba76", "width": 1.3},
        )
    )
    figure.add_hline(
        y=metadata["threshold"],
        line_dash="dash",
        line_color="#a2b3a8",
        annotation_text="Alert threshold",
    )
    figure.add_trace(
        go.Scatter(
            x=frame.datetime[alerts],
            y=scores[alerts],
            mode="markers",
            name="Persistent alert",
            marker={"color": "#ef937b", "size": 4},
        )
    )
    show_labels = st.checkbox(
        "Overlay source annotations for comparison", key="investigation_labels"
    )
    if show_labels and "anomaly" in frame:
        annotated = frame.anomaly.eq(1)
        figure.add_trace(
            go.Scatter(
                x=frame.datetime[annotated],
                y=displayed_scores[annotated],
                mode="markers",
                name="Source anomaly annotation",
                marker={"symbol": "circle-open", "color": "#91c8ad", "size": 7},
            )
        )
    figure.update_layout(
        height=360,
        margin={"l": 15, "r": 15, "t": 30, "b": 20},
        yaxis_title="Anomaly score",
        xaxis_title="Source time (timezone unspecified)",
        legend={"orientation": "h"},
    )
    chart_style(figure, 340)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Investigate alert uses the highest-scoring persistent episode. Source labels are excluded from this choice."
    )
    identity = (metadata["run_id"], selection, record["sha256"])
    if st.session_state.get("report_identity") != identity:
        st.session_state.pop("incident_report", None)
        st.session_state["report_identity"] = identity
        st.session_state["review_confirmed"] = False
    if st.button("Investigate alert", type="primary", key="investigate"):
        with st.spinner("Building the investigation…"):
            try:
                path = PROJECT_ROOT / "knowledge/playbook.json"
                retriever = Retriever(path) if path.exists() else None
                st.session_state["incident_report"] = investigate(
                    frame, detector, metadata, selection, record["sha256"], retriever=retriever
                )
                st.session_state["review_confirmed"] = False
            except (OSError, ValueError, KeyError) as error:
                st.error(f"Investigation could not complete: {error}")
    report = st.session_state.get("incident_report")
    if report is None:
        return
    st.divider()
    st.subheader("Investigation report")
    st.write(report.summary)
    st.caption(
        f"{report.status.replace('_', ' ').capitalize()} · {report.review_status.capitalize()}"
    )
    if report.observation.get("sensor_deviations"):
        st.write(
            f"Selected alert: **{report.observation['alert_start']} → {report.observation['alert_end']}**"
        )
        st.dataframe(report.observation["sensor_deviations"], hide_index=True, width="stretch")
        st.caption(
            "Largest deviations from the pooled normal reference at the peak. These are not causal explanations or model feature attributions."
        )
    for check in report.suggested_checks:
        st.write(f"• {check.text} [{check.citation}]")
    with st.expander("Evidence and citations", expanded=bool(report.evidence)):
        for evidence in report.evidence:
            st.markdown(f"**{evidence.citation} · {evidence.title}**")
            st.text(evidence.excerpt)
            st.caption(
                f"{evidence.authorship} · {evidence.source} · relevance {evidence.relevance:.3f}"
            )
        if not report.evidence:
            st.write("No supported guidance was added.")
    with st.expander("Workflow trace and limitations"):
        st.dataframe(report.trace, hide_index=True, width="stretch")
        for limitation in report.limitations:
            st.write(f"• {limitation}")
        st.caption(f"Model run: {report.model_run}")
    if report.status == "needs_review" and report.review_status == "unreviewed":
        reviewer = st.text_input("Reviewer name", key="reviewer_name", max_chars=100)
        confirmed = st.checkbox(
            "I reviewed the observations, citations, and limitations.", key="review_confirmed"
        )
        if st.button(
            "Mark reviewed", disabled=not (confirmed and reviewer.strip()), key="mark_reviewed"
        ):
            report = report.reviewed(reviewer)
            st.session_state["incident_report"] = report
            st.success("Review recorded. This does not authorize an equipment action.")
    elif report.review_status == "reviewed":
        st.success(f"Reviewed by {report.reviewer}.")
    suffix = "reviewed" if report.review_status == "reviewed" else "draft"
    first, second = st.columns(2)
    first.download_button(
        f"Download {suffix} · Markdown",
        markdown(report),
        file_name=f"forge-{report.report_id}-{suffix}.md",
        mime="text/markdown",
        key="download_markdown",
    )
    second.download_button(
        f"Download {suffix} · JSON",
        report.model_dump_json(indent=2),
        file_name=f"forge-{report.report_id}-{suffix}.json",
        mime="application/json",
        key="download_json",
    )
