"""Inspect real model scores, evidence, and the report review boundary."""

import json

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


def render_investigation():
    st.subheader("Investigate a recording")
    st.caption("Local policy agents · project-authored evidence · no LLM calls")
    try:
        detector, metadata, _ = load_model()
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
    st.caption(
        "Training and validation recordings only. The test set is reserved for explicit evaluation."
    )
    try:
        frame, record = load_development_recording(selection)
        scores = detector.score(frame)
        alerts = causal_alerts(scores, frame.datetime, metadata["threshold"], **metadata["policy"])
    except (ValueError, OSError) as error:
        st.error(str(error))
        return
    left, middle, right = st.columns(3)
    left.metric(
        "Detector", "Isolation Forest" if detector.forest is not None else "Robust deviation"
    )
    middle.metric("Alerted observations", f"{int(alerts.sum()):,} / {len(frame):,}")
    right.metric("Frozen threshold", f"{metadata['threshold']:.4f}")
    st.caption(
        "An alert requires three consecutive readings above the threshold. Scores are not failure probabilities."
    )
    figure = go.Figure(
        go.Scatter(
            x=frame.datetime,
            y=scores,
            mode="lines",
            name="Anomaly score",
            line={"color": "#548bff", "width": 1.3},
        )
    )
    figure.add_hline(y=metadata["threshold"], line_dash="dash", annotation_text="Frozen threshold")
    figure.add_trace(
        go.Scatter(
            x=frame.datetime[alerts],
            y=scores[alerts],
            mode="markers",
            name="Persistent alert",
            marker={"color": "#e97b45", "size": 4},
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
                y=scores[annotated],
                mode="markers",
                name="Source anomaly annotation",
                marker={"symbol": "circle-open", "color": "#aa74cf", "size": 7},
            )
        )
    figure.update_layout(
        height=360,
        margin={"l": 15, "r": 15, "t": 30, "b": 20},
        yaxis_title="Anomaly score",
        xaxis_title="Source time (timezone unspecified)",
        legend={"orientation": "h"},
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Investigation selects the highest-scoring persistent alert episode. It does not use source labels to choose evidence."
    )
    identity = (metadata["run_id"], selection, record["sha256"])
    if st.session_state.get("report_identity") != identity:
        st.session_state.pop("incident_report", None)
        st.session_state["report_identity"] = identity
        st.session_state["review_confirmed"] = False
    if st.button("Investigate alert", type="primary", key="investigate"):
        with st.spinner("Scoring, retrieving evidence, and checking citations…"):
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
    st.subheader("Investigation record")
    st.write(report.summary)
    st.caption(f"Workflow status: {report.status} · Review: {report.review_status}")
    if report.observation.get("sensor_deviations"):
        st.write(
            f"Selected alert: **{report.observation['alert_start']} → {report.observation['alert_end']}**"
        )
        st.dataframe(report.observation["sensor_deviations"], hide_index=True, width="stretch")
        st.caption(
            "Largest deviations from the pooled normal reference at the peak. These are not causal explanations or forest feature attributions."
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


def render_results():
    st.subheader("Frozen benchmark")
    path = PROJECT_ROOT / "docs/evaluation-results.json"
    if not path.exists():
        st.info("No benchmark summary has been published in this checkout.")
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = result["test"]["metrics"]["pooled"]
    st.caption(f"Saved benchmark run {result['run_id']} · {result['split_id']}")
    columns = st.columns(4)
    for column, label, value in zip(
        columns,
        ["Point F1", "Precision", "Recall", "Events detected"],
        [
            f"{metrics['f1']:.3f}",
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['detected_events']} / {metrics['events']}",
        ],
        strict=True,
    ):
        column.metric(label, value)
    st.warning(
        f"Research baseline: {metrics['false_alarms_per_normal_hour']:.1f} false alert onsets per normal hour on this test set. This operating point is unsuitable for unattended deployment."
    )
    st.write(
        "Selection used validation F1. The statistical baseline had better ranking quality and fewer false alert onsets on validation; the learned detector's F1 advantage does not make it operationally superior."
    )
    st.dataframe(
        [
            {
                "Candidate": c["name"],
                "Validation F1": c["pooled"]["f1"],
                "Validation AP": c["pooled"]["average_precision"],
                "False onsets / normal hour": c["pooled"]["false_alarms_per_normal_hour"],
            }
            for c in result["validation_candidates"]
        ],
        hide_index=True,
        width="stretch",
    )
    st.write(
        "Test observations were deduplicated within overlap groups. Conflicting anomaly annotations were excluded and broke evaluation continuity. See `docs/evaluation.md` for the full protocol and limitations."
    )
    with st.expander("Per-group held-out results"):
        st.dataframe(
            [
                {
                    "Group": name,
                    "Rows": m["rows"],
                    "F1": m["f1"],
                    "Average precision": m["average_precision"],
                    "Detected events": m["detected_events"],
                    "Events": m["events"],
                }
                for name, m in result["test"]["metrics"]["per_group"].items()
            ],
            hide_index=True,
            width="stretch",
        )
