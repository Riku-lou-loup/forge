"""FORGE's local recording investigation and benchmark workspace."""

import streamlit as st

from forge.cli import environment_report
from forge.config import PROJECT_ROOT
from forge.ui.investigation import render_investigation, render_results
from forge.ui.recording import render_recording

st.set_page_config(page_title="FORGE | Equipment investigation", page_icon="⚙", layout="wide")
with st.sidebar:
    st.title("FORGE")
    st.caption("Equipment investigation workspace")
    st.divider()
    st.markdown("**Current stage**  \nEvaluated ML baseline and evidence workflow")
    st.markdown("**First domain**  \nRecorded pump measurements")
    st.caption("Local analysis · human review · no model API calls")

st.caption("FORGE / EQUIPMENT INVESTIGATION")
st.title("From a machine alert to evidence you can inspect.")
st.write(
    "Score real pump measurements, inspect a persistent alert, and retrieve analytical guidance with citations. Review the findings before exporting an investigation record."
)
investigation, recording, results, environment, data_notes = st.tabs(
    ["Investigate", "Explore recording", "Evaluation", "Environment", "Data notes"]
)
with investigation:
    render_investigation()
with recording:
    render_recording()
with results:
    render_results()
with environment:
    report = environment_report()
    st.code(str(PROJECT_ROOT), language=None)
    if all(report["packages"].values()) and report["isolated_environment"]:
        st.success("The project interpreter and required libraries are available.")
    else:
        st.error("Run the setup instructions in README.md to repair the environment.")
    st.dataframe(
        [
            {"Library": name, "Version": value or "Missing"}
            for name, value in report["packages"].items()
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(f"Python {report['python']} · {report['interpreter']}")
    st.write(
        "Trained artifact available"
        if report["trained_artifact_present"]
        else "Run the training command to create the local artifact."
    )
    st.caption(
        "The implemented workflow uses local policy agents and extractive drafting. Provider configuration does not enable language-model calls."
    )
with data_notes:
    st.subheader("Reading the evidence")
    st.markdown(
        "1. Eight sensor columns are model inputs; timestamps preserve ordering and gaps.\n2. Anomaly and change-point annotations are source labels, excluded from fitting and investigation evidence.\n3. Training uses normal rows only; validation selects settings; the held-out test measures the frozen detector.\n4. Retrieved notes are project-authored analytical guidance, not manufacturer instructions.\n5. This laboratory benchmark does not establish fault diagnosis, reliability on another machine, or saved downtime."
    )
    st.write(
        "See `docs/architecture.md`, `docs/evaluation.md`, and `data/README.md` for implementation and provenance."
    )
    st.code(
        r".\.venv\Scripts\python.exe -m forge investigate --recording valve1/1 --export",
        language="powershell",
    )
