"""Recording explorer, project roadmap, and local environment information."""

import streamlit as st

from forge.cli import environment_report
from forge.config import PROJECT_ROOT
from forge.ui.recording import render_recording

st.set_page_config(page_title="FORGE | Equipment investigation", page_icon="⚙", layout="wide")

with st.sidebar:
    st.title("FORGE")
    st.caption("Equipment investigation workspace")
    st.divider()
    st.markdown("**Current stage**  \nData exploration")
    st.markdown("**First domain**  \nRecorded pump measurements")
    st.caption("Local development · no model calls")

st.caption("FORGE / RECORDING EXPLORER")
st.title("From a machine alert to evidence you can inspect.")
st.write(
    "Explore recorded equipment measurements and their source annotations. "
    "Anomaly detection, document retrieval, and investigation reports are planned."
)
st.info(
    "This development version explores a source-tracked recording. No detector has been trained "
    "and no investigation report has been generated."
)

recording, overview, environment, data_notes = st.tabs(
    ["Explore recording", "Roadmap", "Environment", "Data notes"]
)
with recording:
    render_recording()
with overview:
    columns = st.columns(3)
    for column, title, detail in zip(
        columns,
        ["01 / Observe", "02 / Investigate", "03 / Review"],
        [
            "Load measurements and evaluate unusual behavior against a baseline.",
            "Retrieve relevant documentation and coordinate a bounded evidence workflow.",
            "Inspect citations, review proposed checks, and export an incident record.",
        ],
        strict=True,
    ):
        with column:
            with st.container(border=True):
                st.subheader(title)
                st.write(detail)
                st.caption("Planned milestone")
    st.subheader("Implementation milestones")
    st.dataframe(
        [
            {"Component": "Recording validation and exploration", "Status": "Implemented"},
            {"Component": "Experiment inventory and evaluation split", "Status": "Planned"},
            {"Component": "Statistical baseline and learned detector", "Status": "Planned"},
            {"Component": "Document retrieval with citations", "Status": "Planned"},
            {"Component": "Bounded investigation workflow", "Status": "Planned"},
            {"Component": "Report review and export", "Status": "Planned"},
            {"Component": "Held-out evaluation and failure checks", "Status": "Planned"},
        ],
        hide_index=True,
        width="stretch",
    )
with environment:
    report = environment_report()
    st.code(str(PROJECT_ROOT), language=None)
    if all(report["packages"].values()) and report["isolated_environment"]:
        st.success("The project interpreter and all foundation libraries are available.")
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
        "LLM configuration detected" if report["model_configured"] else "LLM integration pending"
    )
    st.caption("This page checks configuration only. It never calls a model provider.")
with data_notes:
    st.subheader("Reading the recording")
    st.markdown(
        "1. Each row contains measurements at one recorded timestamp.\n"
        "2. Eight sensor columns contain the candidate model inputs.\n"
        "3. Anomaly and change-point annotations come from the source dataset.\n"
        "4. Sampling gaps are preserved; source labels are not model predictions."
    )
    st.write("See `data/README.md` for provenance and `docs/roadmap.md` for evaluation plans.")
    st.code(r".\.venv\Scripts\python.exe -m forge doctor", language="powershell")
    st.caption("The recording explorer needs neither an API key nor purchased hardware.")
