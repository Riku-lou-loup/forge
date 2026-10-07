"""FORGE's local recording investigation and benchmark workspace."""

import streamlit as st

from forge.cli import environment_report
from forge.config import PROJECT_ROOT
from forge.ui.evaluation import render_results
from forge.ui.investigation import render_investigation
from forge.ui.motion import render_workflow
from forge.ui.presentation import apply_style
from forge.ui.recording import render_recording

st.set_page_config(page_title="FORGE | Pump investigation", layout="wide")
apply_style()
with st.sidebar:
    st.title("FORGE")
    st.caption("Pump anomaly investigation")
    st.divider()
    st.write("Run investigations locally.")
    st.caption("Inspect an alert, read the supporting passages, and export a report for review.")
    show_workflow = st.toggle("Show workflow animation", value=True)
    st.divider()
    st.markdown("[Project repository](https://github.com/Riku-lou-loup/forge)")
    st.caption("Python / scikit-learn / LangGraph")
    st.html(
        '<p class="forge-note">Research prototype using SKAB laboratory recordings. Equipment actions remain outside this application.</p>'
    )

st.html('<div class="forge-rule" aria-hidden="true"></div>')
st.title("Pump measurements and evidence")
st.write("Choose a recording to inspect its signal and investigate an alert.")
if show_workflow:
    render_workflow()

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
    st.subheader("Local environment")
    report = environment_report()
    if all(report["packages"].values()) and report["isolated_environment"]:
        st.success("The project environment is ready.")
    else:
        st.error("Run the setup commands in README.md to repair the environment.")
    st.dataframe(
        [
            {"Library": name, "Version": value or "Missing"}
            for name, value in report["packages"].items()
        ],
        hide_index=True,
        width="stretch",
    )
    with st.expander("Paths and interpreter"):
        st.code(str(PROJECT_ROOT), language=None)
        st.caption(f"Python {report['python']} · {report['interpreter']}")
    st.caption("Investigations run locally. Provider settings do not enable LLM calls.")
with data_notes:
    st.subheader("About these recordings")
    st.write(
        "SKAB records pump experiments across eight sensor channels. Timestamps preserve the order and gaps in the measurements. Source labels can be overlaid on a chart for comparison."
    )
    with st.expander("Models and startup", expanded=True):
        st.write(
            "The active gradient-boosting model learns from normal and anomalous training examples. Its first 60 readings establish a reference and receive no assessment. A fault present throughout startup may be missed."
        )
        st.write(
            "Isolation Forest remains the original baseline. It was fitted on normal readings only. The grouped audit refits both approaches on different recording groups; the tuned research artifact has not been activated."
        )
    with st.expander("Labels, guidance and limits"):
        st.write(
            "Anomaly annotations supply training targets and evaluation labels. They are excluded from model inputs and investigation evidence. An annotated anomaly does not establish a physical fault."
        )
        st.write(
            "Retrieved passages are project-authored analytical notes. They are not manufacturer instructions. The workflow copies supported checks and verifies their citations before human review."
        )
    st.caption(
        "Sources and reproduction: data/README.md, docs/architecture.md, docs/generalization-results.md."
    )
