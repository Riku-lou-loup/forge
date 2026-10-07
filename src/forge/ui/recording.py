"""Interactive exploration of measurements and source-provided labels."""

import plotly.graph_objects as go
import streamlit as st

from forge.data.sample import SENSOR_UNITS, audit_recording, load_sample, sample_manifest
from forge.ui.presentation import chart_style


def render_recording() -> None:
    st.subheader("Sensor explorer")
    st.write(
        "Each row contains measurements at one recorded time. Choose a sensor and compare "
        "its signal with the experiment's annotation."
    )
    try:
        frame = load_sample()
        manifest = sample_manifest()
    except FileNotFoundError:
        st.info("Fetch the sample once, then refresh this page:")
        st.code(
            r".\.venv\Scripts\python.exe -m forge.data.sample --download",
            language="powershell",
        )
        return
    except ValueError as error:
        st.error(f"Recording could not be loaded: {error}")
        return

    audit = audit_recording(frame)
    columns = st.columns(3)
    columns[0].metric("Recorded rows", f"{audit['rows']:,}")
    columns[1].metric("Sensor measurements", len(SENSOR_UNITS))
    columns[2].metric("Rows labeled anomalous", audit["anomalous_rows"])
    st.caption(
        f"SKAB / {manifest['experiment_id']} · source time zone unspecified · "
        "development sample, excluded from the held-out test set"
    )
    st.caption(
        "The sample spans 20 minutes with 56 two-second gaps. Lines connect recorded "
        "points; no missing measurements have been filled in."
    )
    sensors = list(SENSOR_UNITS)
    sensor = st.selectbox("Sensor to explore", sensors, index=sensors.index("Volume Flow RateRMS"))
    annotated = frame[frame["anomaly"] == 1]
    chart = go.Figure()
    chart.add_scatter(
        x=frame["datetime"],
        y=frame[sensor],
        mode="lines",
        name="Measured signal",
        line={"color": "#DDBA76", "width": 1.5},
    )
    chart.add_scatter(
        x=annotated["datetime"],
        y=annotated[sensor],
        mode="markers",
        name="Source label: anomaly = 1",
        marker={"color": "#EF846A", "size": 4},
    )
    chart.update_layout(
        height=420,
        xaxis_title="Recorded time",
        yaxis_title=f"{sensor} ({SENSOR_UNITS[sensor]})",
        legend={"orientation": "h", "y": 1.12},
        margin={"t": 45, "b": 35},
    )
    chart_style(chart, 390)
    st.plotly_chart(chart, width="stretch")
    st.caption(
        "The colored points show labels supplied with the dataset. They are not model "
        "predictions. Model scores are available in the Investigate tab."
    )
    with st.expander("See the rows and data-quality audit"):
        st.dataframe(frame.head(10), hide_index=True, width="stretch")
        st.json(audit)
    with st.expander("Source and reproducibility"):
        st.markdown(f"[Experiment documentation]({manifest['documentation_url']})")
        st.markdown(f"[Repository license]({manifest['license_url']})")
        st.write(manifest["experiment_description"])
        st.caption("Source: Iurii D. Katser and Vyacheslav O. Kozitsin, SKAB.")
        st.code(f"Revision: {manifest['revision']}\nSHA-256: {manifest['sha256']}")
    st.markdown(
        "Compare flow, pressure, and vibration around the annotated regions. "
        "These observations describe signal changes; they do not establish a fault cause."
    )
