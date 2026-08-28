"""Exercise imports and the real Streamlit script without any API requests."""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from langchain_openai import ChatOpenAI  # noqa: F401
from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader  # noqa: F401
from sklearn.ensemble import IsolationForest
from streamlit.testing.v1 import AppTest

from forge.config import PROJECT_ROOT

frame = pd.DataFrame({"value": np.arange(10, dtype=float)})
model = IsolationForest(random_state=42).fit(frame)
assert model.predict(frame).shape == (10,)
assert len(go.Figure(go.Scatter(y=frame["value"])).data) == 1
graph = StateGraph(dict)
graph.add_node("check", lambda state: {"ready": True})
graph.add_edge(START, "check")
graph.add_edge("check", END)
with patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}):
    assert graph.compile().invoke({})["ready"]
app = AppTest.from_file(str(PROJECT_ROOT / "src/forge/ui/app.py")).run(timeout=45)
assert not app.exception, str(app.exception)
assert len(app.tabs) == 4
assert any("evidence" in heading.value for heading in app.title)
print("PASS: numerical stack, ML import, chart, LangGraph execution, and Streamlit rendering.")
print("No model API called. This smoke check does not evaluate the future product.")
