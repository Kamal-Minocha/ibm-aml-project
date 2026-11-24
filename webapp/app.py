"""
AML risk-scoring demo: Streamlit front end for the project's final XGBoost
model. This file only sets up the page and navigation; each page lives in
``views/``, shared UI pieces in ``ui.py`` and all model logic in ``scoring.py``.

Run from the project root:
    .venv\\Scripts\\python.exe -m streamlit run webapp/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ui  # noqa: E402

st.set_page_config(
    page_title="AML Risk Scoring",
    page_icon=":material/policy:",
    layout="wide",
)
ui.inject_css()

navigation = st.navigation(
    [
        st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True),
        st.Page("views/sample_explorer.py", title="Sample Explorer", icon=":material/search:"),
        st.Page("views/batch_scoring.py", title="Batch Scoring", icon=":material/table_view:"),
        st.Page("views/playground.py", title="Feature Playground", icon=":material/tune:"),
    ],
    position="top",
)
navigation.run()
ui.footer()
