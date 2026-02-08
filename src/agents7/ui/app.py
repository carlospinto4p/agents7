"""Agent Builder UI — Streamlit entry point."""

from __future__ import annotations

import importlib


def main():
    """Launch the multi-page Streamlit app."""
    import streamlit as st

    st.set_page_config(
        page_title="agents7 — Agent Builder",
        page_icon="🛠️",
        layout="wide",
    )

    # Initialise session state
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "state_dict" not in st.session_state:
        st.session_state.state_dict = {}
    if "test_history" not in st.session_state:
        st.session_state.test_history = []
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = None

    _PAGES = [
        ("Home", _home_page),
        ("1. Define", "agents7.ui.pages.1_Define"),
        ("2. Configure", "agents7.ui.pages.2_Configure"),
        ("3. Test", "agents7.ui.pages.3_Test"),
        ("4. Evaluate", "agents7.ui.pages.4_Evaluate"),
        ("5. Export", "agents7.ui.pages.5_Export"),
    ]

    # Sidebar navigation
    st.sidebar.title("Agent Builder")
    labels = [p[0] for p in _PAGES]
    choice = st.sidebar.radio(
        "Step",
        labels,
        index=st.session_state.step,
    )
    st.session_state.step = labels.index(choice)

    page = _PAGES[st.session_state.step]
    if callable(page[1]):
        page[1]()
    else:
        mod = importlib.import_module(page[1])
        mod.render()


def _home_page():
    import streamlit as st

    st.title("Agent Builder")
    st.markdown(
        """
Welcome to the **agents7 Agent Builder**.

This wizard guides you through creating, testing, and
exporting an LLM agent in 5 steps:

| Step | Description |
|------|-------------|
| **1. Define** | Choose agent type and define the output model |
| **2. Configure** | Select LLM model, write the prompt, add examples |
| **3. Test** | Quick-test with individual or batch inputs |
| **4. Evaluate** | Run against labeled data and see accuracy metrics |
| **5. Export** | Save generated code, config, and eval data |

Use the sidebar to navigate between steps.
"""
    )


if __name__ == "__main__":
    main()
