"""Agent Builder UI — Streamlit entry point."""

import argparse
import importlib
from collections import deque


def main() -> None:
    """Launch the multi-page Streamlit app.

    When invoked from the console entry point, parses `--host`
    and `--port` arguments and re-launches Streamlit with the
    corresponding server flags.  When already running inside a
    Streamlit process, renders the app directly.
    """
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is not None:
        _render_app()
        return

    parser = argparse.ArgumentParser(
        description="agents7 Agent Builder UI",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Server address (default: localhost).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Server port (default: 8501).",
    )
    args = parser.parse_args()

    from streamlit.web.cli import main_run

    main_run(
        [
            __file__,
            "--server.address",
            args.host,
            "--server.port",
            str(args.port),
        ]
    )


def _render_app() -> None:
    """Render the multi-page Streamlit application."""
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
        st.session_state.test_history = deque(maxlen=50)
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = None
    if "prompt_versions" not in st.session_state:
        st.session_state.prompt_versions = []

    _PAGES = [
        ("Home", _home_page),
        ("1. Define", "agents7.ui._pages.1_Define"),
        ("2. Configure", "agents7.ui._pages.2_Configure"),
        ("3. Test", "agents7.ui._pages.3_Test"),
        ("4. Evaluate", "agents7.ui._pages.4_Evaluate"),
        ("5. Export", "agents7.ui._pages.5_Export"),
    ]

    # Sidebar navigation
    st.sidebar.title("Agent Builder")
    labels = [p[0] for p in _PAGES]
    choice = st.sidebar.radio(
        "Step",
        labels,
        index=st.session_state.step,
        key="nav_step",
    )
    st.session_state.step = labels.index(choice)

    page = _PAGES[st.session_state.step]
    if callable(page[1]):
        page[1]()
    else:
        mod = importlib.import_module(page[1])
        mod.render()


def _home_page() -> None:
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
