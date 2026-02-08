"""Agent Builder UI — optional Streamlit-based interface."""

import sys


def launch(host: str = "localhost", port: int = 8501):
    """Launch the Agent Builder Streamlit app.

    Args:
        host: Server address to bind to (default: ``"localhost"``).
        port: Server port (default: ``8501``).
    """
    try:
        import streamlit  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The UI requires extra dependencies. "
            "Install them with: pip install agents7[ui]"
        ) from e

    sys.argv = [
        "agents7-ui",
        "--host",
        host,
        "--port",
        str(port),
    ]

    from agents7.ui.app import main

    main()
