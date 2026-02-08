"""Agent Builder UI — optional Streamlit-based interface."""


def launch():
    """Launch the Agent Builder Streamlit app."""
    try:
        import streamlit  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The UI requires extra dependencies. "
            "Install them with: pip install agents7[ui]"
        ) from e

    from agents7.ui.app import main

    main()
