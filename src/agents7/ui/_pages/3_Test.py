"""Step 3 — Quick Test (single and batch)."""

from __future__ import annotations


def render():
    import streamlit as st

    from agents7.ui.helpers import build_agent_from_state
    from agents7.ui._pages import _state_from_dict

    st.header("3. Quick Test")

    sd = st.session_state.state_dict
    state = _state_from_dict(sd)

    # --- Single test ---
    st.subheader("Single Input")
    test_input = st.text_area("Input (plain text or JSON dict)", height=80)

    if st.button("Run"):
        if not test_input.strip():
            st.warning("Please enter some input.")
        else:
            input_data = _parse_input(test_input)
            try:
                agent = build_agent_from_state(state)
                with st.spinner("Running agent..."):
                    response = agent.act(input_data)
                _show_response(response)
                st.session_state.test_history.append(
                    {
                        "input": test_input,
                        "output": (
                            response.model.model_dump()
                            if response.model
                            else None
                        ),
                        "metrics": response.metrics.model_dump(),
                    }
                )
            except Exception as e:
                st.error(f"Error: {e}")

    # --- Batch test ---
    st.subheader("Batch Test")
    st.caption("One input per line (plain text) or a JSON array.")
    batch_input = st.text_area("Batch inputs", height=150, key="batch_input")

    if st.button("Run Batch"):
        inputs = _parse_batch(batch_input)
        if not inputs:
            st.warning("No valid inputs found.")
        else:
            try:
                agent = build_agent_from_state(state)
                progress = st.progress(0)
                results = []
                for i, inp in enumerate(inputs):
                    resp = agent.act(inp)
                    results.append(resp)
                    progress.progress((i + 1) / len(inputs))
                for resp in results:
                    _show_response(resp)
                    st.divider()
            except Exception as e:
                st.error(f"Error: {e}")

    # --- History ---
    if st.session_state.test_history:
        st.subheader("Test History")
        for i, entry in enumerate(reversed(st.session_state.test_history)):
            with st.expander(
                f"Test {len(st.session_state.test_history) - i}"
            ):
                st.json(entry)


def _parse_input(text: str):
    """Try to parse as JSON dict, fall back to string."""
    import json

    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return {"text": text}


def _parse_batch(text: str) -> list:
    """Parse batch input: JSON array or one-per-line."""
    import json

    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            items = json.loads(text)
            return [
                item if isinstance(item, dict) else {"text": str(item)}
                for item in items
            ]
        except json.JSONDecodeError:
            pass
    return [
        {"text": line.strip()} for line in text.splitlines() if line.strip()
    ]


def _show_response(response):
    import streamlit as st

    if response.model is not None:
        st.json(response.model.model_dump())
    else:
        st.warning("Model returned None")
        st.text(response.response)

    cols = st.columns(3)
    m = response.metrics
    cols[0].metric("Tokens", m.tokens)
    cols[1].metric("Prompt Tokens", m.prompt_tokens)
    cols[2].metric("Duration (s)", f"{m.total_duration:.2f}")
