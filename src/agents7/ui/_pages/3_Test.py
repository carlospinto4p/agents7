"""Step 3 — Quick Test (single, batch, and A/B comparison)."""

from __future__ import annotations


def render():
    import streamlit as st

    from agents7.ui._pages import _state_from_dict

    st.header("3. Quick Test")

    sd = st.session_state.state_dict
    state = _state_from_dict(sd)

    # --- Version Management ---
    _render_version_management(sd)

    # --- Tabs ---
    tab_single, tab_batch, tab_compare, tab_history = st.tabs(
        ["Single Input", "Batch Test", "A/B Compare", "History"]
    )

    with tab_single:
        _render_single_test(state)

    with tab_batch:
        _render_batch_test(state)

    with tab_compare:
        _render_ab_compare(state)

    with tab_history:
        _render_history()


# ----------------------------------------------------------------
# Version Management
# ----------------------------------------------------------------


def _render_version_management(sd: dict):
    import streamlit as st

    from agents7.ui.comparison.models import PromptVersion

    st.subheader("Prompt Versions")

    col_name, col_btn = st.columns([3, 1])
    with col_name:
        version_name = st.text_input(
            "Version name",
            key="version_name_input",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        save_clicked = st.button("Save Current")

    if save_clicked:
        if not version_name.strip():
            st.warning("Please enter a version name.")
        else:
            names = [v.name for v in st.session_state.prompt_versions]
            if version_name.strip() in names:
                st.warning(
                    f"Version '{version_name.strip()}' already exists."
                )
            else:
                version = PromptVersion(
                    name=version_name.strip(),
                    system_prompt=sd.get("system_prompt", ""),
                    examples=sd.get("examples", []),
                    model_name=sd.get("model_name", "qwen3"),
                )
                st.session_state.prompt_versions.append(version)
                st.success(f"Saved version '{version.name}'.")

    if st.session_state.prompt_versions:
        for i, v in enumerate(st.session_state.prompt_versions):
            col_info, col_load, col_del = st.columns([4, 1, 1])
            with col_info:
                st.text(f"{v.name}  (model: {v.model_name})")
            with col_load:
                if st.button("Load", key=f"load_version_{i}"):
                    sd["system_prompt"] = v.system_prompt
                    sd["examples"] = v.examples
                    sd["model_name"] = v.model_name
                    st.rerun()
            with col_del:
                if st.button("Delete", key=f"del_version_{i}"):
                    st.session_state.prompt_versions.pop(i)
                    st.rerun()

    st.divider()


# ----------------------------------------------------------------
# Single Input Tab
# ----------------------------------------------------------------


def _render_single_test(state):
    import streamlit as st

    from agents7.ui.helpers import build_agent_from_state

    test_input = st.text_area(
        "Input (plain text or JSON dict)",
        height=80,
        key="single_test_input",
    )

    if st.button("Run", key="run_single"):
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
                        "metrics": (response.metrics.model_dump()),
                    }
                )
            except Exception as e:
                st.error(f"Error: {e}")


# ----------------------------------------------------------------
# Batch Test Tab
# ----------------------------------------------------------------


def _render_batch_test(state):
    import streamlit as st

    from agents7.ui.helpers import build_agent_from_state

    st.caption("One input per line (plain text) or a JSON array.")
    batch_input = st.text_area(
        "Batch inputs",
        height=150,
        key="batch_test_input",
    )

    if st.button("Run Batch", key="run_batch"):
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


# ----------------------------------------------------------------
# A/B Compare Tab
# ----------------------------------------------------------------


def _render_ab_compare(state):
    import streamlit as st

    from agents7.ui.comparison.runner import (
        run_comparison,
    )

    versions = st.session_state.prompt_versions
    if len(versions) < 2:
        st.info(
            "Save at least 2 prompt versions above to run an A/B comparison."
        )
        return

    names = [v.name for v in versions]

    col_a, col_b = st.columns(2)
    with col_a:
        sel_a = st.selectbox("Version A", names, key="ab_version_a")
    with col_b:
        sel_b = st.selectbox(
            "Version B",
            names,
            index=min(1, len(names) - 1),
            key="ab_version_b",
        )

    if sel_a == sel_b:
        st.warning("Select two different versions.")
        return

    version_a = versions[names.index(sel_a)]
    version_b = versions[names.index(sel_b)]

    # Prompt previews
    with st.expander("Prompt previews"):
        prev_a, prev_b = st.columns(2)
        with prev_a:
            st.markdown(f"**{version_a.name}**")
            st.text(version_a.system_prompt[:300])
        with prev_b:
            st.markdown(f"**{version_b.name}**")
            st.text(version_b.system_prompt[:300])

    # Input area
    st.caption("One input per line (plain text) or a JSON array.")
    compare_input = st.text_area(
        "Comparison inputs",
        height=150,
        key="ab_compare_input",
    )

    if st.button("Run A/B Comparison", key="run_ab_compare"):
        inputs = _parse_batch(compare_input)
        if not inputs:
            st.warning("No valid inputs found.")
            return

        progress = st.progress(0)

        def _progress(current, total):
            progress.progress(current / total)

        with st.spinner("Running comparison..."):
            results = run_comparison(
                state,
                version_a,
                version_b,
                inputs,
                progress_callback=_progress,
            )

        _show_comparison(results, version_a, version_b)


def _show_comparison(results, version_a, version_b):
    """Display comparison results with summary and
    per-input detail."""
    import streamlit as st

    total = len(results)
    matching = sum(1 for r in results if r.outputs_match)
    errors_a = sum(1 for r in results if r.error_a is not None)
    errors_b = sum(1 for r in results if r.error_b is not None)

    # Summary metrics
    st.subheader("Summary")
    cols = st.columns(4)
    cols[0].metric("Total Inputs", total)
    cols[1].metric("Matching Outputs", matching)
    cols[2].metric(
        "Agreement",
        f"{(matching / total * 100):.1f}%" if total > 0 else "N/A",
    )
    cols[3].metric(
        "Errors",
        f"A:{errors_a} / B:{errors_b}",
    )

    # Per-input results
    st.subheader("Per-Input Results")
    for i, r in enumerate(results):
        label = f"Input {i + 1}" + (" ✓" if r.outputs_match else " ✗")
        with st.expander(label):
            st.markdown("**Input:**")
            st.json(r.input_data)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**{version_a.name}**")
                if r.error_a:
                    st.error(r.error_a)
                elif r.output_a is not None:
                    st.json(r.output_a)
                else:
                    st.warning("Model returned None")
                if r.metrics_a:
                    st.caption(f"Tokens: {r.metrics_a.get('tokens', '?')}")

            with col_b:
                st.markdown(f"**{version_b.name}**")
                if r.error_b:
                    st.error(r.error_b)
                elif r.output_b is not None:
                    st.json(r.output_b)
                else:
                    st.warning("Model returned None")
                if r.metrics_b:
                    st.caption(f"Tokens: {r.metrics_b.get('tokens', '?')}")


# ----------------------------------------------------------------
# History Tab
# ----------------------------------------------------------------


def _render_history():
    import streamlit as st

    if not st.session_state.test_history:
        st.info("No test history yet.")
        return

    for i, entry in enumerate(reversed(st.session_state.test_history)):
        with st.expander(f"Test {len(st.session_state.test_history) - i}"):
            st.json(entry)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


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
