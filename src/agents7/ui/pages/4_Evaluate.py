"""Step 4 — Evaluate against labeled data."""

from __future__ import annotations


def render():
    import streamlit as st

    from agents7.ui.helpers import build_agent_from_state
    from agents7.ui.evaluation.runner import run_evaluation
    from agents7.ui.evaluation.metrics import compute_metrics
    from agents7.ui.pages import _state_from_dict

    st.header("4. Evaluate")

    sd = st.session_state.state_dict
    state = _state_from_dict(sd)

    # --- Data source ---
    tab_csv, tab_inline = st.tabs(["Upload CSV", "Inline"])

    with tab_csv:
        uploaded = st.file_uploader(
            "Upload CSV (columns: input, + expected fields)",
            type=["csv"],
        )
        if uploaded is not None:
            import pandas as pd

            df = pd.read_csv(uploaded)
            sd["eval_data"] = df.to_dict(orient="records")
            st.dataframe(df, use_container_width=True)

    with tab_inline:
        st.caption(
            "Add rows manually. First column is the input; "
            "remaining columns are expected output fields."
        )
        eval_data = sd.get("eval_data", [])
        num_rows = st.number_input(
            "Number of rows",
            min_value=0,
            value=len(eval_data),
            step=1,
        )
        expected_fields = _get_expected_fields(state)

        while len(eval_data) < num_rows:
            row = {"input": ""}
            for f in expected_fields:
                row[f] = ""
            eval_data.append(row)
        eval_data = eval_data[:num_rows]

        for i, row in enumerate(eval_data):
            cols = st.columns(1 + len(expected_fields))
            row["input"] = cols[0].text_input(
                "Input",
                value=str(row.get("input", "")),
                key=f"ev_in_{i}",
            )
            for j, f in enumerate(expected_fields):
                row[f] = cols[j + 1].text_input(
                    f,
                    value=str(row.get(f, "")),
                    key=f"ev_{f}_{i}",
                )
        sd["eval_data"] = eval_data

    # --- Run evaluation ---
    eval_data = sd.get("eval_data", [])
    expected_fields = _get_expected_fields(state)

    if st.button("Run Evaluation") and eval_data:
        try:
            agent = build_agent_from_state(state)
            progress = st.progress(0.0)

            def _progress(current, total):
                progress.progress(current / total)

            results = run_evaluation(
                agent,
                eval_data,
                expected_fields,
                progress_callback=_progress,
            )
            st.session_state.eval_results = results

        except Exception as e:
            st.error(f"Evaluation error: {e}")

    # --- Show results ---
    results = st.session_state.eval_results
    if results:
        metrics = compute_metrics(results, expected_fields)

        st.subheader("Overall Metrics")
        cols = st.columns(4)
        cols[0].metric("Total", metrics["total"])
        cols[1].metric("Passed", metrics["passed"])
        cols[2].metric("Failed", metrics["failed"])
        cols[3].metric(
            "Accuracy",
            f"{metrics['accuracy']:.1%}",
        )

        if metrics["per_field"]:
            st.subheader("Per-Field Accuracy")
            for fname, acc in metrics["per_field"].items():
                st.metric(fname, f"{acc:.1%}")

        # Confusion matrix (classifier only)
        if state.agent_type == "classifier":
            _show_confusion_matrix(results)

        # Per-case detail
        st.subheader("Per-Case Results")
        for i, r in enumerate(results):
            icon = "✅" if r.passed else "❌"
            with st.expander(f"{icon} Row {i + 1}"):
                st.write("**Input:**", r.input_data)
                st.write("**Expected:**", r.expected)
                st.write("**Actual:**", r.actual)
                if r.error:
                    st.error(r.error)


def _get_expected_fields(state) -> list[str]:
    """Determine which fields to compare."""
    if state.agent_type == "classifier":
        return ["category", "reason"]
    if state.agent_type in ("detector", "regressor"):
        return ["reason"]
    return [f.name for f in state.fields]


def _show_confusion_matrix(results):
    import streamlit as st

    from agents7.ui.evaluation.metrics import (
        build_confusion_matrix,
    )

    cm = build_confusion_matrix(results, "category")
    if not cm["labels"]:
        return

    st.subheader("Confusion Matrix (category)")
    try:
        import pandas as pd

        df = pd.DataFrame(
            cm["matrix"],
            index=cm["labels"],
            columns=cm["labels"],
        )
        st.dataframe(df, use_container_width=True)
    except ImportError:
        st.json(cm)
