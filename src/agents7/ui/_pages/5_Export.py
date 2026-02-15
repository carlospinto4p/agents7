"""Step 5 — Export agent code, config, and eval data."""

from __future__ import annotations

import csv
import os

import streamlit as st

from agents7.ui._pages import _state_from_dict, get_expected_fields
from agents7.ui.codegen.model_generator import generate_model_code
from agents7.ui.codegen.yaml_generator import generate_yaml
from agents7.ui.evaluation.export import (
    format_results_csv,
    format_results_jsonl,
)


def render():
    st.header("5. Export")

    sd = st.session_state.state_dict
    state = _state_from_dict(sd)

    agent_name = state.agent_name or "my_agent"
    workspace_dir = os.path.join("workspace", agent_name)

    st.info(f"Files will be saved to: `{workspace_dir}/`")

    # --- Preview files ---
    model_code = generate_model_code(state)
    yaml_config = generate_yaml(state)
    usage_code = _generate_usage_example(state)

    tab_model, tab_yaml, tab_usage = st.tabs(
        ["models.py", "config.yml", "usage_example.py"]
    )
    with tab_model:
        st.code(model_code, language="python")
    with tab_yaml:
        st.code(yaml_config, language="yaml")
    with tab_usage:
        st.code(usage_code, language="python")

    # --- Export button ---
    if st.button("Export to workspace"):
        try:
            os.makedirs(workspace_dir, exist_ok=True)

            _write(workspace_dir, "models.py", model_code)
            _write(workspace_dir, "config.yml", yaml_config)
            _write(workspace_dir, "usage_example.py", usage_code)

            # Eval data CSV
            eval_data = sd.get("eval_data", [])
            if eval_data:
                _write_csv(workspace_dir, eval_data)

            # Eval results CSV + JSONL
            eval_results = st.session_state.get("eval_results", [])
            if eval_results:
                _write_eval_results(workspace_dir, eval_results, state)

            st.success(f"Exported to `{workspace_dir}/`")
        except Exception as e:
            st.error(f"Export failed: {e}")


def _write(directory: str, filename: str, content: str):
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_csv(directory: str, data: list[dict]):
    path = os.path.join(directory, "eval_data.csv")
    if not data:
        return
    fieldnames = list(data[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def _write_eval_results(directory: str, eval_results, state):
    expected_fields = get_expected_fields(state)

    csv_content = format_results_csv(eval_results, expected_fields)
    path_csv = os.path.join(directory, "eval_results.csv")
    with open(path_csv, "w", encoding="utf-8") as f:
        f.write(csv_content)

    jsonl_content = format_results_jsonl(eval_results)
    path_jsonl = os.path.join(directory, "eval_results.jsonl")
    with open(path_jsonl, "w", encoding="utf-8") as f:
        f.write(jsonl_content)


def _generate_usage_example(state) -> str:
    """Generate a usage_example.py showing how to use the agent."""
    agent_type = state.agent_type.capitalize()

    lines = [
        f"from agents7.agents import {agent_type}",
        "",
        "",
        f'agent = {agent_type}.from_yaml("config.yml")',
        "",
        "# Single input",
        'response = agent.act({"text": "your input here"})',
        "print(response.model)",
        "",
        "# Batch input",
        "inputs = [",
        '    {"text": "input 1"},',
        '    {"text": "input 2"},',
        "]",
        "responses = agent.act_in_batch(inputs)",
        "for r in responses:",
        "    print(r.model)",
        "",
    ]
    return "\n".join(lines)
