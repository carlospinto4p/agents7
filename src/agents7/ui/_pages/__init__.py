"""Shared helpers used across Streamlit pages."""

from __future__ import annotations

from agents7.ui.state import (
    AgentBuilderState,
    EnumMember,
    FieldDefinition,
)


def make_progress_callback(progress_bar) -> callable:
    """Create a progress callback for a Streamlit progress bar.
    :param progress_bar: A `st.progress()` widget instance.
    :return: Callback accepting `(current, total)`.
    """

    def _callback(current: int, total: int) -> None:
        progress_bar.progress(current / total)

    return _callback


def get_expected_fields(state: AgentBuilderState) -> list[str]:
    """Determine which output fields to compare for evaluation.
    :param state: The current agent builder state.
    :return: List of field names.
    """
    if state.agent_type == "classifier":
        return ["category", "reason"]
    if state.agent_type in ("detector", "regressor"):
        return ["reason"]
    return [f.name for f in state.fields]


def _state_from_dict(sd: dict) -> AgentBuilderState:
    """Reconstruct an AgentBuilderState from session dict."""
    return AgentBuilderState(
        agent_type=sd.get("agent_type", "classifier"),
        agent_name=sd.get("agent_name", ""),
        categories=[EnumMember(**c) for c in sd.get("categories", [])],
        reasons=[EnumMember(**r) for r in sd.get("reasons", [])],
        category_reasons=sd.get("category_reasons", {}),
        value_type=sd.get("value_type", "float"),
        fields=[FieldDefinition(**f) for f in sd.get("fields", [])],
        model_name=sd.get("model_name", "qwen3"),
        system_prompt=sd.get("system_prompt", ""),
        examples=sd.get("examples", []),
        eval_data=sd.get("eval_data", []),
    )
