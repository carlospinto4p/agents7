"""Shared helpers used across Streamlit pages."""

from __future__ import annotations

from agents7.ui.state import (
    AgentBuilderState,
    EnumMember,
    FieldDefinition,
)


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
