"""Shared helpers used across Streamlit pages."""

from __future__ import annotations

import json
from collections.abc import Callable

from agents7.ui.state import (
    AgentBuilderState,
    EnumMember,
    FieldDefinition,
)


def make_progress_callback(
    progress_bar,
) -> Callable[[int, int], None]:
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


def parse_jsonl(text: str) -> list[dict]:
    """Parse JSONL text into a list of dicts.

    Each non-empty line is parsed as JSON. Lines that are valid
    JSON objects are kept as-is; non-object JSON values and
    malformed lines are wrapped as ``{"text": <line>}``.

    :param text: Raw JSONL content (one JSON object per line).
    :return: List of dicts, one per non-empty line.
    """
    records: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
            else:
                records.append({"text": str(obj)})
        except json.JSONDecodeError:
            records.append({"text": line})
    return records


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
