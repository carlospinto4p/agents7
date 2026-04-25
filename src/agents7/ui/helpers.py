"""Shared helpers for building agents from UI state."""

import hashlib
import json

import streamlit as st

from agents7.agents.classifier import Classifier
from agents7.agents.detector import Detector
from agents7.agents.labeler import Labeler
from agents7.agents.regressor import Regressor
from agents7.llm import LLMEngine
from agents7.ui.codegen.model_generator import create_model_class
from agents7.ui.state import AgentBuilderState


_AGENT_CLASS_MAP: dict[str, type[Labeler]] = {
    "classifier": Classifier,
    "detector": Detector,
    "regressor": Regressor,
    "labeler": Labeler,
}


def build_agent_from_state(
    state: AgentBuilderState,
) -> Labeler:
    """Construct a live agent from the current UI state.

    The agent is fully in-memory — no files are written.
    """
    agent_cls = _AGENT_CLASS_MAP.get(state.agent_type, Labeler)
    output_model = create_model_class(state)
    engine = LLMEngine(provider="ollama", model_name=state.model_name)

    return agent_cls(
        name=state.agent_name or "ui_agent",
        version=1,
        engine=engine,
        system_prompt=state.system_prompt,
        output_model=output_model,
        examples=state.examples or None,
    )


def _state_cache_key(state: AgentBuilderState) -> str:
    """Return an MD5 hex digest of the agent-construction fields."""
    data = {
        "agent_type": state.agent_type,
        "agent_name": state.agent_name,
        "model_name": state.model_name,
        "system_prompt": state.system_prompt,
        "examples": state.examples,
        "categories": [
            {"name": m.name, "description": m.description}
            for m in state.categories
        ],
        "reasons": [
            {"name": m.name, "description": m.description}
            for m in state.reasons
        ],
        "category_reasons": state.category_reasons,
        "value_type": state.value_type,
        "fields": [
            {
                "name": f.name,
                "field_type": f.field_type,
                "optional": f.optional,
                "description": f.description,
            }
            for f in state.fields
        ],
    }
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


@st.cache_resource
def _build_agent_cached(cache_key: str, _state: AgentBuilderState) -> Labeler:
    return build_agent_from_state(_state)


def get_cached_agent(state: AgentBuilderState) -> Labeler:
    """Return a cached agent, rebuilding only when state changes."""
    return _build_agent_cached(_state_cache_key(state), state)
