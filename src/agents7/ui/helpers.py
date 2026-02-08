"""Shared helpers for building agents from UI state."""

from __future__ import annotations

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
