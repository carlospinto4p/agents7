"""Generate YAML configuration from AgentBuilderState."""

from __future__ import annotations

import yaml

from agents7.ui.state import AgentBuilderState


def generate_yaml(state: AgentBuilderState) -> str:
    """Return a YAML config string for the agent."""
    config: dict = {
        "name": state.agent_name or "my_agent",
        "version": 1,
        "engine": {
            "provider": "ollama",
            "model": state.model_name,
        },
        "system_prompt": state.system_prompt,
        "output_model": (f"models.{state.model_class_name}"),
    }

    if state.examples:
        config["system_examples"] = state.examples

    return yaml.dump(
        config,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
