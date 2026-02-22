"""Step 2 — Configure the agent (model, prompt, examples)."""

from __future__ import annotations

import streamlit as st

from agents7.ui._pages import _state_from_dict, resize_list
from agents7.ui.codegen.yaml_generator import generate_yaml
from agents7.ui.helpers import build_agent_from_state
from agents7.ui.ollama_utils import (
    check_connection,
    list_available_models,
)


def render() -> None:
    st.header("2. Configure the Agent")

    sd = st.session_state.state_dict

    # --- Ollama model selection ---
    st.subheader("LLM Model")
    connected = check_connection()
    if connected:
        models = list_available_models()
        if models:
            current = sd.get("model_name", "qwen3")
            idx = models.index(current) if current in models else 0
            sd["model_name"] = st.selectbox("Ollama model", models, index=idx)
        else:
            st.warning("Ollama is running but no models found.")
            sd["model_name"] = st.text_input(
                "Model name", value=sd.get("model_name", "qwen3")
            )
    else:
        st.error(
            "Cannot connect to Ollama. "
            "Make sure it is running (`ollama serve`)."
        )
        sd["model_name"] = st.text_input(
            "Model name (manual)",
            value=sd.get("model_name", "qwen3"),
        )

    # --- System prompt ---
    st.subheader("System Prompt")

    # Generate from discovery data
    _prompt_generation_section(sd)

    sd["system_prompt"] = st.text_area(
        "System prompt",
        value=sd.get("system_prompt", ""),
        height=200,
    )

    # --- Examples ---
    st.subheader("Input / Output Examples")
    examples = sd.get("examples", [])
    num_ex = st.number_input(
        "Number of examples",
        min_value=0,
        value=len(examples),
        step=1,
    )
    examples = resize_list(
        examples,
        num_ex,
        lambda: {"input": "", "output": ""},
    )

    for i, ex in enumerate(examples):
        cols = st.columns(2)
        ex["input"] = cols[0].text_area(
            f"Example {i + 1} — Input",
            value=ex.get("input", ""),
            key=f"ex_in_{i}",
            height=80,
        )
        ex["output"] = cols[1].text_area(
            f"Example {i + 1} — Output",
            value=ex.get("output", ""),
            key=f"ex_out_{i}",
            height=80,
        )

    sd["examples"] = examples

    # --- YAML preview ---
    st.subheader("YAML Config Preview")
    state = _state_from_dict(sd)
    yml = generate_yaml(state)
    st.code(yml, language="yaml")

    # --- Resolved prompt preview ---
    st.subheader("Resolved Prompt Preview")
    try:
        agent = build_agent_from_state(state)
        st.code(agent.prompt, language="text")
    except Exception as e:
        st.warning(f"Cannot preview prompt: {e}")


def _prompt_generation_section(sd: dict) -> None:
    """Button to auto-generate system prompt and examples."""
    has_intent = bool(sd.get("intent", "").strip())
    has_data = bool(sd.get("sample_data"))

    if not (has_intent and has_data):
        st.info(
            "To auto-generate the prompt, provide an "
            "**intent** and **sample data** in Step 1 "
            "(Discover from Data)."
        )
        return

    if st.button(
        "Generate Prompt & Examples",
        key="generate_prompt_btn",
    ):
        _run_prompt_generation(sd)


def _run_prompt_generation(sd: dict) -> None:
    """Execute prompt generation and merge results."""
    from agents7.llm import LLMEngine
    from agents7.ui.discovery.analyzer import generate_prompt

    agent_type = sd.get("agent_type", "classifier")
    intent = sd["intent"]
    sample_data = sd["sample_data"]
    model_name = sd.get("model_name", "qwen3")

    engine = LLMEngine(provider="ollama", model_name=model_name)

    with st.spinner("Generating prompt and examples..."):
        try:
            result = generate_prompt(
                engine, agent_type, intent, sd, sample_data
            )
            sd["system_prompt"] = result["system_prompt"]
            sd["examples"] = result["examples"]
            st.success(
                "Prompt and examples generated! Review and edit below."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Prompt generation failed: {exc}")
