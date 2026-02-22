"""Step 1 — Define the output model."""

from __future__ import annotations

import json

import streamlit as st

from agents7.ui._pages import (
    _state_from_dict,
    parse_jsonl,
    resize_list,
    safe_index,
)
from agents7.ui.codegen.model_generator import (
    generate_model_code,
)


def render() -> None:
    st.header("1. Define the Output Model")

    sd = st.session_state.state_dict

    # Agent type
    _agent_types = ["classifier", "detector", "regressor", "labeler"]
    agent_type = st.selectbox(
        "Agent type",
        _agent_types,
        index=safe_index(_agent_types, sd.get("agent_type", "classifier")),
        key="agent_type_select",
    )
    sd["agent_type"] = agent_type

    _AGENT_TYPE_INFO = {
        "classifier": (
            "Assigns a **category** and **reason** to each input. "
            "Define the possible categories and reasons as enums, "
            "and optionally constrain which reasons are valid for "
            "each category."
        ),
        "detector": (
            "Detects whether a property holds (boolean) and "
            "provides a **reason**. Use for yes/no decisions "
            "like anomaly detection or content filtering."
        ),
        "regressor": (
            "Extracts a **scalar value** (float, int, str, or "
            "bool) and provides a **reason**. Use for "
            "measurements like scores, ratings, or counts."
        ),
        "labeler": (
            "Generic agent with **custom output fields**. "
            "Define any combination of typed fields. Most "
            "flexible — use when the other types don't fit."
        ),
    }
    st.info(_AGENT_TYPE_INFO[agent_type])

    # Agent name
    sd["agent_name"] = st.text_input(
        "Agent name (snake_case)",
        value=sd.get("agent_name", ""),
    )

    # ---- Discover from Data ----
    _discovery_section(sd)

    # ---- Classifier ----
    if agent_type == "classifier":
        _classifier_form(sd)
    # ---- Detector / Regressor ----
    elif agent_type in ("detector", "regressor"):
        _property_form(sd)
    # ---- Labeler ----
    else:
        _labeler_form(sd)

    # Preview
    st.subheader("Model Code Preview")
    state = _state_from_dict(sd)
    code = generate_model_code(state)
    st.code(code, language="python")


def _classifier_form(sd: dict) -> None:
    st.subheader("Categories")
    _enum_editor(sd, "categories", "Category")

    st.subheader("Reasons")
    _enum_editor(sd, "reasons", "Reason")

    st.subheader("Category → Reason Mapping")
    categories = sd.get("categories", [])
    reasons = sd.get("reasons", [])
    if categories and reasons:
        cat_names = [c["name"] for c in categories]
        rsn_names = [r["name"] for r in reasons]
        mapping = sd.get("category_reasons", {})
        for cat in cat_names:
            selected = st.multiselect(
                f"Reasons for {cat}",
                rsn_names,
                default=mapping.get(cat, []),
                key=f"cr_{cat}",
            )
            mapping[cat] = selected
        sd["category_reasons"] = mapping


def _property_form(sd: dict) -> None:
    _value_types = ["float", "int", "str", "bool"]
    sd["value_type"] = st.selectbox(
        "Value type",
        _value_types,
        index=safe_index(_value_types, sd.get("value_type", "float")),
        key="value_type_select",
    )

    st.subheader("Reasons")
    _enum_editor(sd, "reasons", "Reason")


def _labeler_form(sd: dict) -> None:
    st.subheader("Output Fields")
    fields = sd.get("fields", [])
    num = st.number_input(
        "Number of fields",
        min_value=0,
        value=len(fields),
        step=1,
    )
    fields = resize_list(
        fields,
        num,
        lambda: {
            "name": "",
            "field_type": "str",
            "optional": False,
            "description": "",
        },
    )

    for i, f in enumerate(fields):
        cols = st.columns([2, 1, 1, 3])
        f["name"] = cols[0].text_input(
            "Name",
            value=f.get("name", ""),
            key=f"fn_{i}",
        )
        _field_types = ["str", "int", "float", "bool"]
        f["field_type"] = cols[1].selectbox(
            "Type",
            _field_types,
            index=safe_index(_field_types, f.get("field_type", "str")),
            key=f"ft_{i}",
        )
        f["optional"] = cols[2].checkbox(
            "Optional",
            value=f.get("optional", False),
            key=f"fo_{i}",
        )
        f["description"] = cols[3].text_input(
            "Description",
            value=f.get("description", ""),
            key=f"fd_{i}",
        )

    sd["fields"] = fields


def _discovery_section(sd: dict) -> None:
    """Collapsible section for LLM-powered schema discovery."""
    with st.expander("Discover from Data", expanded=False):
        st.caption(
            "Upload sample data and describe your intent — "
            "the LLM will suggest categories, reasons, or "
            "fields to pre-fill the form below."
        )

        # Model selector
        from agents7.ui.ollama_utils import (
            check_connection,
            list_available_models,
        )

        connected = check_connection()
        if connected:
            models = list_available_models()
            if models:
                current = sd.get("model_name", "qwen3")
                idx = models.index(current) if current in models else 0
                sd["model_name"] = st.selectbox(
                    "Ollama model",
                    models,
                    index=idx,
                    key="discovery_model_select",
                )
            else:
                sd["model_name"] = st.text_input(
                    "Model name",
                    value=sd.get("model_name", "qwen3"),
                    key="discovery_model_input",
                )
        else:
            st.warning(
                "Ollama not connected. Start it with "
                "`ollama serve` for auto-discovery."
            )
            sd["model_name"] = st.text_input(
                "Model name (manual)",
                value=sd.get("model_name", "qwen3"),
                key="discovery_model_manual",
            )

        # Intent
        sd["intent"] = st.text_input(
            "Intent (what should this agent do?)",
            value=sd.get("intent", ""),
            placeholder=(
                "e.g. Classify customer support tickets by urgency and topic"
            ),
            key="discovery_intent",
        )

        # JSONL upload
        uploaded = st.file_uploader(
            "Upload sample data (JSONL)",
            type=["jsonl"],
            key="discovery_upload",
        )
        if uploaded is not None:
            text = uploaded.getvalue().decode("utf-8")
            sd["sample_data"] = parse_jsonl(text)

        # Preview
        sample_data = sd.get("sample_data", [])
        if sample_data:
            st.write(
                f"**{len(sample_data)} records loaded.** Preview (first 5):"
            )
            for record in sample_data[:5]:
                st.code(json.dumps(record, ensure_ascii=False))

        # Discover button
        has_intent = bool(sd.get("intent", "").strip())
        has_data = bool(sd.get("sample_data"))

        if st.button(
            "Discover Schema",
            disabled=not (has_intent and has_data),
            key="discover_schema_btn",
        ):
            _run_discovery(sd)


def _run_discovery(sd: dict) -> None:
    """Execute schema discovery and merge results."""
    from agents7.llm import LLMEngine
    from agents7.ui.discovery.analyzer import discover_schema

    agent_type = sd.get("agent_type", "classifier")
    intent = sd["intent"]
    sample_data = sd["sample_data"]
    model_name = sd.get("model_name", "qwen3")

    engine = LLMEngine(provider="ollama", model_name=model_name)

    with st.spinner("Discovering schema..."):
        try:
            result = discover_schema(engine, agent_type, intent, sample_data)
            for key, value in result.items():
                sd[key] = value
            st.success("Schema discovered! Review below.")
            st.rerun()
        except Exception as exc:
            st.error(f"Discovery failed: {exc}")


def _enum_editor(sd: dict, key: str, label: str) -> None:
    """Reusable editor for a list of EnumMember dicts."""
    members = sd.get(key, [])
    num = st.number_input(
        f"Number of {key}",
        min_value=0,
        value=len(members),
        step=1,
        key=f"num_{key}",
    )
    members = resize_list(
        members,
        num,
        lambda: {"name": "", "description": ""},
    )

    for i, m in enumerate(members):
        cols = st.columns(2)
        m["name"] = cols[0].text_input(
            f"{label} name",
            value=m.get("name", ""),
            key=f"{key}_n_{i}",
        )
        m["description"] = cols[1].text_input(
            f"{label} description",
            value=m.get("description", ""),
            key=f"{key}_d_{i}",
        )

    sd[key] = members
