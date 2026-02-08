"""Step 1 — Define the output model."""

from __future__ import annotations


def render():
    import streamlit as st

    from agents7.ui.codegen.model_generator import (
        generate_model_code,
    )
    from agents7.ui._pages import _state_from_dict

    st.header("1. Define the Output Model")

    sd = st.session_state.state_dict

    # Agent type
    agent_type = st.selectbox(
        "Agent type",
        ["classifier", "detector", "regressor", "labeler"],
        index=["classifier", "detector", "regressor", "labeler"].index(
            sd.get("agent_type", "classifier")
        ),
    )
    sd["agent_type"] = agent_type

    # Agent name
    sd["agent_name"] = st.text_input(
        "Agent name (snake_case)",
        value=sd.get("agent_name", ""),
    )

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


def _classifier_form(sd: dict):
    import streamlit as st

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


def _property_form(sd: dict):
    import streamlit as st

    sd["value_type"] = st.selectbox(
        "Value type",
        ["float", "int", "str", "bool"],
        index=["float", "int", "str", "bool"].index(
            sd.get("value_type", "float")
        ),
    )

    st.subheader("Reasons")
    _enum_editor(sd, "reasons", "Reason")


def _labeler_form(sd: dict):
    import streamlit as st

    st.subheader("Output Fields")
    fields = sd.get("fields", [])
    num = st.number_input(
        "Number of fields",
        min_value=0,
        value=len(fields),
        step=1,
    )
    while len(fields) < num:
        fields.append(
            {
                "name": "",
                "field_type": "str",
                "optional": False,
                "description": "",
            }
        )
    fields = fields[:num]

    for i, f in enumerate(fields):
        cols = st.columns([2, 1, 1, 3])
        f["name"] = cols[0].text_input(
            "Name",
            value=f.get("name", ""),
            key=f"fn_{i}",
        )
        f["field_type"] = cols[1].selectbox(
            "Type",
            ["str", "int", "float", "bool"],
            index=["str", "int", "float", "bool"].index(
                f.get("field_type", "str")
            ),
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


def _enum_editor(sd: dict, key: str, label: str):
    """Reusable editor for a list of EnumMember dicts."""
    import streamlit as st

    members = sd.get(key, [])
    num = st.number_input(
        f"Number of {label.lower()}s",
        min_value=0,
        value=len(members),
        step=1,
        key=f"num_{key}",
    )
    while len(members) < num:
        members.append({"name": "", "description": ""})
    members = members[:num]

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
