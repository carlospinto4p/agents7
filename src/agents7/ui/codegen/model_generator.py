"""Generate Pydantic model source code and live classes.

Two public functions:

* `generate_model_code(state)` — returns the `.py` file content
  that users will export.
* `create_model_class(state)` — returns a live class usable by
  the in-memory agent during Test / Evaluate steps.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, create_model

from agents7.models.base import ClassificationProperty, Property
from agents7.ui.state import AgentBuilderState, EnumMember


# ------------------------------------------------------------------
# Source code generation  (Step 5 — Export)
# ------------------------------------------------------------------


def _enum_code(
    class_name: str,
    members: list[EnumMember],
) -> list[str]:
    """Generate source lines for a StrEnum class."""
    lines = [f"class {class_name}(StrEnum):"]
    for m in members:
        lines.append(f'    {m.name} = "{m.description}"')
    lines += ["", ""]
    return lines


def generate_model_code(state: AgentBuilderState) -> str:
    """Return full `.py` source for the output model."""
    if state.agent_type == "classifier":
        return _code_for_classifier(state)
    if state.agent_type in ("detector", "regressor"):
        return _code_for_property(state)
    return _code_for_labeler(state)


def _code_for_classifier(state: AgentBuilderState) -> str:
    lines: list[str] = [
        "from enum import StrEnum",
        "",
        "from agents7.models import ClassificationProperty",
        "",
        "",
    ]
    cat_name = state.category_enum_name
    lines += _enum_code(cat_name, state.categories)

    rsn_name = state.reason_enum_name
    lines += _enum_code(rsn_name, state.reasons)

    # Model class
    cls_name = state.model_class_name
    lines.append(f"class {cls_name}(ClassificationProperty):")
    lines.append(f"    category: {cat_name}")
    lines.append(f"    reason: {rsn_name}")

    if state.category_reasons:
        lines.append("")
        lines.append("    CATEGORY_REASONS = {")
        for cat, rsns in state.category_reasons.items():
            reason_refs = ", ".join(f"{rsn_name}.{r}" for r in rsns)
            lines.append(f"        {cat_name}.{cat}: [{reason_refs}],")
        lines.append("    }")

    lines.append("")
    return "\n".join(lines)


def _code_for_property(state: AgentBuilderState) -> str:
    lines: list[str] = [
        "from enum import StrEnum",
        "",
        "from agents7.models import Property",
        "",
        "",
    ]
    rsn_name = state.reason_enum_name
    lines += _enum_code(rsn_name, state.reasons)

    cls_name = state.model_class_name
    vt = state.value_type
    lines.append(f"class {cls_name}(Property[{vt}]):")
    lines.append(f"    reason: {rsn_name}")
    lines.append("")
    return "\n".join(lines)


def _code_for_labeler(state: AgentBuilderState) -> str:
    lines: list[str] = [
        "from pydantic import BaseModel",
        "",
        "",
    ]
    cls_name = state.model_class_name
    lines.append(f"class {cls_name}(BaseModel):")
    if not state.fields:
        lines.append("    pass")
    else:
        for f in state.fields:
            ft = f.field_type
            if f.optional:
                ft = f"{ft} | None"
                default = " = None"
            else:
                default = ""
            lines.append(f"    {f.name}: {ft}{default}")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Dynamic class creation  (Test / Evaluate — in-memory)
# ------------------------------------------------------------------


def create_model_class(
    state: AgentBuilderState,
) -> type[BaseModel]:
    """Build a live Pydantic model class from *state*.

    The returned class can be passed straight to an agent
    constructor as `output_model`.
    """
    if state.agent_type == "classifier":
        return _class_for_classifier(state)
    if state.agent_type in ("detector", "regressor"):
        return _class_for_property(state)
    return _class_for_labeler(state)


def _make_str_enum(
    name: str, members: list[tuple[str, str]]
) -> type[StrEnum]:
    """Create a StrEnum dynamically."""
    return StrEnum(name, members)


def _class_for_classifier(
    state: AgentBuilderState,
) -> type[ClassificationProperty]:
    cat_members = [(m.name, m.description) for m in state.categories]
    rsn_members = [(m.name, m.description) for m in state.reasons]

    cat_enum = _make_str_enum(state.category_enum_name, cat_members)
    rsn_enum = _make_str_enum(state.reason_enum_name, rsn_members)

    # Build CATEGORY_REASONS mapping
    cat_rsn_map: dict[StrEnum, list[StrEnum]] = {}
    for cat_name, reason_names in state.category_reasons.items():
        cat_rsn_map[cat_enum[cat_name]] = [rsn_enum[r] for r in reason_names]

    # Create model using type() so __init_subclass__ fires
    cls = type(
        state.model_class_name,
        (ClassificationProperty,),
        {
            "__annotations__": {
                "category": cat_enum,
                "reason": rsn_enum,
            },
            "CATEGORY_REASONS": cat_rsn_map,
        },
    )
    return cls


def _class_for_property(
    state: AgentBuilderState,
) -> type[Property]:
    rsn_members = [(m.name, m.description) for m in state.reasons]
    rsn_enum = _make_str_enum(state.reason_enum_name, rsn_members)

    value_python_type = _FIELD_TYPE_MAP.get(state.value_type, float)

    cls = type(
        state.model_class_name,
        (Property[value_python_type],),
        {
            "__annotations__": {
                "reason": rsn_enum,
            },
        },
    )
    return cls


_FIELD_TYPE_MAP = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


def _class_for_labeler(
    state: AgentBuilderState,
) -> type[BaseModel]:
    field_definitions: dict = {}
    for f in state.fields:
        ft = _FIELD_TYPE_MAP.get(f.field_type, str)
        if f.optional:
            ft = ft | None
            default = None
        else:
            default = ...
        field_definitions[f.name] = (
            ft,
            Field(default=default, description=f.description),
        )
    return create_model(state.model_class_name, **field_definitions)
