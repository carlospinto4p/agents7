"""System prompts for the discovery LLM calls."""

_AGENT_TYPE_DESCRIPTIONS: dict[str, str] = {
    "classifier": (
        "A classifier agent assigns each input a CATEGORY "
        "(from a fixed enum) and a REASON (from a fixed enum). "
        "Categories and reasons are StrEnum members with "
        "UPPER_SNAKE_CASE names and human-readable descriptions "
        "as values. A category-to-reason mapping constrains "
        "which reasons are valid for each category."
    ),
    "detector": (
        "A detector agent outputs a boolean value (true/false) "
        "indicating whether a property holds, plus a REASON "
        "(from a fixed enum). Reasons are StrEnum members with "
        "UPPER_SNAKE_CASE names and human-readable descriptions."
    ),
    "regressor": (
        "A regressor agent extracts a scalar value (float, int, "
        "str, or bool) from the input, plus a REASON (from a "
        "fixed enum). You must also choose the value type."
    ),
    "labeler": (
        "A labeler agent extracts structured data with custom "
        "output fields. Each field has a name (snake_case), "
        "a type (str, int, float, or bool), an optional flag, "
        "and a description."
    ),
}


def schema_discovery_prompt(
    agent_type: str,
    intent: str,
    sample_data: list[dict],
) -> str:
    """Build the system prompt for schema discovery.

    :param agent_type: One of ``"classifier"``,
        ``"detector"``, ``"regressor"``, ``"labeler"``.
    :param intent: User's 1-2 sentence description of what
        the agent should do.
    :param sample_data: First N sample records (dicts).
    :return: System prompt string.
    """
    type_desc = _AGENT_TYPE_DESCRIPTIONS[agent_type]
    samples_text = "\n".join(str(r) for r in sample_data[:30])

    return (
        "You are an expert at designing output schemas for "
        "LLM-based data processing agents.\n\n"
        f"Agent type: {agent_type}\n"
        f"Description: {type_desc}\n\n"
        f"User intent: {intent}\n\n"
        "Sample data (up to 30 records):\n"
        f"{samples_text}\n\n"
        "Analyze the sample data and the user's intent, then "
        "suggest an appropriate output schema.\n\n"
        "Rules:\n"
        "- Enum member names must be UPPER_SNAKE_CASE.\n"
        "- Enum descriptions should be short and clear.\n"
        "- Field names must be snake_case.\n"
        "- Keep the schema focused on the user's intent.\n"
        "- Derive categories/reasons/fields from patterns "
        "you observe in the data.\n"
        "- Respond ONLY with the JSON object matching the "
        "required output format."
    )


def prompt_generation_prompt(
    agent_type: str,
    intent: str,
    schema_description: str,
    sample_data: list[dict],
) -> str:
    """Build the system prompt for prompt & example generation.

    :param agent_type: One of ``"classifier"``,
        ``"detector"``, ``"regressor"``, ``"labeler"``.
    :param intent: User's 1-2 sentence description.
    :param schema_description: Text summary of the schema
        defined in Step 1 (categories, reasons, fields, etc.).
    :param sample_data: First N sample records (dicts).
    :return: System prompt string.
    """
    type_desc = _AGENT_TYPE_DESCRIPTIONS[agent_type]
    samples_text = "\n".join(str(r) for r in sample_data[:30])

    return (
        "You are an expert at writing system prompts and "
        "few-shot examples for LLM-based agents.\n\n"
        f"Agent type: {agent_type}\n"
        f"Description: {type_desc}\n\n"
        f"User intent: {intent}\n\n"
        f"Output schema:\n{schema_description}\n\n"
        "Sample data (up to 30 records):\n"
        f"{samples_text}\n\n"
        "Generate:\n"
        "1. A clear, concise system prompt that instructs the "
        "LLM to produce outputs matching the schema above.\n"
        "2. A few (2-4) input/output examples picked from or "
        "inspired by the sample data. Each example output must "
        "be a valid JSON string matching the output schema.\n\n"
        "Rules:\n"
        "- The system prompt should describe the task, the "
        "expected output format, and any constraints.\n"
        "- Examples should cover different categories/cases.\n"
        "- Example outputs must be JSON strings (not objects).\n"
        "- Respond ONLY with the JSON object matching the "
        "required output format."
    )


def describe_schema(sd: dict, agent_type: str) -> str:
    """Build a text description of the current schema state.

    :param sd: The session ``state_dict``.
    :param agent_type: The agent type string.
    :return: Human-readable schema summary.
    """
    lines: list[str] = []

    if agent_type == "classifier":
        cats = sd.get("categories", [])
        if cats:
            lines.append("Categories:")
            for c in cats:
                lines.append(f"  - {c['name']}: {c.get('description', '')}")
        rsns = sd.get("reasons", [])
        if rsns:
            lines.append("Reasons:")
            for r in rsns:
                lines.append(f"  - {r['name']}: {r.get('description', '')}")
        cr = sd.get("category_reasons", {})
        if cr:
            lines.append("Category-Reason mapping:")
            for cat, reasons in cr.items():
                lines.append(f"  {cat} -> {', '.join(reasons)}")

    elif agent_type in ("detector", "regressor"):
        vt = sd.get("value_type", "float")
        lines.append(f"Value type: {vt}")
        rsns = sd.get("reasons", [])
        if rsns:
            lines.append("Reasons:")
            for r in rsns:
                lines.append(f"  - {r['name']}: {r.get('description', '')}")

    elif agent_type == "labeler":
        fields = sd.get("fields", [])
        if fields:
            lines.append("Fields:")
            for f in fields:
                opt = " (optional)" if f.get("optional") else ""
                lines.append(
                    f"  - {f['name']}: {f.get('field_type', 'str')}"
                    f"{opt} — {f.get('description', '')}"
                )

    return "\n".join(lines) if lines else "(no schema defined yet)"
