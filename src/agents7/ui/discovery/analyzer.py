"""Discovery logic: schema analysis and prompt generation."""

from __future__ import annotations

from agents7.llm import LLMEngine
from agents7.ui.discovery.models import (
    ClassifierSuggestion,
    LabelerSuggestion,
    PromptSuggestion,
    PropertySuggestion,
)
from agents7.ui.discovery.prompts import (
    describe_schema,
    prompt_generation_prompt,
    schema_discovery_prompt,
)

_SCHEMA_MODELS: dict[str, type] = {
    "classifier": ClassifierSuggestion,
    "detector": PropertySuggestion,
    "regressor": PropertySuggestion,
    "labeler": LabelerSuggestion,
}

_MAX_SAMPLES = 30


def discover_schema(
    engine: LLMEngine,
    agent_type: str,
    intent: str,
    sample_data: list[dict],
) -> dict:
    """Discover an output schema from sample data via LLM.

    Builds a prompt with the user's intent and up to 30
    sample records, calls the LLM with structured output,
    and returns a dict ready to merge into ``state_dict``.

    :param engine: An ``LLMEngine`` instance (Ollama).
    :param agent_type: One of ``"classifier"``,
        ``"detector"``, ``"regressor"``, ``"labeler"``.
    :param intent: User's description of what the agent
        should do (1-2 sentences).
    :param sample_data: List of dicts parsed from JSONL.
    :return: Dict with keys matching ``state_dict`` fields
        (e.g. ``categories``, ``reasons``, ``fields``).
    :raises ValueError: If ``agent_type`` is not recognized.
    """
    if agent_type not in _SCHEMA_MODELS:
        msg = f"Unknown agent type: {agent_type!r}"
        raise ValueError(msg)

    model_cls = _SCHEMA_MODELS[agent_type]
    truncated = sample_data[:_MAX_SAMPLES]

    system_prompt = schema_discovery_prompt(agent_type, intent, truncated)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": ("Analyze the data and suggest the output schema."),
        },
    ]

    response = engine.chat(
        messages=messages,
        temperature=0.3,
        output_format=model_cls.model_json_schema(),
    )

    raw = response["response"]
    suggestion = model_cls.model_validate_json(raw)
    return _suggestion_to_state(agent_type, suggestion)


def generate_prompt(
    engine: LLMEngine,
    agent_type: str,
    intent: str,
    state_dict: dict,
    sample_data: list[dict],
) -> dict:
    """Generate a system prompt and examples via LLM.

    Uses the user's intent, the schema from Step 1, and
    sample data to produce a system prompt and few-shot
    examples.

    :param engine: An ``LLMEngine`` instance (Ollama).
    :param agent_type: The agent type string.
    :param intent: User's description of the task.
    :param state_dict: Current ``state_dict`` with schema
        fields already set.
    :param sample_data: List of dicts parsed from JSONL.
    :return: Dict with ``"system_prompt"`` and ``"examples"``
        keys.
    """
    truncated = sample_data[:_MAX_SAMPLES]
    schema_desc = describe_schema(state_dict, agent_type)

    system_prompt = prompt_generation_prompt(
        agent_type, intent, schema_desc, truncated
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": ("Generate the system prompt and examples."),
        },
    ]

    response = engine.chat(
        messages=messages,
        temperature=0.3,
        output_format=PromptSuggestion.model_json_schema(),
    )

    raw = response["response"]
    suggestion = PromptSuggestion.model_validate_json(raw)
    return {
        "system_prompt": suggestion.system_prompt,
        "examples": [
            {"input": ex.input, "output": ex.output}
            for ex in suggestion.examples
        ],
    }


def _suggestion_to_state(
    agent_type: str,
    suggestion: (
        ClassifierSuggestion | PropertySuggestion | LabelerSuggestion
    ),
) -> dict:
    """Convert a suggestion model to a state_dict fragment."""
    if agent_type == "classifier":
        return _classifier_to_state(suggestion)
    if agent_type in ("detector", "regressor"):
        return _property_to_state(suggestion)
    return _labeler_to_state(suggestion)


def _classifier_to_state(
    suggestion: ClassifierSuggestion,
) -> dict:
    return {
        "categories": [
            {"name": c.name, "description": c.description}
            for c in suggestion.categories
        ],
        "reasons": [
            {"name": r.name, "description": r.description}
            for r in suggestion.reasons
        ],
        "category_reasons": suggestion.category_reasons,
    }


def _property_to_state(suggestion: PropertySuggestion) -> dict:
    return {
        "value_type": suggestion.value_type,
        "reasons": [
            {"name": r.name, "description": r.description}
            for r in suggestion.reasons
        ],
    }


def _labeler_to_state(suggestion: LabelerSuggestion) -> dict:
    return {
        "fields": [
            {
                "name": f.name,
                "field_type": f.field_type,
                "optional": f.optional,
                "description": f.description,
            }
            for f in suggestion.fields
        ],
    }
