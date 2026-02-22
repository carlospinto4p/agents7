"""Pydantic output schemas for structured LLM discovery responses."""

from __future__ import annotations

from pydantic import BaseModel


class DiscoveredEnum(BaseModel):
    """A single enum member discovered from sample data.

    :param name: UPPER_SNAKE_CASE enum member name.
    :param description: Human-readable description.
    """

    name: str
    description: str


class ClassifierSuggestion(BaseModel):
    """Suggested schema for a classifier agent.

    :param categories: Discovered category enum members.
    :param reasons: Discovered reason enum members.
    :param category_reasons: Mapping from category name to
        valid reason names.
    """

    categories: list[DiscoveredEnum]
    reasons: list[DiscoveredEnum]
    category_reasons: dict[str, list[str]]


class PropertySuggestion(BaseModel):
    """Suggested schema for a detector or regressor agent.

    :param value_type: One of ``"float"``, ``"int"``,
        ``"str"``, ``"bool"``.
    :param reasons: Discovered reason enum members.
    """

    value_type: str
    reasons: list[DiscoveredEnum]


class FieldSuggestion(BaseModel):
    """A single field discovered for a labeler agent.

    :param name: snake_case field name.
    :param field_type: One of ``"str"``, ``"int"``,
        ``"float"``, ``"bool"``.
    :param optional: Whether the field is optional.
    :param description: Human-readable description.
    """

    name: str
    field_type: str
    optional: bool
    description: str


class LabelerSuggestion(BaseModel):
    """Suggested schema for a labeler agent.

    :param fields: Discovered output fields.
    """

    fields: list[FieldSuggestion]


class ExampleSuggestion(BaseModel):
    """A single input/output example for the agent.

    :param input: Example input text or JSON string.
    :param output: Expected output JSON string.
    """

    input: str
    output: str


class PromptSuggestion(BaseModel):
    """Suggested system prompt and examples.

    :param system_prompt: Generated system prompt text.
    :param examples: Generated input/output examples.
    """

    system_prompt: str
    examples: list[ExampleSuggestion]
