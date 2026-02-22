"""Unit tests for agents7.ui.discovery (models, prompts, analyzer)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents7.ui.discovery.models import (
    ClassifierSuggestion,
    DiscoveredEnum,
    ExampleSuggestion,
    FieldSuggestion,
    LabelerSuggestion,
    PromptSuggestion,
    PropertySuggestion,
)
from agents7.ui.discovery.prompts import (
    describe_schema,
    prompt_generation_prompt,
    schema_discovery_prompt,
)
from agents7.ui.discovery.analyzer import (
    discover_schema,
    generate_prompt,
)


# --- Pydantic models ---


def test_discovered_enum():
    e = DiscoveredEnum(name="POSITIVE", description="Good")
    assert e.name == "POSITIVE"
    assert e.description == "Good"


def test_classifier_suggestion():
    s = ClassifierSuggestion(
        categories=[
            DiscoveredEnum(name="A", description="Cat A"),
        ],
        reasons=[
            DiscoveredEnum(name="R1", description="Reason 1"),
        ],
        category_reasons={"A": ["R1"]},
    )
    assert len(s.categories) == 1
    assert s.category_reasons["A"] == ["R1"]


def test_property_suggestion():
    s = PropertySuggestion(
        value_type="float",
        reasons=[
            DiscoveredEnum(name="HIGH", description="High"),
        ],
    )
    assert s.value_type == "float"
    assert len(s.reasons) == 1


def test_field_suggestion():
    f = FieldSuggestion(
        name="age",
        field_type="int",
        optional=False,
        description="Person age",
    )
    assert f.name == "age"
    assert f.field_type == "int"
    assert not f.optional


def test_labeler_suggestion():
    s = LabelerSuggestion(
        fields=[
            FieldSuggestion(
                name="name",
                field_type="str",
                optional=False,
                description="Person name",
            ),
        ],
    )
    assert len(s.fields) == 1
    assert s.fields[0].name == "name"


def test_example_suggestion():
    ex = ExampleSuggestion(input="hello", output='{"x": 1}')
    assert ex.input == "hello"
    assert ex.output == '{"x": 1}'


def test_prompt_suggestion():
    ps = PromptSuggestion(
        system_prompt="Do X.",
        examples=[
            ExampleSuggestion(input="a", output="b"),
        ],
    )
    assert ps.system_prompt == "Do X."
    assert len(ps.examples) == 1


# --- Prompt templates ---


def test_schema_discovery_prompt_contains_intent():
    prompt = schema_discovery_prompt(
        "classifier",
        "Classify emails",
        [{"text": "Hi there"}],
    )
    assert "Classify emails" in prompt
    assert "classifier" in prompt
    assert "Hi there" in prompt


def test_schema_discovery_prompt_truncates_samples():
    data = [{"text": f"record_{i}"} for i in range(50)]
    prompt = schema_discovery_prompt("labeler", "Extract names", data)
    assert "record_29" in prompt
    assert "record_30" not in prompt


def test_prompt_generation_prompt_contains_schema():
    prompt = prompt_generation_prompt(
        "classifier",
        "Classify tickets",
        "Categories:\n  - BUG: A bug report.",
        [{"text": "ticket 1"}],
    )
    assert "Classify tickets" in prompt
    assert "BUG: A bug report" in prompt
    assert "ticket 1" in prompt


def test_describe_schema_classifier():
    sd = {
        "categories": [
            {"name": "POS", "description": "Positive"},
        ],
        "reasons": [
            {"name": "HAPPY", "description": "Happy"},
        ],
        "category_reasons": {"POS": ["HAPPY"]},
    }
    desc = describe_schema(sd, "classifier")
    assert "POS" in desc
    assert "HAPPY" in desc
    assert "POS -> HAPPY" in desc


def test_describe_schema_detector():
    sd = {
        "value_type": "bool",
        "reasons": [
            {"name": "FOUND", "description": "Found it"},
        ],
    }
    desc = describe_schema(sd, "detector")
    assert "bool" in desc
    assert "FOUND" in desc


def test_describe_schema_labeler():
    sd = {
        "fields": [
            {
                "name": "city",
                "field_type": "str",
                "optional": False,
                "description": "City name",
            },
        ],
    }
    desc = describe_schema(sd, "labeler")
    assert "city" in desc
    assert "str" in desc


def test_describe_schema_empty():
    desc = describe_schema({}, "classifier")
    assert desc == "(no schema defined yet)"


# --- Analyzer: discover_schema ---


def _mock_engine_chat(response_json: str) -> MagicMock:
    """Create a mock LLMEngine whose chat() returns the JSON."""
    engine = MagicMock()
    engine.chat.return_value = {"response": response_json}
    return engine


def test_discover_schema_classifier():
    suggestion = ClassifierSuggestion(
        categories=[
            DiscoveredEnum(name="SPAM", description="Spam"),
            DiscoveredEnum(name="HAM", description="Not spam"),
        ],
        reasons=[
            DiscoveredEnum(name="KEYWORDS", description="Has spam keywords"),
        ],
        category_reasons={"SPAM": ["KEYWORDS"]},
    )
    engine = _mock_engine_chat(suggestion.model_dump_json())

    result = discover_schema(
        engine,
        "classifier",
        "Classify spam",
        [{"text": "Buy now!"}],
    )
    assert "categories" in result
    assert len(result["categories"]) == 2
    assert result["categories"][0]["name"] == "SPAM"
    assert "reasons" in result
    assert "category_reasons" in result


def test_discover_schema_detector():
    suggestion = PropertySuggestion(
        value_type="bool",
        reasons=[
            DiscoveredEnum(name="PRESENT", description="Is present"),
        ],
    )
    engine = _mock_engine_chat(suggestion.model_dump_json())

    result = discover_schema(
        engine,
        "detector",
        "Detect anomalies",
        [{"value": 42}],
    )
    assert result["value_type"] == "bool"
    assert len(result["reasons"]) == 1


def test_discover_schema_regressor():
    suggestion = PropertySuggestion(
        value_type="float",
        reasons=[
            DiscoveredEnum(name="HIGH_SCORE", description="Score is high"),
        ],
    )
    engine = _mock_engine_chat(suggestion.model_dump_json())

    result = discover_schema(
        engine,
        "regressor",
        "Extract score",
        [{"text": "Score: 9.5"}],
    )
    assert result["value_type"] == "float"
    assert result["reasons"][0]["name"] == "HIGH_SCORE"


def test_discover_schema_labeler():
    suggestion = LabelerSuggestion(
        fields=[
            FieldSuggestion(
                name="person_name",
                field_type="str",
                optional=False,
                description="Name of person",
            ),
            FieldSuggestion(
                name="age",
                field_type="int",
                optional=True,
                description="Age",
            ),
        ],
    )
    engine = _mock_engine_chat(suggestion.model_dump_json())

    result = discover_schema(
        engine,
        "labeler",
        "Extract person info",
        [{"text": "John, 30"}],
    )
    assert len(result["fields"]) == 2
    assert result["fields"][0]["name"] == "person_name"
    assert result["fields"][1]["optional"] is True


def test_discover_schema_unknown_type():
    engine = _mock_engine_chat("{}")
    with pytest.raises(ValueError, match="Unknown agent type"):
        discover_schema(engine, "unknown", "test", [])


def test_discover_schema_truncates_samples():
    suggestion = PropertySuggestion(
        value_type="float",
        reasons=[
            DiscoveredEnum(name="R", description="Reason"),
        ],
    )
    engine = _mock_engine_chat(suggestion.model_dump_json())

    big_data = [{"i": i} for i in range(100)]
    discover_schema(engine, "detector", "test", big_data)

    call_args = engine.chat.call_args
    messages = call_args[1]["messages"]
    system_msg = messages[0]["content"]
    assert "{'i': 29}" in system_msg
    assert "{'i': 30}" not in system_msg


def test_discover_schema_malformed_response():
    engine = _mock_engine_chat("not valid json")
    with pytest.raises(Exception):
        discover_schema(engine, "classifier", "test", [{"x": 1}])


# --- Analyzer: generate_prompt ---


def test_generate_prompt():
    suggestion = PromptSuggestion(
        system_prompt="Classify the input.",
        examples=[
            ExampleSuggestion(
                input="hello",
                output='{"category": "GREETING"}',
            ),
        ],
    )
    engine = _mock_engine_chat(suggestion.model_dump_json())

    sd = {
        "categories": [
            {"name": "GREETING", "description": "A greeting"},
        ],
        "reasons": [],
    }
    result = generate_prompt(
        engine,
        "classifier",
        "Classify greetings",
        sd,
        [{"text": "hi"}],
    )
    assert result["system_prompt"] == "Classify the input."
    assert len(result["examples"]) == 1
    assert result["examples"][0]["input"] == "hello"
    assert "GREETING" in result["examples"][0]["output"]


def test_generate_prompt_malformed_response():
    engine = _mock_engine_chat("{invalid}")
    with pytest.raises(Exception):
        generate_prompt(engine, "classifier", "test", {}, [{"x": 1}])


def test_generate_prompt_uses_schema_description():
    suggestion = PromptSuggestion(
        system_prompt="Do it.",
        examples=[],
    )
    engine = _mock_engine_chat(suggestion.model_dump_json())

    sd = {
        "categories": [
            {"name": "A", "description": "Cat A"},
        ],
        "reasons": [],
    }
    generate_prompt(engine, "classifier", "intent", sd, [{"x": 1}])

    call_args = engine.chat.call_args
    messages = call_args[1]["messages"]
    system_msg = messages[0]["content"]
    assert "Cat A" in system_msg
