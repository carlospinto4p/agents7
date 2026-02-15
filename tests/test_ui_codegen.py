"""Unit tests for agents7.ui.codegen (model_generator + yaml_generator)."""

import yaml

from pydantic import BaseModel

from agents7.models.base import ClassificationProperty, Property
from agents7.ui.state import (
    AgentBuilderState,
    EnumMember,
    FieldDefinition,
)
from agents7.ui.codegen.model_generator import (
    create_model_class,
    generate_model_code,
)
from agents7.ui.codegen.yaml_generator import generate_yaml


# ----------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------


def _classifier_state() -> AgentBuilderState:
    return AgentBuilderState(
        agent_type="classifier",
        agent_name="sentiment",
        categories=[
            EnumMember("POSITIVE", "Positive sentiment."),
            EnumMember("NEGATIVE", "Negative sentiment."),
        ],
        reasons=[
            EnumMember("HAPPY", "Feels happy."),
            EnumMember("SAD", "Feels sad."),
        ],
        category_reasons={
            "POSITIVE": ["HAPPY"],
            "NEGATIVE": ["SAD"],
        },
        model_name="qwen3",
        system_prompt="Classify sentiment.",
    )


def _detector_state() -> AgentBuilderState:
    return AgentBuilderState(
        agent_type="detector",
        agent_name="temperature",
        reasons=[
            EnumMember("TOO_COLD", "Too cold."),
            EnumMember("COMFORTABLE", "Comfortable."),
        ],
        value_type="float",
        model_name="qwen3",
        system_prompt="Detect temperature.",
    )


def _regressor_state() -> AgentBuilderState:
    return AgentBuilderState(
        agent_type="regressor",
        agent_name="price_estimate",
        reasons=[
            EnumMember("CHEAP", "Very cheap."),
            EnumMember("EXPENSIVE", "Very expensive."),
        ],
        value_type="int",
        model_name="qwen3",
        system_prompt="Estimate price.",
    )


def _labeler_state() -> AgentBuilderState:
    return AgentBuilderState(
        agent_type="labeler",
        agent_name="person_info",
        fields=[
            FieldDefinition("name", "str", False, "Person name"),
            FieldDefinition("age", "int", True, "Person age"),
        ],
        model_name="qwen3",
        system_prompt="Extract person info.",
    )


# --- State properties ---


def test_model_class_name():
    s = AgentBuilderState(agent_name="my_cool_agent")
    assert s.model_class_name == "MyCoolAgent"


def test_model_class_name_empty():
    s = AgentBuilderState(agent_name="")
    assert s.model_class_name == "Output"


def test_model_class_name_hyphens():
    s = AgentBuilderState(agent_name="gender-classifier")
    assert s.model_class_name == "GenderClassifier"


def test_category_enum_name():
    s = AgentBuilderState(agent_name="sentiment")
    assert s.category_enum_name == "SentimentCategory"


def test_reason_enum_name():
    s = AgentBuilderState(agent_name="sentiment")
    assert s.reason_enum_name == "SentimentReason"


# --- Code generation: generate_model_code() ---


def test_classifier_code_has_enums_and_class():
    code = generate_model_code(_classifier_state())
    assert "class SentimentCategory(StrEnum):" in code
    assert "class SentimentReason(StrEnum):" in code
    assert "class Sentiment(ClassificationProperty):" in code
    assert 'POSITIVE = "Positive sentiment."' in code
    assert "CATEGORY_REASONS" in code


def test_detector_code_has_property():
    code = generate_model_code(_detector_state())
    assert "class TemperatureReason(StrEnum):" in code
    assert "class Temperature(Property[float]):" in code
    assert "reason: TemperatureReason" in code


def test_regressor_code_uses_int():
    code = generate_model_code(_regressor_state())
    assert "Property[int]" in code


def test_labeler_code_has_fields():
    code = generate_model_code(_labeler_state())
    assert "class PersonInfo(BaseModel):" in code
    assert "name: str" in code
    assert "age: int | None = None" in code


def test_labeler_empty_fields():
    s = AgentBuilderState(agent_type="labeler", agent_name="empty")
    code = generate_model_code(s)
    assert "pass" in code


def test_classifier_no_category_reasons():
    s = _classifier_state()
    s.category_reasons = {}
    code = generate_model_code(s)
    assert "CATEGORY_REASONS" not in code


# --- Dynamic class creation: create_model_class() ---


def test_classifier_creates_valid_class():
    cls = create_model_class(_classifier_state())
    assert issubclass(cls, ClassificationProperty)
    assert cls.__name__ == "Sentiment"
    # __init_subclass__ should have added NOT_APPLICABLE
    cat_enum = cls.__annotations__["category"]
    assert hasattr(cat_enum, "NOT_APPLICABLE")


def test_classifier_category_reasons_map():
    cls = create_model_class(_classifier_state())
    assert cls.CATEGORY_REASONS is not None
    assert len(cls.CATEGORY_REASONS) == 2


def test_classifier_can_instantiate():
    cls = create_model_class(_classifier_state())
    cat_enum = cls.__annotations__["category"]
    rsn_enum = cls.__annotations__["reason"]
    obj = cls(
        category=cat_enum.POSITIVE,
        reason=rsn_enum.HAPPY,
    )
    assert obj.category.name == "POSITIVE"
    assert obj.reason.name == "HAPPY"


def test_detector_creates_property_subclass():
    cls = create_model_class(_detector_state())
    assert issubclass(cls, Property)
    assert cls.__name__ == "Temperature"


def test_detector_has_not_applicable():
    cls = create_model_class(_detector_state())
    rsn_enum = cls.__annotations__["reason"]
    assert hasattr(rsn_enum, "NOT_APPLICABLE")


def test_detector_can_instantiate():
    cls = create_model_class(_detector_state())
    rsn_enum = cls.__annotations__["reason"]
    obj = cls(value=22.5, reason=rsn_enum.COMFORTABLE)
    assert obj.value == 22.5


def test_regressor_int_type():
    cls = create_model_class(_regressor_state())
    assert issubclass(cls, Property)
    rsn_enum = cls.__annotations__["reason"]
    obj = cls(value=100, reason=rsn_enum.CHEAP)
    assert obj.value == 100


def test_labeler_creates_base_model():
    cls = create_model_class(_labeler_state())
    assert issubclass(cls, BaseModel)
    assert cls.__name__ == "PersonInfo"


def test_labeler_required_field():
    cls = create_model_class(_labeler_state())
    obj = cls(name="Alice")
    assert obj.name == "Alice"
    assert obj.age is None


def test_labeler_json_schema():
    cls = create_model_class(_labeler_state())
    schema = cls.model_json_schema()
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]


# --- YAML generation ---


def test_basic_yaml_structure():
    yml = generate_yaml(_classifier_state())
    config = yaml.safe_load(yml)
    assert config["name"] == "sentiment"
    assert config["version"] == 1
    assert config["engine"]["provider"] == "ollama"
    assert config["engine"]["model"] == "qwen3"
    assert config["system_prompt"] == "Classify sentiment."
    assert config["output_model"] == "models.Sentiment"


def test_examples_included():
    s = _classifier_state()
    s.examples = [
        {
            "input": "I love it!",
            "output": {"category": "POSITIVE"},
        }
    ]
    yml = generate_yaml(s)
    config = yaml.safe_load(yml)
    assert "system_examples" in config
    assert len(config["system_examples"]) == 1


def test_no_examples_when_empty():
    yml = generate_yaml(_classifier_state())
    config = yaml.safe_load(yml)
    assert "system_examples" not in config


def test_default_name_when_empty():
    s = AgentBuilderState()
    yml = generate_yaml(s)
    config = yaml.safe_load(yml)
    assert config["name"] == "my_agent"
