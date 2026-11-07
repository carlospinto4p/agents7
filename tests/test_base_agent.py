"""Unit tests for agents7.agents.base.BaseAgent."""

from pydantic import BaseModel

from helpers import DummyOutput, DummyOutputWithEnum, make_labeler


# ── describe() ─────────────────────────────────────────────────────────────


def test_describe_contains_name_and_version():
    agent = make_labeler(name="MyAgent", version=3)
    desc = agent.describe()
    assert "MyAgent" in desc
    assert "v3" in desc


def test_describe_contains_model_id():
    agent = make_labeler()
    assert agent.engine.model_id in agent.describe()


def test_describe_shows_output_model_name():
    agent = make_labeler(output_model=DummyOutput)
    assert "DummyOutput" in agent.describe()


def test_describe_no_enum_fields_omits_section():
    """Models with no StrEnum fields omit the Enum fields section."""

    class PlainOutput(BaseModel):
        value: str

    agent = make_labeler(output_model=PlainOutput)
    assert "Enum fields:" not in agent.describe()


def test_describe_with_enum_fields_shows_members():
    """Models with StrEnum fields list them under 'Enum fields:'."""
    agent = make_labeler(output_model=DummyOutputWithEnum)
    desc = agent.describe()
    assert "Enum fields:" in desc
    assert "reason" in desc
    assert "GOOD" in desc
    assert "BAD" in desc


def test_describe_output_is_string():
    assert isinstance(make_labeler().describe(), str)


# ── _set_attr() ────────────────────────────────────────────────────────────


def test_set_attr_mutates_frozen_name():
    agent = make_labeler(name="before")
    agent._set_attr("name", "after")
    assert agent.name == "after"


def test_set_attr_mutates_frozen_version():
    agent = make_labeler(version=1)
    agent._set_attr("version", 99)
    assert agent.version == 99


def test_set_attr_modifies_in_place():
    """_set_attr changes the same object, not a copy."""
    agent = make_labeler(name="original")
    original_id = id(agent)
    agent._set_attr("name", "changed")
    assert id(agent) == original_id
    assert agent.name == "changed"


# ── resolved_system_prompt edge cases ─────────────────────────────────────


def test_resolved_prompt_value_containing_dollar():
    """A value containing $ triggers no further substitution."""
    agent = make_labeler(
        system_prompt="Price: $price",
        prompt_params={"price": "$100"},
    )
    assert agent.resolved_system_prompt == "Price: $100"


def test_resolved_prompt_empty_string_value():
    """Empty string value replaces placeholder with empty string."""
    agent = make_labeler(
        system_prompt="Hello $name!",
        prompt_params={"name": ""},
    )
    assert agent.resolved_system_prompt == "Hello !"


def test_resolved_prompt_multiple_identical_placeholders():
    """Same placeholder used twice is substituted both times."""
    agent = make_labeler(
        system_prompt="$x and $x",
        prompt_params={"x": "yes"},
    )
    assert agent.resolved_system_prompt == "yes and yes"
