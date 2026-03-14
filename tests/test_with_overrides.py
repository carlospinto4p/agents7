"""Unit tests for Labeler.with_overrides."""

from pydantic import BaseModel

from agents7.llm import LLMEngine

from helpers import DummyOutput, make_labeler


# --- Basic override ---


def test_override_system_prompt():
    """with_overrides replaces a single field."""
    agent = make_labeler(system_prompt="Original.")
    clone = agent.with_overrides(system_prompt="New prompt.")
    assert clone.system_prompt == "New prompt."
    assert agent.system_prompt == "Original."


def test_override_name():
    """with_overrides replaces name."""
    agent = make_labeler(name="original")
    clone = agent.with_overrides(name="cloned")
    assert clone.name == "cloned"
    assert agent.name == "original"


def test_override_think():
    """with_overrides replaces think flag."""
    agent = make_labeler(think=False)
    clone = agent.with_overrides(think=True)
    assert clone.think is True
    assert agent.think is False


def test_override_max_parse_retries():
    """with_overrides replaces max_parse_retries."""
    agent = make_labeler(max_parse_retries=0)
    clone = agent.with_overrides(max_parse_retries=3)
    assert clone.max_parse_retries == 3
    assert agent.max_parse_retries == 0


# --- Multiple overrides ---


def test_override_multiple_fields():
    """Multiple fields can be overridden at once."""
    agent = make_labeler(name="old", system_prompt="old prompt")
    clone = agent.with_overrides(name="new", system_prompt="new prompt")
    assert clone.name == "new"
    assert clone.system_prompt == "new prompt"
    assert agent.name == "old"


# --- Unchanged fields preserved ---


def test_unchanged_fields_preserved():
    """Fields not overridden retain original values."""
    agent = make_labeler(name="test", version=5, system_prompt="prompt")
    clone = agent.with_overrides(name="cloned")
    assert clone.version == 5
    assert clone.system_prompt == "prompt"
    assert clone.output_model is DummyOutput
    assert clone.engine is agent.engine


# --- Output model swap ---


class AltOutput(BaseModel):
    label: str


def test_override_output_model():
    """with_overrides can swap the output model."""
    agent = make_labeler(output_model=DummyOutput)
    clone = agent.with_overrides(output_model=AltOutput)
    assert clone.output_model is AltOutput
    assert clone.output_schema != agent.output_schema
    assert "label" in clone.output_schema["properties"]


# --- Engine swap ---


def test_override_engine():
    """with_overrides can swap the engine."""
    engine2 = LLMEngine(provider="ollama", model_name="llama3")
    agent = make_labeler()
    clone = agent.with_overrides(engine=engine2)
    assert clone.engine.model_name == "llama3"
    assert agent.engine.model_name == "qwen3"


# --- Returns same type ---


def test_returns_same_type():
    """Clone is the same type as the original."""
    agent = make_labeler()
    clone = agent.with_overrides(name="clone")
    assert type(clone) is type(agent)


# --- Independent instances ---


def test_clone_is_independent():
    """Clone and original are independent instances."""
    agent = make_labeler()
    clone = agent.with_overrides(name="clone")
    assert agent is not clone
    assert agent.name != clone.name


# --- No args is identity copy ---


def test_no_overrides_creates_copy():
    """with_overrides() with no args creates an equal copy."""
    agent = make_labeler()
    clone = agent.with_overrides()
    assert clone is not agent
    assert clone.name == agent.name
    assert clone.version == agent.version
    assert clone.system_prompt == agent.system_prompt
