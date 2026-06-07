"""Unit tests for Persona/PhysicalTraits and Chatter personas.

`PersonalityChatter` was removed in v4.0.0; persona behaviour
now lives on the base `Chatter` via the optional `persona`
composition plugin.
"""

from unittest.mock import patch


from agents7.agents import Chatter, Persona, PhysicalTraits
from agents7.llm import LLMEngine

from helpers import DUMMY_METRICS, ENGINE


# --- PhysicalTraits ---


def test_physical_traits_to_text():
    """Renders non-empty fields as text."""
    traits = PhysicalTraits(
        height="178 cm",
        build="medium",
        hair="dark brown",
        eyes="brown",
    )
    text = traits.to_text()
    assert "Height: 178 cm" in text
    assert "Build: medium" in text
    assert "Hair: dark brown" in text
    assert "Eyes: brown" in text


def test_physical_traits_empty():
    """Empty traits produce empty text."""
    traits = PhysicalTraits()
    assert traits.to_text() == ""


# --- Persona ---


def test_persona_to_prompt_contains_name():
    """Prompt starts with character name."""
    persona = Persona(full_name="David García")
    prompt = persona.to_prompt()
    assert "You are David García." in prompt


def test_persona_to_prompt_contains_fields():
    """Prompt includes all set fields."""
    persona = Persona(
        full_name="David García",
        birthdate="1990-12-04",
        nationality="Spanish",
        occupation="journalist",
        personality=["curious", "empathetic"],
        backstory="Grew up in Madrid.",
        interests=["politics", "football"],
        speaking_style="casual and direct",
    )
    prompt = persona.to_prompt()
    assert "1990-12-04" in prompt
    assert "Spanish" in prompt
    assert "journalist" in prompt
    assert "curious, empathetic" in prompt
    assert "Grew up in Madrid." in prompt
    assert "politics, football" in prompt
    assert "casual and direct" in prompt
    assert "Stay in character" in prompt


def test_persona_to_prompt_language():
    """Language appears in profile and instructions."""
    persona = Persona(
        full_name="David",
        language="Spanish",
    )
    prompt = persona.to_prompt()
    assert "Language: Spanish" in prompt
    assert "ALWAYS respond in Spanish" in prompt


def test_persona_to_prompt_no_language():
    """No language instruction when not set."""
    persona = Persona(full_name="Test")
    prompt = persona.to_prompt()
    assert "ALWAYS respond in" not in prompt


def test_persona_to_prompt_extra_fields():
    """Extra dict fields appear in prompt."""
    persona = Persona(
        full_name="Test",
        extra={"Favorite color": "blue"},
    )
    prompt = persona.to_prompt()
    assert "Favorite color: blue" in prompt


def test_persona_to_prompt_with_physical():
    """Physical traits appear in prompt."""
    persona = Persona(
        full_name="Test",
        physical=PhysicalTraits(
            height="180 cm",
            eyes="green",
        ),
    )
    prompt = persona.to_prompt()
    assert "Height: 180 cm" in prompt
    assert "Eyes: green" in prompt


def test_persona_from_dict():
    """from_dict creates Persona with nested physical."""
    data = {
        "full_name": "Ana López",
        "nationality": "Mexican",
        "physical": {
            "hair": "black",
            "eyes": "dark brown",
        },
        "personality": ["warm", "funny"],
    }
    persona = Persona.from_dict(data)
    assert persona.full_name == "Ana López"
    assert persona.nationality == "Mexican"
    assert persona.physical.hair == "black"
    assert persona.physical.eyes == "dark brown"
    assert persona.personality == ["warm", "funny"]


def test_persona_from_dict_no_physical():
    """from_dict works without physical section."""
    data = {"full_name": "Bob"}
    persona = Persona.from_dict(data)
    assert persona.full_name == "Bob"
    assert persona.physical.height == ""


# --- Chatter with a persona plugin ---


def _chat_response(text: str = "Hello!") -> dict:
    return {
        "role": "assistant",
        "response": text,
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }


def test_chatter_with_persona_creates():
    """Chatter initializes with a persona plugin."""
    persona = Persona(full_name="David García")
    agent = Chatter(
        name="david",
        version=1,
        engine=ENGINE,
        system_prompt="",
        persona=persona,
    )
    assert agent.name == "david"
    assert agent.persona.full_name == "David García"
    assert isinstance(agent, Chatter)


def test_system_prompt_auto_generated():
    """system_prompt is built from persona."""
    persona = Persona(
        full_name="David García",
        nationality="Spanish",
        occupation="journalist",
    )
    agent = Chatter(
        name="david",
        version=1,
        engine=ENGINE,
        system_prompt="",
        persona=persona,
    )
    assert "You are David García." in agent.system_prompt
    assert "Spanish" in agent.system_prompt
    assert "journalist" in agent.system_prompt
    assert "Stay in character" in agent.system_prompt


def test_extra_instructions_appended():
    """extra_instructions appear after persona prompt."""
    persona = Persona(full_name="Test")
    agent = Chatter(
        name="test",
        version=1,
        engine=ENGINE,
        system_prompt="",
        persona=persona,
        extra_instructions="Always respond in Spanish.",
    )
    assert "Always respond in Spanish." in agent.system_prompt
    assert "Extra Instructions" in agent.system_prompt


def test_system_prompt_not_manually_set():
    """Manual system_prompt is overridden by persona."""
    persona = Persona(full_name="Override Test")
    agent = Chatter(
        name="test",
        version=1,
        engine=ENGINE,
        system_prompt="This should be overridden.",
        persona=persona,
    )
    assert "Override Test" in agent.system_prompt
    assert "This should be overridden." not in agent.system_prompt


@patch.object(LLMEngine, "chat")
def test_chat_uses_persona_prompt(mock_chat):
    """Chat sends the persona-based system prompt."""
    mock_chat.return_value = _chat_response("I'm David!")
    persona = Persona(
        full_name="David García",
        occupation="journalist",
    )
    agent = Chatter(
        name="david",
        version=1,
        engine=ENGINE,
        system_prompt="",
        persona=persona,
    )
    response = agent.chat("What's your name?")
    assert response.response == "I'm David!"

    call_args = mock_chat.call_args
    messages = call_args[1]["messages"]
    system_msg = messages[0]["content"]
    assert "David García" in system_msg
    assert "journalist" in system_msg


def test_describe_includes_persona_name():
    """describe() mentions the agent name."""
    persona = Persona(full_name="David García")
    agent = Chatter(
        name="david",
        version=1,
        engine=ENGINE,
        system_prompt="",
        persona=persona,
    )
    desc = agent.describe()
    assert "david" in desc


# --- YAML loading (persona section) ---


def test_from_yaml(tmp_path):
    """Loads a persona-driven Chatter from YAML."""
    yaml_content = (
        "name: david\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3:4b\n"
        "persona:\n"
        "    full_name: David García\n"
        "    birthdate: '1990-12-04'\n"
        "    nationality: Spanish\n"
        "    occupation: journalist\n"
        "    physical:\n"
        "        height: 178 cm\n"
        "        hair: dark brown\n"
        "    personality:\n"
        "        - curious\n"
        "        - empathetic\n"
        "    backstory: Grew up in Madrid.\n"
        "    interests:\n"
        "        - politics\n"
        "        - football\n"
        "extra_instructions: Always respond in Spanish.\n"
    )
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    agent = Chatter.from_yaml(str(yaml_file))
    assert agent.name == "david"
    assert agent.persona.full_name == "David García"
    assert agent.persona.birthdate == "1990-12-04"
    assert agent.persona.nationality == "Spanish"
    assert agent.persona.physical.height == "178 cm"
    assert "curious" in agent.persona.personality
    assert "David García" in agent.system_prompt
    assert "Always respond in Spanish." in agent.system_prompt


def test_from_yaml_minimal(tmp_path):
    """Minimal YAML with only persona name."""
    yaml_content = (
        "name: agent\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3:4b\n"
        "persona:\n"
        "    full_name: Alice\n"
    )
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    agent = Chatter.from_yaml(str(yaml_file))
    assert agent.persona.full_name == "Alice"
    assert "You are Alice." in agent.system_prompt
