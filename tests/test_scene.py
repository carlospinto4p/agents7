"""Unit tests for Scene and SceneChatter (no LLM required)."""

from unittest.mock import patch

from agents7.agents.personality import (
    Persona,
    Scene,
)
from agents7.agents.scene import SceneChatter
from agents7.llm import LLMEngine

from helpers import DUMMY_METRICS, ENGINE


# --- Scene ---


def test_scene_to_prompt_contains_fields():
    """Prompt includes all set fields."""
    scene = Scene(
        setting="A medieval tavern at night",
        situation="The tavern is about to close.",
        time_period="Medieval England, 14th century",
        goal="Close the tavern for the night.",
    )
    prompt = scene.to_prompt()
    assert "Setting: A medieval tavern at night" in prompt
    assert "Situation: The tavern is about to close." in prompt
    assert "Time period: Medieval England" in prompt
    assert "Your goal: Close the tavern" in prompt


def test_scene_to_prompt_rules():
    """Rules appear with a MUST follow instruction."""
    scene = Scene(
        rules=[
            "No knowledge of modern technology.",
            "Suspicious of strangers.",
        ],
    )
    prompt = scene.to_prompt()
    assert "Scene Rules" in prompt
    assert "MUST follow" in prompt
    assert "No knowledge of modern technology." in prompt
    assert "Suspicious of strangers." in prompt


def test_scene_to_prompt_other_characters():
    """Other characters are listed."""
    scene = Scene(
        other_characters=[
            "A drunk sailor at the bar",
            "A hooded stranger in the corner",
        ],
    )
    prompt = scene.to_prompt()
    assert "Other Characters Present" in prompt
    assert "A drunk sailor at the bar" in prompt
    assert "A hooded stranger in the corner" in prompt


def test_scene_to_prompt_extra():
    """Extra dict fields appear in prompt."""
    scene = Scene(
        extra={"Weather": "Stormy"},
    )
    prompt = scene.to_prompt()
    assert "Weather: Stormy" in prompt


def test_scene_to_prompt_empty():
    """Empty scene produces minimal section."""
    scene = Scene()
    prompt = scene.to_prompt()
    assert "## Scene" in prompt
    assert "Rules" not in prompt
    assert "Other Characters" not in prompt


def test_scene_from_dict():
    """from_dict creates Scene from a dictionary."""
    data = {
        "setting": "A castle hall",
        "situation": "A feast is underway.",
        "rules": ["Bow before the king."],
        "goal": "Survive the evening.",
    }
    scene = Scene.from_dict(data)
    assert scene.setting == "A castle hall"
    assert scene.situation == "A feast is underway."
    assert scene.rules == ["Bow before the king."]
    assert scene.goal == "Survive the evening."


def test_scene_from_dict_empty():
    """from_dict works with empty dictionary."""
    scene = Scene.from_dict({})
    assert scene.setting == ""
    assert scene.rules == []


# --- SceneChatter ---


def _chat_response(text: str = "Hello!") -> dict:
    return {
        "role": "assistant",
        "response": text,
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }


def test_scene_chatter_creates():
    """SceneChatter initializes correctly."""
    persona = Persona(full_name="Old Thomas")
    scene = Scene(setting="A tavern")
    agent = SceneChatter(
        name="thomas",
        version=1,
        engine=ENGINE,
        system_prompt="",
        persona=persona,
        scene=scene,
    )
    assert agent.name == "thomas"
    assert agent.persona.full_name == "Old Thomas"
    assert agent.scene.setting == "A tavern"
    assert isinstance(agent, SceneChatter)


def test_system_prompt_contains_roleplay_frame():
    """System prompt starts with role-play framing."""
    persona = Persona(full_name="Old Thomas")
    scene = Scene(setting="A tavern")
    agent = SceneChatter(
        name="thomas",
        version=1,
        engine=ENGINE,
        system_prompt="",
        persona=persona,
        scene=scene,
    )
    assert "role-play scenario" in agent.system_prompt


def test_system_prompt_contains_persona_and_scene():
    """System prompt includes both persona and scene."""
    persona = Persona(
        full_name="Old Thomas",
        occupation="tavern keeper",
    )
    scene = Scene(
        setting="A medieval tavern",
        time_period="14th century",
        rules=["No modern technology."],
    )
    agent = SceneChatter(
        name="thomas",
        version=1,
        engine=ENGINE,
        system_prompt="",
        persona=persona,
        scene=scene,
    )
    prompt = agent.system_prompt
    # Scene content
    assert "A medieval tavern" in prompt
    assert "14th century" in prompt
    assert "No modern technology." in prompt
    # Persona content
    assert "You are Old Thomas." in prompt
    assert "tavern keeper" in prompt
    assert "Stay in character" in prompt


def test_scene_before_persona_in_prompt():
    """Scene section appears before persona section."""
    persona = Persona(full_name="Test Character")
    scene = Scene(setting="A dark forest")
    agent = SceneChatter(
        name="test",
        version=1,
        engine=ENGINE,
        system_prompt="",
        persona=persona,
        scene=scene,
    )
    prompt = agent.system_prompt
    scene_pos = prompt.index("## Scene")
    persona_pos = prompt.index("You are Test Character.")
    assert scene_pos < persona_pos


def test_extra_instructions_appended():
    """extra_instructions appear after scene + persona."""
    persona = Persona(full_name="Test")
    scene = Scene(setting="Anywhere")
    agent = SceneChatter(
        name="test",
        version=1,
        engine=ENGINE,
        system_prompt="",
        persona=persona,
        scene=scene,
        extra_instructions="Keep responses under 3 sentences.",
    )
    prompt = agent.system_prompt
    assert "Keep responses under 3 sentences." in prompt
    assert "Extra Instructions" in prompt


def test_system_prompt_overrides_manual():
    """Manual system_prompt is overridden by scene + persona."""
    persona = Persona(full_name="Override Test")
    scene = Scene(setting="Override Scene")
    agent = SceneChatter(
        name="test",
        version=1,
        engine=ENGINE,
        system_prompt="This should be overridden.",
        persona=persona,
        scene=scene,
    )
    assert "Override Test" in agent.system_prompt
    assert "Override Scene" in agent.system_prompt
    assert "This should be overridden." not in agent.system_prompt


@patch.object(LLMEngine, "chat")
def test_chat_uses_scene_prompt(mock_chat):
    """Chat sends the scene-based system prompt."""
    mock_chat.return_value = _chat_response(
        "We're closing soon.",
    )
    persona = Persona(
        full_name="Old Thomas",
        occupation="tavern keeper",
    )
    scene = Scene(
        setting="A medieval tavern at night",
        rules=["No modern technology."],
    )
    agent = SceneChatter(
        name="thomas",
        version=1,
        engine=ENGINE,
        system_prompt="",
        persona=persona,
        scene=scene,
    )
    response = agent.chat("Good evening!")
    assert response.response == "We're closing soon."

    call_args = mock_chat.call_args
    messages = call_args[1]["messages"]
    system_msg = messages[0]["content"]
    assert "Old Thomas" in system_msg
    assert "medieval tavern" in system_msg
    assert "No modern technology." in system_msg


# --- YAML loading ---


def test_from_yaml(tmp_path):
    """Loads SceneChatter from YAML."""
    yaml_content = (
        "name: thomas\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3:4b\n"
        "persona:\n"
        "    full_name: Old Thomas\n"
        "    occupation: tavern keeper\n"
        "    personality:\n"
        "        - gruff\n"
        "        - practical\n"
        "scene:\n"
        "    setting: A medieval tavern at night\n"
        "    situation: The tavern is about to close.\n"
        "    time_period: Medieval England, 14th century\n"
        "    rules:\n"
        "        - No modern technology.\n"
        "        - Suspicious of strangers.\n"
        "    other_characters:\n"
        "        - A drunk sailor\n"
        "    goal: Close the tavern for the night.\n"
        "extra_instructions: Keep responses short.\n"
    )
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    agent = SceneChatter.from_yaml(str(yaml_file))
    assert agent.name == "thomas"
    assert agent.persona.full_name == "Old Thomas"
    assert agent.scene.setting == "A medieval tavern at night"
    assert agent.scene.time_period == ("Medieval England, 14th century")
    assert len(agent.scene.rules) == 2
    assert "A drunk sailor" in agent.scene.other_characters
    assert "role-play scenario" in agent.system_prompt
    assert "Old Thomas" in agent.system_prompt
    assert "medieval tavern" in agent.system_prompt
    assert "Keep responses short." in agent.system_prompt


def test_from_yaml_minimal(tmp_path):
    """Minimal YAML with only persona and empty scene."""
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

    agent = SceneChatter.from_yaml(str(yaml_file))
    assert agent.persona.full_name == "Alice"
    assert agent.scene.setting == ""
    assert "You are Alice." in agent.system_prompt
    assert "role-play scenario" in agent.system_prompt


def test_from_yaml_with_memory(tmp_path):
    """YAML with memory configuration loads correctly."""
    yaml_content = (
        "name: thomas\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3:4b\n"
        "persona:\n"
        "    full_name: Old Thomas\n"
        "scene:\n"
        "    setting: A tavern\n"
        "temperature: 0.9\n"
        "max_history: 20\n"
        "memory:\n"
        "    backend: json\n"
        f"    path: {tmp_path / 'history.json'}\n"
        "conversation_id: test-session\n"
    )
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    agent = SceneChatter.from_yaml(str(yaml_file))
    assert agent.temperature == 0.9
    assert agent.max_history == 20
    assert agent.memory is not None
    assert agent.conversation_id == "test-session"
