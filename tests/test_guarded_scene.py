"""Unit tests for GuardedSceneChatter (no LLM required)."""

import json
from unittest.mock import patch

from agents7.agents.context_guard import (
    ContextGuard,
    GuardReason,
    GuardVerdict,
)
from agents7.agents.guarded_scene import GuardedSceneChatter
from agents7.agents.personality import Persona, Scene
from agents7.llm import LLMEngine
from agents7.models import AgentResponse

from helpers import DUMMY_METRICS, ENGINE, chat_response_dict, make_guard


PERSONA = Persona(
    full_name="Old Thomas",
    occupation="tavern keeper",
)
SCENE = Scene(
    setting="A medieval tavern at night",
    situation="The tavern is about to close.",
    time_period="Medieval England, 14th century",
    rules=["No knowledge of anything modern."],
    goal="Close the tavern and get everyone out.",
)
PERSONA_SUMMARY = "Old Thomas, medieval tavern keeper"


def make_guarded_scene(**kwargs) -> GuardedSceneChatter:
    defaults = {
        "name": "guarded-thomas",
        "version": 1,
        "engine": ENGINE,
        "system_prompt": "",
        "persona": PERSONA,
        "scene": SCENE,
        "guard": make_guard(persona_summary=PERSONA_SUMMARY),
    }
    defaults.update(kwargs)
    return GuardedSceneChatter(**defaults)


def _verdict_response(
    value: bool,
    reason: str = "IN_CHARACTER",
    fix: str | None = None,
) -> AgentResponse:
    verdict = GuardVerdict(
        value=value,
        reason=GuardReason(GuardReason[reason].value),
        fix=fix,
    )
    return AgentResponse(
        response=json.dumps(verdict.model_dump()),
        model=verdict,
        thinking=None,
        metrics=DUMMY_METRICS,
    )


# --- Prompt generation ---


def test_system_prompt_includes_scene_and_persona():
    agent = make_guarded_scene()
    prompt = agent.system_prompt
    assert "role-play" in prompt.lower()
    assert "medieval tavern" in prompt
    assert "Old Thomas" in prompt
    assert "MUST follow" in prompt
    assert "No knowledge of anything modern" in prompt


def test_extra_instructions_appended():
    agent = make_guarded_scene(extra_instructions="Speak slowly.")
    assert "Extra Instructions" in agent.system_prompt
    assert "Speak slowly." in agent.system_prompt


def test_empty_scene_still_generates_prompt():
    agent = make_guarded_scene(scene=Scene())
    assert "Old Thomas" in agent.system_prompt
    assert "role-play" in agent.system_prompt.lower()


def test_system_prompt_includes_situation():
    """Scene situation appears in the auto-generated system prompt."""
    agent = make_guarded_scene()
    assert "The tavern is about to close" in agent.system_prompt


def test_system_prompt_includes_time_period():
    """Scene time_period appears in the auto-generated system prompt."""
    agent = make_guarded_scene()
    assert "Medieval England" in agent.system_prompt


def test_system_prompt_includes_goal():
    """Scene goal appears in the auto-generated system prompt."""
    agent = make_guarded_scene()
    assert "Close the tavern" in agent.system_prompt


def test_system_prompt_includes_occupation():
    """Persona occupation appears in the auto-generated system prompt."""
    agent = make_guarded_scene()
    assert "tavern keeper" in agent.system_prompt


# --- Guard integration ---


@patch.object(ContextGuard, "act")
@patch.object(LLMEngine, "chat")
def test_chat_accepts_when_guard_passes(mock_chat, mock_guard):
    """When the guard accepts, response passes through."""
    mock_chat.return_value = chat_response_dict("Evenin', traveller.")
    mock_guard.return_value = _verdict_response(True, "IN_CHARACTER")

    agent = make_guarded_scene()
    response = agent.chat("Good evening!")

    assert response.response == "Evenin', traveller."
    assert agent.history[-1]["content"] == "Evenin', traveller."


@patch.object(ContextGuard, "act")
@patch.object(LLMEngine, "chat")
def test_chat_uses_fix_when_retries_exhausted(mock_chat, mock_guard):
    """Falls back to the guard's fix after max retries."""
    mock_chat.return_value = chat_response_dict("I'll charge your phone.")
    mock_guard.return_value = _verdict_response(
        False,
        "BROKE_CHARACTER",
        fix="We have no such thing here, stranger.",
    )

    agent = make_guarded_scene(max_guard_retries=1)
    response = agent.chat("Can I charge my phone?")

    assert response.response == "We have no such thing here, stranger."


# --- Guard context ---


def test_derive_persona_summary_includes_scene_context():
    """_derive_persona_summary includes scene setting and rules."""
    from agents7.agents.config_loader import _derive_persona_summary

    summary = _derive_persona_summary(PERSONA, SCENE)
    assert "medieval tavern" in summary.lower()
    assert "No knowledge of anything modern" in summary


def test_derive_persona_summary_without_scene_has_no_scene_keys():
    """_derive_persona_summary without scene omits scene keys."""
    from agents7.agents.config_loader import _derive_persona_summary

    summary = _derive_persona_summary(PERSONA)
    assert "Setting:" not in summary
    assert "Scene rules:" not in summary


# --- Guard retry flow ---


@patch.object(ContextGuard, "act")
@patch.object(LLMEngine, "chat")
def test_guard_retry_succeeds_early(mock_chat, mock_guard):
    """Guard accepts on first retry — fix is never used."""
    mock_chat.return_value = chat_response_dict("I'll get ye some mead.")
    reject = _verdict_response(
        False,
        "BROKE_CHARACTER",
        fix="We have no such device, stranger.",
    )
    accept = _verdict_response(True, "IN_CHARACTER")
    mock_guard.side_effect = [reject, accept]

    agent = make_guarded_scene(max_guard_retries=2)
    response = agent.chat("Can I charge my phone?")

    assert mock_guard.call_count == 2
    assert response.response != "We have no such device, stranger."
    assert agent.history[-1]["content"] == response.response


@patch.object(ContextGuard, "act")
@patch.object(LLMEngine, "chat")
def test_guard_input_has_correct_structure(mock_chat, mock_guard):
    """Guard act receives user_message, assistant_response, recent_history."""
    mock_chat.return_value = chat_response_dict("Evenin', traveller.")
    mock_guard.return_value = _verdict_response(True, "IN_CHARACTER")

    agent = make_guarded_scene()
    agent.chat("Good evening!")

    call_input = mock_guard.call_args[0][0]
    assert call_input["user_message"] == "Good evening!"
    assert call_input["assistant_response"] == "Evenin', traveller."
    assert "recent_history" in call_input
    assert isinstance(call_input["recent_history"], list)


@patch.object(ContextGuard, "act")
@patch.object(LLMEngine, "chat")
def test_guard_called_once_on_immediate_accept(mock_chat, mock_guard):
    """Guard is called exactly once when first check passes."""
    mock_chat.return_value = chat_response_dict("Aye, what can I do for ye?")
    mock_guard.return_value = _verdict_response(True, "IN_CHARACTER")

    agent = make_guarded_scene()
    agent.chat("Hello!")

    assert mock_guard.call_count == 1


# --- YAML loading ---


def test_from_yaml_builds_guarded_scene(tmp_path):
    yaml_text = """
name: guarded-thomas
version: 1
engine:
  provider: ollama
  model: qwen3:4b
persona:
  full_name: Old Thomas
  occupation: tavern keeper
scene:
  setting: A medieval tavern at night
  rules:
    - No knowledge of modern things.
guard:
  engine:
    provider: ollama
    model: qwen3:4b
max_guard_retries: 1
"""
    path = tmp_path / "guarded_scene.yml"
    path.write_text(yaml_text, encoding="utf-8")

    agent = GuardedSceneChatter.from_yaml(str(path))
    assert isinstance(agent, GuardedSceneChatter)
    assert agent.scene.setting == "A medieval tavern at night"
    assert agent.persona.full_name == "Old Thomas"
    assert agent.max_guard_retries == 1
    # Guard gets scene context via YAML loader
    assert "medieval tavern" in agent.guard.persona_summary.lower()
    assert "No knowledge of modern things" in agent.guard.persona_summary
    assert "medieval tavern" in agent.system_prompt


def test_from_yaml_persona_summary_includes_situation(tmp_path):
    """Derived persona_summary includes situation when scene has one."""
    yaml_text = """
name: guarded-thomas
version: 1
engine:
  provider: ollama
  model: qwen3:4b
persona:
  full_name: Old Thomas
  occupation: tavern keeper
scene:
  setting: A medieval tavern at night
  situation: The last customer is leaving.
  rules:
    - No knowledge of modern things.
guard:
  engine:
    provider: ollama
    model: qwen3:4b
"""
    path = tmp_path / "guarded_scene.yml"
    path.write_text(yaml_text, encoding="utf-8")

    agent = GuardedSceneChatter.from_yaml(str(path))
    summary = agent.guard.persona_summary
    assert "Situation: The last customer is leaving." in summary
    assert "Setting: A medieval tavern at night" in summary
    assert "Scene rules:" in summary


def test_from_yaml_persona_summary_explicit_override(tmp_path):
    """Explicit persona_summary overrides the persona description.

    Scene context is still appended so the guard can enforce
    scene constraints — the explicit summary replaces only the
    auto-derived persona description, not the scene rules.
    """
    yaml_text = """
name: guarded-thomas
version: 1
engine:
  provider: ollama
  model: qwen3:4b
persona:
  full_name: Old Thomas
  occupation: tavern keeper
scene:
  setting: A medieval tavern at night
guard:
  engine:
    provider: ollama
    model: qwen3:4b
  persona_summary: Custom summary for guard.
"""
    path = tmp_path / "guarded_scene.yml"
    path.write_text(yaml_text, encoding="utf-8")

    agent = GuardedSceneChatter.from_yaml(str(path))
    # Explicit persona description preserved.
    assert "Custom summary for guard." in agent.guard.persona_summary
    # Scene context still reaches the guard.
    assert "medieval tavern" in agent.guard.persona_summary.lower()


def test_python_construction_enriches_guard_with_scene():
    """Direct Python construction folds scene into the guard.

    Regression: previously the scene reached the guard only via
    the YAML loader, so a Python-built GuardedSceneChatter had a
    guard that could not see the scene rules its prompt refers
    to.
    """
    agent = make_guarded_scene()
    summary = agent.guard.persona_summary
    # Persona portion preserved, scene portion appended.
    assert PERSONA_SUMMARY in summary
    assert "Setting: A medieval tavern at night" in summary
    assert "No knowledge of anything modern" in summary
    # The regenerated guard prompt embeds the scene context too.
    assert "No knowledge of anything modern" in agent.guard.system_prompt


def test_python_construction_scene_enrichment_idempotent():
    """Re-using an already-enriched guard does not double-add."""
    agent = make_guarded_scene()
    # Build a second agent from the first's (already enriched)
    # guard — the scene block must not be appended twice.
    agent2 = make_guarded_scene(guard=agent.guard)
    assert agent2.guard.persona_summary.count("Setting: A medieval") == 1


def test_from_yaml_persona_summary_no_scene_omits_scene_keys(tmp_path):
    """Without a scene section, persona_summary has no scene keys."""
    yaml_text = """
name: guarded-thomas
version: 1
engine:
  provider: ollama
  model: qwen3:4b
persona:
  full_name: Old Thomas
  occupation: tavern keeper
guard:
  engine:
    provider: ollama
    model: qwen3:4b
"""
    path = tmp_path / "guarded_scene.yml"
    path.write_text(yaml_text, encoding="utf-8")

    agent = GuardedSceneChatter.from_yaml(str(path))
    summary = agent.guard.persona_summary
    assert "Old Thomas" in summary
    assert "Setting:" not in summary
    assert "Situation:" not in summary
    assert "Scene rules:" not in summary
