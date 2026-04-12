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
    assert "medieval tavern" in agent.system_prompt
