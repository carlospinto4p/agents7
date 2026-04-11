"""Integration tests for SceneChatter with qwen2.5:3b.

Uses the demo_scene_chatter.yml config (Old Thomas, a
medieval tavern keeper). Tests that the agent stays in
scene context: responds in character, rejects modern
concepts, references other characters, and pursues its
goal.

LLM responses are non-deterministic, so each test
retries once on failure (fresh agent each attempt).

Expected model: qwen2.5:3b (Ollama).
"""

import pytest

from agents7.agents.scene import SceneChatter

_YAML = "examples/configs/demo_scene_chatter.yml"
_MAX_ATTEMPTS = 2


# --- Helpers ---


def _chat_fresh(prompt: str) -> str:
    """Create a fresh SceneChatter and send one message."""
    agent = SceneChatter.from_yaml(_YAML)
    response = agent.chat(prompt)
    assert response.response, "Response should not be empty"
    return response.response


def _assert_contains_any(
    prompt: str,
    keywords: list[str],
    category: str,
) -> None:
    """Assert the response contains at least one keyword.

    Retries once with a fresh agent.
    """
    last_response = ""
    for _ in range(_MAX_ATTEMPTS):
        text = _chat_fresh(prompt)
        lower = text.lower()
        if any(kw in lower for kw in keywords):
            return
        last_response = text

    pytest.fail(
        f"{category}: expected any of {keywords} in "
        f"response, got: {last_response[:200]}"
    )


def _assert_excludes_all(
    prompt: str,
    bad_words: list[str],
    category: str,
) -> None:
    """Assert the response contains none of the bad words.

    Retries once with a fresh agent.
    """
    last_bad = None
    last_response = ""
    for _ in range(_MAX_ATTEMPTS):
        text = _chat_fresh(prompt)
        lower = text.lower()
        found = next((w for w in bad_words if w in lower), None)
        if found is None:
            return
        last_bad = found
        last_response = text

    pytest.fail(
        f"{category}: response contains '{last_bad}' "
        f"after {_MAX_ATTEMPTS} attempts: "
        f"{last_response[:200]}"
    )


# --- In-character responses ---


@pytest.mark.parametrize(
    "prompt",
    [
        "Who are you?",
        "What do you have to eat?",
    ],
)
def test_stays_in_character(prompt: str) -> None:
    """Agent should respond as Old Thomas the tavern keeper.

    Look for tavern/medieval vocabulary in the response.
    """
    _assert_contains_any(
        prompt,
        [
            "tavern",
            "ale",
            "stew",
            "drink",
            "thomas",
            "inn",
            "bread",
            "penny",
            "pennies",
            "mutton",
        ],
        "In-character",
    )


# --- Rejects modern concepts ---


@pytest.mark.parametrize(
    "prompt",
    [
        "Can I charge my phone here?",
        "Do you have WiFi?",
    ],
)
def test_rejects_modern_concepts(prompt: str) -> None:
    """Agent should not understand modern technology.

    Should not produce a helpful modern-context reply.
    """
    _assert_excludes_all(
        prompt,
        [
            "wifi",
            "wi-fi",
            "outlet",
            "charger",
            "password",
            "internet",
            "smartphone",
            "battery",
        ],
        "Modern concept rejection",
    )


# --- Awareness of other characters ---


def test_references_other_characters() -> None:
    """Agent should mention patrons defined in the scene."""
    _assert_contains_any(
        "Who else is here tonight?",
        [
            "sailor",
            "drunk",
            "stranger",
            "hooded",
            "corner",
            "bar",
        ],
        "Other character awareness",
    )


# --- Goal pursuit ---


def test_pursues_closing_goal() -> None:
    """Agent should try to close the tavern when asked."""
    _assert_contains_any(
        "I'd like to stay a bit longer.",
        [
            "clos",
            "leave",
            "go",
            "late",
            "time",
            "storm",
            "last call",
            "finish",
            "night",
            "home",
        ],
        "Goal pursuit",
    )
