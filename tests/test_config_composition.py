"""Unit tests for character / chat_goal config composition.

A role-play agent has two independent axes: who the character is
and what it is doing. They live in separate files so either can
be swapped without touching the other — rename David to Marta
and every goal still reads right; drop David into a different
conversation and his voice comes with him.

These tests cover the composition itself. The `$character`
placeholder that makes a goal file character-agnostic is tested
in `test_chatter.py`.
"""

import pytest

from agents7.agents.config_loader import (
    _compose,
    resolve_config,
)
from agents7.exceptions import AgentConfigurationError


CHARACTER = """\
persona:
    full_name: David
    language: Spanish
    speaking_style: Short chat bursts.
extra_instructions: |
    How he writes.
"""

CHAT_GOAL = """\
scene:
    setting: A café in Malasaña.
    rules:
        - Never give your surname.
can_open: true
extra_instructions: |
    How he plays.
"""

AGENT = """\
character: characters/david.yml
chat_goal: goals/cafe_chat.yml
name: david-chat
version: 1
engine:
    provider: ollama
    model: qwen3
temperature: 0.85
"""


@pytest.fixture
def split(tmp_path):
    """A config split across a character and a goal file."""
    (tmp_path / "characters").mkdir()
    (tmp_path / "goals").mkdir()
    (tmp_path / "characters" / "david.yml").write_text(CHARACTER)
    (tmp_path / "goals" / "cafe_chat.yml").write_text(CHAT_GOAL)
    config = tmp_path / "agent.yml"
    config.write_text(AGENT)
    return config


# ── _compose ───────────────────────────────────────────────


def test_compose_merges_like_deep_merge():
    """Ordinary keys behave exactly as inheritance does."""
    base = {"engine": {"provider": "ollama", "model": "qwen3"}}
    override = {"engine": {"model": "llama3"}}
    result = _compose(base, override)
    assert result["engine"] == {"provider": "ollama", "model": "llama3"}


def test_compose_joins_the_text_blocks():
    """Both axes get to speak; neither silently wins.

    An override that replaced `extra_instructions` would drop
    the character's voice the moment a goal declared any of its
    own — the failure this rule exists to prevent.
    """
    base = {"extra_instructions": "How he writes."}
    override = {"extra_instructions": "How he plays."}
    assert (
        _compose(base, override)["extra_instructions"]
        == "How he writes.\n\nHow he plays."
    )


def test_compose_joins_opening_instructions_too():
    """The opening block accumulates on the same rule."""
    base = {"opening_instructions": "In Spanish."}
    override = {"opening_instructions": "One line."}
    assert (
        _compose(base, override)["opening_instructions"]
        == "In Spanish.\n\nOne line."
    )


def test_compose_keeps_a_lone_text_block():
    """One side declaring nothing leaves the other untouched."""
    assert _compose({"extra_instructions": "Only this."}, {}) == {
        "extra_instructions": "Only this."
    }
    assert _compose({}, {"extra_instructions": "Only this."}) == {
        "extra_instructions": "Only this."
    }


def test_compose_ignores_a_blank_text_block():
    """An empty block is nothing to join, not a blank line."""
    result = _compose(
        {"extra_instructions": "Real text."},
        {"extra_instructions": "   \n"},
    )
    assert result["extra_instructions"] == "Real text."


def test_compose_does_not_mutate():
    """Inputs survive the merge."""
    base = {"extra_instructions": "A"}
    override = {"extra_instructions": "B"}
    _compose(base, override)
    assert base == {"extra_instructions": "A"}
    assert override == {"extra_instructions": "B"}


# ── resolve_config ─────────────────────────────────────────


def test_both_fragments_are_folded_in(split):
    """The persona and the scene arrive from separate files."""
    config = resolve_config(str(split))
    assert config["persona"]["full_name"] == "David"
    assert config["scene"]["setting"] == "A café in Malasaña."


def test_the_composition_keys_are_gone(split):
    """They are references, not configuration."""
    config = resolve_config(str(split))
    assert "character" not in config
    assert "chat_goal" not in config


def test_the_agent_config_keeps_its_own_keys(split):
    """Composition adds to the config, it does not replace it."""
    config = resolve_config(str(split))
    assert config["name"] == "david-chat"
    assert config["temperature"] == 0.85


def test_a_goal_key_reaches_the_config(split):
    """Opening behaviour belongs to the goal, not the character."""
    assert resolve_config(str(split))["can_open"] is True


def test_the_two_instruction_blocks_are_joined_in_order(split):
    """Character first, then goal — a fixed order, not a guess."""
    text = resolve_config(str(split))["extra_instructions"]
    assert text.index("How he writes.") < text.index("How he plays.")


def test_the_agent_config_wins_a_clash(tmp_path):
    """A deployment can pin what the fragments left open."""
    (tmp_path / "who.yml").write_text(
        "persona:\n    full_name: David\ntemperature: 0.5\n"
    )
    config = tmp_path / "agent.yml"
    config.write_text(
        "character: who.yml\nname: a\nversion: 1\ntemperature: 0.9\n"
    )
    assert resolve_config(str(config))["temperature"] == 0.9


def test_a_character_alone_composes(tmp_path):
    """Either axis may be used without the other."""
    (tmp_path / "who.yml").write_text("persona:\n    full_name: David\n")
    config = tmp_path / "agent.yml"
    config.write_text("character: who.yml\nname: a\nversion: 1\n")
    assert resolve_config(str(config))["persona"]["full_name"] == "David"


def test_a_goal_alone_composes(tmp_path):
    """A goal with no character is still a valid fragment."""
    (tmp_path / "what.yml").write_text("scene:\n    setting: A tavern.\n")
    config = tmp_path / "agent.yml"
    config.write_text("chat_goal: what.yml\nname: a\nversion: 1\n")
    assert resolve_config(str(config))["scene"]["setting"] == "A tavern."


def test_a_config_without_fragments_is_untouched(tmp_path):
    """The whole mechanism is opt-in."""
    config = tmp_path / "agent.yml"
    config.write_text("name: a\nversion: 1\nextra_instructions: One.\n")
    assert resolve_config(str(config)) == {
        "name": "a",
        "version": 1,
        "extra_instructions": "One.",
    }


def test_fragments_compose_on_top_of_inheritance(tmp_path):
    """`base_config` and the two axes work together."""
    (tmp_path / "base.yml").write_text(
        "engine:\n    provider: ollama\n    model: qwen3\n"
    )
    (tmp_path / "who.yml").write_text("persona:\n    full_name: David\n")
    config = tmp_path / "agent.yml"
    config.write_text(
        "base_config: base.yml\ncharacter: who.yml\nname: a\nversion: 1\n"
    )
    result = resolve_config(str(config))
    assert result["engine"]["model"] == "qwen3"
    assert result["persona"]["full_name"] == "David"


# ── The split is enforced, not merely suggested ────────────


def test_a_character_may_not_declare_a_scene(tmp_path):
    """A character carrying a scene is the entanglement again.

    Left unchecked, the axis that was split apart grows back
    into one file and the swap stops working — silently, since
    the config still loads.
    """
    (tmp_path / "who.yml").write_text(
        "persona:\n    full_name: David\nscene:\n    setting: A bar.\n"
    )
    config = tmp_path / "agent.yml"
    config.write_text("character: who.yml\nname: a\nversion: 1\n")
    with pytest.raises(AgentConfigurationError, match="declares `scene`"):
        resolve_config(str(config))


def test_a_goal_may_not_declare_a_persona(tmp_path):
    """The same check, pointing the other way."""
    (tmp_path / "what.yml").write_text(
        "scene:\n    setting: A bar.\npersona:\n    full_name: David\n"
    )
    config = tmp_path / "agent.yml"
    config.write_text("chat_goal: what.yml\nname: a\nversion: 1\n")
    with pytest.raises(AgentConfigurationError, match="declares `persona`"):
        resolve_config(str(config))


def test_a_fragment_may_not_nest_another_fragment(tmp_path):
    """One level, so a config is readable from the config."""
    (tmp_path / "inner.yml").write_text("persona:\n    full_name: X\n")
    (tmp_path / "who.yml").write_text("character: inner.yml\n")
    config = tmp_path / "agent.yml"
    config.write_text("character: who.yml\nname: a\nversion: 1\n")
    with pytest.raises(AgentConfigurationError, match="not a config of"):
        resolve_config(str(config))


def test_a_fragment_may_not_inherit(tmp_path):
    """`base_config` inside a fragment is the same nesting."""
    (tmp_path / "base.yml").write_text("name: b\n")
    (tmp_path / "who.yml").write_text("base_config: base.yml\n")
    config = tmp_path / "agent.yml"
    config.write_text("character: who.yml\nname: a\nversion: 1\n")
    with pytest.raises(AgentConfigurationError, match="not a config of"):
        resolve_config(str(config))


def test_an_absolute_fragment_path_is_rejected(tmp_path):
    """Same reasoning as `base_config`: no arbitrary files."""
    (tmp_path / "who.yml").write_text("persona:\n    full_name: X\n")
    config = tmp_path / "agent.yml"
    config.write_text(
        f"character: {tmp_path / 'who.yml'}\nname: a\nversion: 1\n"
    )
    with pytest.raises(AgentConfigurationError, match="must be relative"):
        resolve_config(str(config))


def test_a_missing_fragment_says_so(tmp_path):
    """A typo in a path fails where it happened."""
    config = tmp_path / "agent.yml"
    config.write_text("character: nope.yml\nname: a\nversion: 1\n")
    with pytest.raises(AgentConfigurationError, match="does not exist"):
        resolve_config(str(config))


def test_a_fragment_must_be_a_path(tmp_path):
    """An inline mapping is the mistake this catches."""
    config = tmp_path / "agent.yml"
    config.write_text(
        "character:\n    persona:\n        full_name: X\n"
        "name: a\nversion: 1\n"
    )
    with pytest.raises(AgentConfigurationError, match="must be a path"):
        resolve_config(str(config))
