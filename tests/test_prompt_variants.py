"""Unit tests for `prompt_variants`: one text drawn per conversation."""

import pytest
import yaml

from agents7.agents.chatter import Chatter
from agents7.agents.memory import MemoryConfig, create_memory
from agents7.eval.chat_review import (
    ConversationRecorder,
    load_recording,
    save_recording,
)
from agents7.eval.style import ConversationScript, StyleContract
from agents7.eval.style_history import fingerprint_run
from agents7.exceptions import AgentConfigurationError, PromptParameterError

from helpers import make_chatter, make_labeler

TEXTS = ("uno", "dos", "tres")


def _chatter(**kwargs) -> Chatter:
    return make_chatter(
        system_prompt="Ejemplo: $reply",
        prompt_variants={"reply": TEXTS},
        **kwargs,
    )


def _memory(tmp_path):
    return create_memory(
        MemoryConfig(backend="sqlite", path=str(tmp_path / "m.db"))
    )


# ── Drawing ────────────────────────────────────────────────


def test_the_prompt_shows_exactly_one_of_the_texts():
    agent = _chatter()
    shown = [t for t in TEXTS if t in agent.resolved_system_prompt]
    assert shown == [agent.variant_values["reply"]]


def test_separate_agents_draw_different_texts():
    """Fifty builds all drawing one text would be a fixed prompt.

    With three texts that happens by chance with probability
    3 / 3**50, so a single distinct value means no draw happened.
    """
    drawn = {_chatter().variant_values["reply"] for _ in range(50)}
    assert len(drawn) > 1


def test_the_same_seed_draws_the_same_texts():
    agent = _chatter()
    agent.draw_variants(seed="conv-7")
    first = agent.variant_values
    agent.draw_variants(seed="conv-7")
    assert agent.variant_values == first


def test_an_agent_without_variants_draws_nothing():
    assert make_chatter().variant_values == {}


def test_other_agent_types_take_variants_too():
    agent = make_labeler(
        system_prompt="Ejemplo: $reply",
        prompt_variants={"reply": TEXTS},
    )
    assert agent.variant_values["reply"] in agent.resolved_system_prompt


# ── Validation ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "variants",
    [
        {"reply": ("uno",)},
        {"reply": ("uno", "uno")},
        {"reply": ("uno", "  ")},
        {"reply": "uno"},
        ["uno", "dos"],
    ],
    ids=["one text", "repeated", "blank", "string", "not a dict"],
)
def test_unusable_variants_are_rejected(variants):
    with pytest.raises(PromptParameterError):
        make_chatter(prompt_variants=variants)


def test_a_name_cannot_be_a_variant_and_a_param():
    with pytest.raises(PromptParameterError, match="reply"):
        make_chatter(
            prompt_params={"reply": "fijo"},
            prompt_variants={"reply": TEXTS},
        )


# ── Conversations ──────────────────────────────────────────


def test_reset_without_memory_draws_again():
    agent = _chatter()
    drawn = set()
    for _ in range(50):
        agent.reset()
        drawn.add(agent.variant_values["reply"])
    assert len(drawn) > 1


def test_a_conversation_in_memory_keeps_its_draw_when_rebuilt(tmp_path):
    """One process per turn builds the agent every turn.

    Each build must show what the first one showed, or the example
    changes under the model mid-conversation.
    """
    memory = _memory(tmp_path)
    for n in range(20):
        ids = f"conv-{n}"
        first = _chatter(memory=memory, conversation_id=ids)
        again = _chatter(memory=memory, conversation_id=ids)
        assert again.variant_values == first.variant_values


def test_conversations_in_memory_still_differ_from_each_other(tmp_path):
    memory = _memory(tmp_path)
    drawn = {
        _chatter(memory=memory, conversation_id=f"conv-{n}").variant_values[
            "reply"
        ]
        for n in range(50)
    }
    assert len(drawn) > 1


# ── Identity ───────────────────────────────────────────────


def _two_different_draws() -> tuple[Chatter, Chatter]:
    first, second = _chatter(), _chatter()
    for n in range(100):
        second.draw_variants(seed=str(n))
        if second.variant_values != first.variant_values:
            return first, second
    raise AssertionError("never drew differently")


def test_the_identity_ignores_what_was_drawn():
    first, second = _two_different_draws()
    assert first.resolved_system_prompt != second.resolved_system_prompt
    assert first.system_prompt_identity == second.system_prompt_identity


def test_the_identity_lists_every_text():
    identity = _chatter().system_prompt_identity
    assert "$reply" in identity
    assert all(t in identity for t in TEXTS)


def test_without_variants_the_identity_is_the_resolved_prompt():
    """So every fingerprint stored before variants still reproduces."""
    agent = make_chatter(
        system_prompt="Hola $nick", prompt_params={"nick": "ella"}
    )
    assert agent.system_prompt_identity == agent.resolved_system_prompt


def test_runs_that_drew_differently_share_a_fingerprint():
    contract = StyleContract(
        rules=(),
        scripts=(ConversationScript(id="cold", lines=("bien",)),),
    )
    first, second = _two_different_draws()
    assert fingerprint_run(contract, first) == fingerprint_run(
        contract, second
    )


# ── YAML ───────────────────────────────────────────────────


def _write_config(tmp_path, variants) -> str:
    config = {
        "name": "variant-chatter",
        "version": 1,
        "engine": {"provider": "ollama", "model": "qwen3"},
        "system_prompt": "Ejemplo: $reply",
        "prompt_variants": variants,
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


def test_yaml_declares_variants(tmp_path):
    agent = Chatter.from_yaml(_write_config(tmp_path, {"reply": list(TEXTS)}))
    assert agent.prompt_variants == {"reply": TEXTS}
    assert agent.variant_values["reply"] in TEXTS


def test_yaml_rejects_a_variant_that_is_not_a_list(tmp_path):
    with pytest.raises(AgentConfigurationError, match="reply"):
        Chatter.from_yaml(_write_config(tmp_path, {"reply": "uno"}))


# ── Recording ──────────────────────────────────────────────


def test_a_recording_says_which_texts_its_conversation_saw(tmp_path):
    """The texts of the agent that talks, which is the recorder's copy."""
    rec = ConversationRecorder(
        _chatter(), "cold", tmp_path / "c.json", agent_version="0.0.1"
    )
    assert rec.recording.variants == rec.agent.variant_values
    assert rec.recording.variants["reply"] in TEXTS


def test_recorded_variants_survive_a_round_trip(tmp_path):
    rec = ConversationRecorder(
        _chatter(), "cold", tmp_path / "c.json", agent_version="0.0.1"
    )
    save_recording(rec.recording, tmp_path / "c.json")
    assert load_recording(tmp_path / "c.json").variants == (
        rec.agent.variant_values
    )
