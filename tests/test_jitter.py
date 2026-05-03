"""Unit tests for typo jitter."""

import random
from unittest.mock import patch


from agents7.agents.jitter import _inject_typo, apply_typos
from agents7.agents.chatter import Chatter
from agents7.llm import LLMEngine

from helpers import DUMMY_METRICS, ENGINE


# --- _inject_typo word-length boundary ---


def test_inject_typo_4char_swap_modifies():
    """swap on a 4-char word produces a different string."""
    rng = random.Random(0)
    rng.choice = lambda _: "swap"  # force swap path
    # randint(1, 2) will pick index 1 or 2 — word changes
    result = _inject_typo("test", 0, 1, rng)
    assert len(result) == 4
    assert result != "test"


def test_inject_typo_4char_drop_no_change():
    """drop on a 4-char word is skipped (len not > 4)."""

    class _ForcedRng:
        def choice(self, seq):
            return "drop"

        def randint(self, a, b):
            return a  # would pick index 1 if drop fired

    result = _inject_typo("test", 0, 1, _ForcedRng())
    assert result == "test"


def test_inject_typo_5char_drop_fires():
    """drop on a 5-char word does modify the word."""

    class _ForcedRng:
        def choice(self, seq):
            return "drop"

        def randint(self, a, b):
            return 1

    result = _inject_typo("hello", 0, 1, _ForcedRng())
    assert len(result) == 4
    assert result != "hello"


# --- apply_typos ---


def test_zero_rate_no_change():
    """rate=0.0 returns text unchanged."""
    text = "Hello world this is a test"
    assert apply_typos(text, 0.0) == text


def test_negative_rate_no_change():
    """Negative rate returns text unchanged."""
    text = "Hello world"
    assert apply_typos(text, -0.5) == text


def test_high_rate_modifies_text():
    """rate=1.0 modifies at least some words."""
    rng = random.Random(42)
    text = "Hello world this is a longer test sentence"
    result = apply_typos(text, 1.0, rng=rng)
    assert result != text


def test_short_words_untouched():
    """Words shorter than 4 chars are never modified."""
    rng = random.Random(42)
    text = "I am a ok to go"
    result = apply_typos(text, 1.0, rng=rng)
    # All words are <= 3 chars, nothing should change
    assert result == text


def test_deterministic_with_seed():
    """Same seed produces same typos."""
    text = "This is a deterministic test sentence"
    r1 = apply_typos(text, 0.5, rng=random.Random(123))
    r2 = apply_typos(text, 0.5, rng=random.Random(123))
    assert r1 == r2


def test_punctuation_preserved():
    """Trailing punctuation stays after typo."""
    rng = random.Random(42)
    text = "Hello, world!"
    result = apply_typos(text, 1.0, rng=rng)
    # Punctuation should still be at the end
    assert result.endswith("!") or result.endswith(",")


def test_word_count_preserved():
    """Number of words stays the same."""
    rng = random.Random(42)
    text = "Hello world this is testing"
    result = apply_typos(text, 0.5, rng=rng)
    assert len(result.split(" ")) == len(text.split(" "))


# --- Chatter integration ---


def _chat_response(text: str = "Hello world!") -> dict:
    return {
        "role": "assistant",
        "response": text,
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }


@patch.object(LLMEngine, "chat")
def test_chatter_typo_rate_zero_unchanged(mock_chat):
    """typo_rate=0.0 returns LLM response unchanged."""
    mock_chat.return_value = _chat_response("This is a perfect response")
    agent = Chatter(
        name="test",
        version=1,
        engine=ENGINE,
        system_prompt="Test.",
        typo_rate=0.0,
    )
    response = agent.chat("Hi")
    assert response.response == "This is a perfect response"


@patch.object(LLMEngine, "chat")
def test_chatter_typo_rate_applies(mock_chat):
    """typo_rate > 0 modifies responses."""
    long_text = (
        "This is a longer response with several words "
        "that should have some typos introduced into them"
    )
    mock_chat.return_value = _chat_response(long_text)
    agent = Chatter(
        name="test",
        version=1,
        engine=ENGINE,
        system_prompt="Test.",
        typo_rate=1.0,
    )
    response = agent.chat("Hi")
    assert response.response != long_text


@patch.object(LLMEngine, "chat")
def test_chatter_typo_in_history(mock_chat):
    """Typo'd text is stored in history."""
    long_text = (
        "This is a longer response with several words "
        "that should definitely have typos"
    )
    mock_chat.return_value = _chat_response(long_text)
    agent = Chatter(
        name="test",
        version=1,
        engine=ENGINE,
        system_prompt="Test.",
        typo_rate=1.0,
    )
    response = agent.chat("Hi")
    assistant_msg = agent.history[-1]["content"]
    assert assistant_msg == response.response
