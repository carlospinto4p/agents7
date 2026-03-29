"""Unit tests for the Anthropic provider (no API calls)."""

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# --- Mock anthropic module ---


def _mock_response(
    text: str = "Hello!",
    thinking: str = "",
    input_tokens: int = 10,
    output_tokens: int = 5,
    role: str = "assistant",
) -> SimpleNamespace:
    """Build a mock Anthropic Messages response."""
    blocks = [
        SimpleNamespace(type="text", text=text),
    ]
    if thinking:
        blocks.insert(
            0,
            SimpleNamespace(
                type="thinking",
                thinking=thinking,
            ),
        )
    return SimpleNamespace(
        content=blocks,
        role=role,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


# --- Provider construction ---


def test_provider_requires_anthropic_package():
    """Raises ImportError when anthropic not installed."""
    with patch.dict("sys.modules", {"anthropic": None}):
        with pytest.raises(ImportError, match="anthropic"):
            from importlib import reload

            import agents7.llm.anthropic as mod

            reload(mod)
            mod._anthropic = None
            mod._require_anthropic()


# --- _build_params ---


def test_build_params_extracts_system():
    """System message is extracted from messages list."""
    pytest.importorskip("anthropic")

    from agents7.llm.anthropic import AnthropicProvider

    provider = AnthropicProvider("claude-test")
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    params = provider._build_params(
        messages,
        0.5,
        -1,
        False,
    )
    assert params["system"] == "You are helpful."
    assert len(params["messages"]) == 1
    assert params["messages"][0]["role"] == "user"


def test_build_params_default_max_tokens():
    """Uses default_max_tokens when max_tokens=-1."""
    pytest.importorskip("anthropic")

    from agents7.llm.anthropic import AnthropicProvider

    provider = AnthropicProvider(
        "claude-test",
        default_max_tokens=2048,
    )
    params = provider._build_params(
        [{"role": "user", "content": "Hi"}],
        0.7,
        -1,
        False,
    )
    assert params["max_tokens"] == 2048


def test_build_params_explicit_max_tokens():
    """Explicit max_tokens overrides default."""
    pytest.importorskip("anthropic")

    from agents7.llm.anthropic import AnthropicProvider

    provider = AnthropicProvider("claude-test")
    params = provider._build_params(
        [{"role": "user", "content": "Hi"}],
        0.7,
        512,
        False,
    )
    assert params["max_tokens"] == 512


def test_build_params_think_enabled():
    """think=True adds thinking config."""
    pytest.importorskip("anthropic")

    from agents7.llm.anthropic import AnthropicProvider

    provider = AnthropicProvider("claude-test")
    params = provider._build_params(
        [{"role": "user", "content": "Hi"}],
        0.7,
        4096,
        True,
    )
    assert "thinking" in params
    assert params["thinking"]["type"] == "enabled"
    assert params["thinking"]["budget_tokens"] == 2048


def test_build_params_no_system():
    """No system key when no system message provided."""
    pytest.importorskip("anthropic")

    from agents7.llm.anthropic import AnthropicProvider

    provider = AnthropicProvider("claude-test")
    params = provider._build_params(
        [{"role": "user", "content": "Hi"}],
        0.7,
        -1,
        False,
    )
    assert "system" not in params


# --- _parse_response ---


def test_parse_response_text_only():
    """Parses text-only response."""
    pytest.importorskip("anthropic")

    from agents7.llm.anthropic import AnthropicProvider

    resp = _mock_response(text="World")
    start = time.monotonic()
    result = AnthropicProvider._parse_response(
        resp,
        start,
    )
    assert result["response"] == "World"
    assert result["role"] == "assistant"
    assert result["thinking"] is None
    assert result["metrics"]["tokens"] == 5
    assert result["metrics"]["prompt_tokens"] == 10


def test_parse_response_with_thinking():
    """Parses response with thinking block."""
    pytest.importorskip("anthropic")

    from agents7.llm.anthropic import AnthropicProvider

    resp = _mock_response(
        text="Answer",
        thinking="Let me think...",
    )
    result = AnthropicProvider._parse_response(
        resp,
        time.monotonic(),
    )
    assert result["response"] == "Answer"
    assert result["thinking"] == "Let me think..."


# --- LLMEngine integration ---


def test_engine_creates_anthropic_provider():
    """LLMEngine('anthropic', ...) creates provider."""
    pytest.importorskip("anthropic")

    from agents7.llm import LLMEngine

    engine = LLMEngine(
        provider="anthropic",
        model_name="claude-sonnet-4-20250514",
    )
    assert engine.provider == "anthropic"
    assert engine.model_name == "claude-sonnet-4-20250514"
    assert engine.model_id == ("anthropic::claude-sonnet-4-20250514")


def test_engine_anthropic_with_fallback():
    """LLMEngine creates fallback provider."""
    pytest.importorskip("anthropic")

    from agents7.llm import LLMEngine

    engine = LLMEngine(
        provider="anthropic",
        model_name="claude-sonnet-4-20250514",
        fallback_model="claude-haiku-4-5-20251001",
    )
    assert engine._fallback_provider is not None
    assert engine._fallback_provider.model_name == "claude-haiku-4-5-20251001"


# --- Stream adapter ---


def test_adapt_stream_sync():
    """Stream adapter converts Anthropic events."""
    pytest.importorskip("anthropic")

    from agents7.llm.anthropic import _adapt_stream_sync

    events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=10),
            ),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(
                type="text_delta",
                text="Hello",
            ),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(
                type="text_delta",
                text=" world",
            ),
        ),
        SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(output_tokens=5),
        ),
    ]

    chunks = list(_adapt_stream_sync(iter(events)))
    assert len(chunks) == 3
    assert chunks[0].message.content == "Hello"
    assert chunks[1].message.content == " world"
    assert chunks[2].done is True
    assert chunks[2].eval_count == 5
    assert chunks[2].prompt_eval_count == 10
