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
    """Raises ImportError when anthropic not installed.

    The module is reloaded to simulate the missing package, and
    reloaded again afterwards — without the restore it stays
    stubbed for the rest of the session, and every later test
    that builds a real provider fails with an import error it did
    nothing to cause.
    """
    from importlib import reload

    import agents7.llm.anthropic as mod

    try:
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic"):
                reload(mod)
                mod._anthropic = None
                mod._require_anthropic()
    finally:
        reload(mod)


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


# --- Tool calling ---
#
# These exercise the Anthropic-specific wire format without a
# real client: the tool methods never touch ``self._client``, so
# we bypass ``__init__`` with ``object.__new__`` and the tests run
# even when the optional ``anthropic`` package is absent.


def _bare_provider():
    """Build an AnthropicProvider without running ``__init__``."""
    from agents7.llm.anthropic import AnthropicProvider

    provider = object.__new__(AnthropicProvider)
    provider.provider = "anthropic"
    provider.model_name = "claude-test"
    provider.model_id = "anthropic::claude-test"
    return provider


def _tool_use_block(name, input_, id_):
    return SimpleNamespace(
        type="tool_use",
        name=name,
        input=input_,
        id=id_,
    )


def test_anthropic_supports_tools():
    """Anthropic provider advertises tool support."""
    assert _bare_provider().supports_tools() is True


def test_convert_tools_openai_to_anthropic():
    """OpenAI-style schema maps to Anthropic input_schema."""
    from agents7.llm.anthropic import AnthropicProvider

    neutral = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    converted = AnthropicProvider._convert_tools(neutral)
    assert converted == [
        {
            "name": "get_weather",
            "description": "Get weather.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]


def test_convert_tools_flat_entry_and_defaults():
    """A flat (unwrapped) entry and missing keys get defaults."""
    from agents7.llm.anthropic import AnthropicProvider

    converted = AnthropicProvider._convert_tools([{"name": "ping"}])
    assert converted[0]["name"] == "ping"
    assert converted[0]["description"] == ""
    assert converted[0]["input_schema"] == {
        "type": "object",
        "properties": {},
    }


def test_build_params_includes_converted_tools():
    """Tools passed to _build_params land as Anthropic schemas."""
    provider = _bare_provider()
    provider._default_max_tokens = 4096
    tools = [
        {
            "type": "function",
            "function": {"name": "add", "parameters": {"type": "object"}},
        }
    ]
    params = provider._build_params(
        [{"role": "user", "content": "Hi"}],
        0.7,
        -1,
        False,
        tools=tools,
    )
    assert params["tools"] == [
        {
            "name": "add",
            "description": "",
            "input_schema": {"type": "object"},
        }
    ]


def test_build_params_no_tools_key_when_absent():
    """No tools key when none provided."""
    provider = _bare_provider()
    provider._default_max_tokens = 4096
    params = provider._build_params(
        [{"role": "user", "content": "Hi"}],
        0.7,
        -1,
        False,
    )
    assert "tools" not in params


def test_parse_response_extracts_tool_use():
    """tool_use blocks become {name, arguments, id} dicts."""
    from agents7.llm.anthropic import AnthropicProvider

    resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Let me check."),
            _tool_use_block("get_weather", {"city": "Paris"}, "toolu_1"),
        ],
        role="assistant",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    result = AnthropicProvider._parse_response(resp, time.monotonic())
    assert result["response"] == "Let me check."
    assert result["tool_calls"] == [
        {
            "name": "get_weather",
            "arguments": {"city": "Paris"},
            "id": "toolu_1",
        }
    ]


def test_parse_response_no_tools_empty_list():
    """A text-only response yields an empty tool_calls list."""
    from agents7.llm.anthropic import AnthropicProvider

    result = AnthropicProvider._parse_response(
        _mock_response(text="Hi"),
        time.monotonic(),
    )
    assert result["tool_calls"] == []


def test_build_tool_turn_messages_shape():
    """Single call: assistant tool_use + user tool_result linked by id."""
    provider = _bare_provider()
    calls = [{"name": "add", "arguments": {"a": 1, "b": 2}, "id": "toolu_x"}]
    msgs = provider.build_tool_turn_messages("thinking...", calls, ["3"])

    assert len(msgs) == 2
    assistant, user = msgs
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {
        "type": "text",
        "text": "thinking...",
    }
    assert assistant["content"][1] == {
        "type": "tool_use",
        "id": "toolu_x",
        "name": "add",
        "input": {"a": 1, "b": 2},
    }
    assert user["role"] == "user"
    assert user["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_x",
            "content": "3",
        }
    ]


def test_build_tool_turn_messages_no_text_block_when_empty():
    """Empty assistant text omits the leading text block."""
    provider = _bare_provider()
    calls = [{"name": "add", "arguments": {"a": 1}, "id": "toolu_x"}]
    msgs = provider.build_tool_turn_messages("", calls, ["1"])
    assert all(b["type"] == "tool_use" for b in msgs[0]["content"])


def test_build_tool_turn_messages_multi_call_single_user():
    """Parallel calls share one user message; ids pair by index."""
    provider = _bare_provider()
    calls = [
        {"name": "add", "arguments": {"a": 1}, "id": "toolu_a"},
        {"name": "mul", "arguments": {"a": 2}, "id": "toolu_b"},
    ]
    msgs = provider.build_tool_turn_messages("", calls, ["1", "4"])

    # Exactly two messages — one assistant, one user (roles must
    # alternate; consecutive user messages would be rejected).
    assert len(msgs) == 2
    assert msgs[0]["role"] == "assistant"
    assert len(msgs[0]["content"]) == 2
    assert msgs[1]["role"] == "user"

    results = msgs[1]["content"]
    assert len(results) == 2
    assert results[0]["tool_use_id"] == "toolu_a"
    assert results[0]["content"] == "1"
    assert results[1]["tool_use_id"] == "toolu_b"
    assert results[1]["content"] == "4"


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
