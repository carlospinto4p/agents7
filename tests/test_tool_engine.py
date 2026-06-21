"""Unit tests for tool threading through LLMEngine."""

from unittest.mock import AsyncMock

from helpers import DUMMY_METRICS, make_engine
from agents7.llm.cache import _CacheKeyBuilder


def test_engine_forwards_tools_to_provider():
    engine = make_engine()
    engine.llm_provider.chat.return_value = {
        "role": "assistant",
        "response": "ok",
        "thinking": None,
        "tool_calls": [],
        "metrics": DUMMY_METRICS,
    }
    tools = [{"type": "function", "function": {"name": "x"}}]
    engine.chat([{"role": "user", "content": "hi"}], tools=tools)
    _, kwargs = engine.llm_provider.chat.call_args
    assert kwargs["tools"] == tools


def test_engine_supports_tools_delegates():
    engine = make_engine()
    engine.llm_provider.supports_tools.return_value = True
    assert engine.supports_tools() is True


def test_make_key_stable_without_tools():
    # Adding the tools param must not change keys when tools is None.
    a = _CacheKeyBuilder.make_key("m", "chat", messages=[], temperature=0.0)
    b = _CacheKeyBuilder.make_key(
        "m", "chat", messages=[], temperature=0.0, tools=None
    )
    assert a == b


def test_make_key_changes_with_tools():
    a = _CacheKeyBuilder.make_key("m", "chat", messages=[], temperature=0.0)
    c = _CacheKeyBuilder.make_key(
        "m", "chat", messages=[], temperature=0.0, tools=[{"x": 1}]
    )
    assert a != c


def test_engine_build_tool_turn_messages_delegates():
    engine = make_engine()
    engine.llm_provider.build_tool_turn_messages.return_value = [
        {"role": "tool"}
    ]
    out = engine.build_tool_turn_messages(
        "text", [{"name": "f", "arguments": {}}], ["r"]
    )
    assert out == [{"role": "tool"}]
    engine.llm_provider.build_tool_turn_messages.assert_called_once_with(
        "text", [{"name": "f", "arguments": {}}], ["r"]
    )


def test_engine_supports_tools_false_delegates():
    engine = make_engine()
    engine.llm_provider.supports_tools.return_value = False
    assert engine.supports_tools() is False


async def test_engine_forwards_tools_to_provider_async():
    engine = make_engine()
    engine.llm_provider.chat_async = AsyncMock(
        return_value={
            "role": "assistant",
            "response": "ok",
            "thinking": None,
            "tool_calls": [],
            "metrics": DUMMY_METRICS,
        }
    )
    tools = [{"type": "function", "function": {"name": "x"}}]
    await engine.chat_async([{"role": "user", "content": "hi"}], tools=tools)
    _, kwargs = engine.llm_provider.chat_async.call_args
    assert kwargs["tools"] == tools
