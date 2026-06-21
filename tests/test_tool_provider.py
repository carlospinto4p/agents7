"""Unit tests for provider-level tool support (Ollama)."""

from types import SimpleNamespace

from agents7.llm.ollama import (
    OllamaProvider,
    parse_ollama_chat_response,
)


def _msg(content="", tool_calls=None, thinking=None, role="assistant"):
    return SimpleNamespace(
        content=content,
        role=role,
        thinking=thinking,
        tool_calls=tool_calls,
    )


def _chat_response(message):
    return SimpleNamespace(
        message=message,
        eval_count=1,
        prompt_eval_count=2,
        total_duration=3_000_000_000,
    )


def test_parse_no_tool_calls_defaults_empty():
    parsed = parse_ollama_chat_response(_chat_response(_msg("hi")))
    assert parsed["tool_calls"] == []
    assert parsed["response"] == "hi"


def test_parse_extracts_tool_calls():
    tc = SimpleNamespace(
        function=SimpleNamespace(
            name="get_weather", arguments={"city": "Paris"}
        )
    )
    parsed = parse_ollama_chat_response(
        _chat_response(_msg(content=None, tool_calls=[tc]))
    )
    assert parsed["tool_calls"] == [
        {"name": "get_weather", "arguments": {"city": "Paris"}}
    ]
    assert parsed["response"] == ""  # null content is null-safe


def test_ollama_supports_tools():
    assert OllamaProvider("qwen3").supports_tools() is True


def test_build_tool_turn_messages_shape():
    provider = OllamaProvider("qwen3")
    calls = [{"name": "add", "arguments": {"a": 1, "b": 2}}]
    msgs = provider.build_tool_turn_messages("thinking...", calls, ["3"])
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["tool_calls"][0]["function"]["name"] == "add"
    assert msgs[1]["role"] == "tool"
    assert msgs[1]["content"] == "3"
    assert msgs[1]["tool_name"] == "add"


def test_build_tool_turn_messages_multi_call_pairing():
    """zip pairing is correct by index for multiple tool calls."""
    provider = OllamaProvider("qwen3")
    calls = [
        {"name": "add", "arguments": {"a": 1, "b": 2}},
        {"name": "mul", "arguments": {"a": 2, "b": 3}},
    ]
    msgs = provider.build_tool_turn_messages("", calls, ["3", "6"])

    # First message is the assistant turn carrying both tool_calls.
    assert msgs[0]["role"] == "assistant"
    assert len(msgs[0]["tool_calls"]) == 2

    # Tool result messages follow in order: add→3, mul→6.
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["tool_name"] == "add"
    assert tool_msgs[0]["content"] == "3"
    assert tool_msgs[1]["tool_name"] == "mul"
    assert tool_msgs[1]["content"] == "6"
