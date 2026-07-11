"""Unit tests for the OpenAI-compatible LLM provider."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


# --- Import guard ---


def test_import_error_without_openai():
    """Raises ImportError when openai package is missing."""
    with patch.dict("sys.modules", {"openai": None}):
        # Force reimport
        import importlib
        import agents7.llm.openai_compat as mod

        importlib.reload(mod)
        with pytest.raises(ImportError, match="openai"):
            mod._require_openai()


# --- Helpers ---


def _make_usage(prompt_tokens=10, completion_tokens=20):
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _make_choice(content="hello", role="assistant", reasoning=None):
    msg = SimpleNamespace(
        content=content,
        role=role,
    )
    if reasoning is not None:
        msg.reasoning_content = reasoning
    return SimpleNamespace(
        message=msg,
        finish_reason="stop",
    )


def _make_response(content="hello", reasoning=None):
    return SimpleNamespace(
        choices=[_make_choice(content, reasoning=reasoning)],
        usage=_make_usage(),
    )


def _make_delta(content="hi", role="assistant", reasoning=None):
    delta = SimpleNamespace(content=content, role=role)
    if reasoning is not None:
        delta.reasoning_content = reasoning
    return delta


def _make_stream_chunk(content="hi", role=None, usage=None):
    delta = _make_delta(content=content, role=role)
    chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=delta)],
    )
    if usage is not None:
        chunk.usage = usage
    else:
        chunk.usage = None
    return chunk


# --- Provider construction ---


@patch("agents7.llm.openai_compat._openai")
def test_provider_creation(mock_openai):
    """Provider initializes with correct attributes."""
    mock_openai.OpenAI = MagicMock()
    mock_openai.AsyncOpenAI = MagicMock()
    from agents7.llm.openai_compat import OpenAICompatProvider

    p = OpenAICompatProvider("gpt-4o", base_url="http://x")
    assert p.model_name == "gpt-4o"
    assert p.provider == "openai_compat"
    assert p._base_url == "http://x"


# --- Env-var configuration (OPENAI_COMPAT_*) ---


@patch("agents7.llm.openai_compat._openai")
def test_provider_reads_env_vars(mock_openai, monkeypatch):
    """Omitted args fall back to OPENAI_COMPAT_* env vars."""
    mock_openai.OpenAI = MagicMock()
    mock_openai.AsyncOpenAI = MagicMock()
    monkeypatch.setenv("OPENAI_COMPAT_MODEL_NAME", "deepseek-chat")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "sk-env")
    from agents7.llm.openai_compat import OpenAICompatProvider

    p = OpenAICompatProvider()
    assert p.model_name == "deepseek-chat"
    assert p._base_url == "https://api.deepseek.com"
    assert p._api_key == "sk-env"
    mock_openai.OpenAI.assert_called_once_with(
        base_url="https://api.deepseek.com",
        api_key="sk-env",
    )


@patch("agents7.llm.openai_compat._openai")
def test_provider_explicit_args_override_env(mock_openai, monkeypatch):
    """Explicit args take precedence over the env vars."""
    mock_openai.OpenAI = MagicMock()
    mock_openai.AsyncOpenAI = MagicMock()
    monkeypatch.setenv("OPENAI_COMPAT_MODEL_NAME", "deepseek-chat")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://api.deepseek.com")
    from agents7.llm.openai_compat import OpenAICompatProvider

    p = OpenAICompatProvider("gpt-4o", base_url="http://x", api_key="sk-x")
    assert p.model_name == "gpt-4o"
    assert p._base_url == "http://x"
    assert p._api_key == "sk-x"


@patch("agents7.llm.openai_compat._openai")
def test_provider_no_model_raises(mock_openai, monkeypatch):
    """No model (arg or env) raises a clear ValueError."""
    mock_openai.OpenAI = MagicMock()
    mock_openai.AsyncOpenAI = MagicMock()
    monkeypatch.delenv("OPENAI_COMPAT_MODEL_NAME", raising=False)
    from agents7.llm.openai_compat import OpenAICompatProvider

    with pytest.raises(ValueError, match="OPENAI_COMPAT_MODEL_NAME"):
        OpenAICompatProvider()


# --- _parse_response ---


def test_parse_response_basic():
    """Parses a standard completion response."""
    from agents7.llm.openai_compat import OpenAICompatProvider

    resp = _make_response(content="  result  ")
    parsed = OpenAICompatProvider._parse_response(resp)
    assert parsed["response"] == "result"
    assert parsed["role"] == "assistant"
    assert parsed["thinking"] is None
    assert parsed["metrics"]["tokens"] == 20
    assert parsed["metrics"]["prompt_tokens"] == 10


def test_parse_response_with_reasoning():
    """Parses reasoning_content into thinking field."""
    from agents7.llm.openai_compat import OpenAICompatProvider

    resp = _make_response(content="ans", reasoning="  thought  ")
    parsed = OpenAICompatProvider._parse_response(resp)
    assert parsed["thinking"] == "thought"


# --- _build_params ---


def test_build_params_basic():
    """Builds minimal params without optional fields."""
    from agents7.llm.openai_compat import OpenAICompatProvider

    params = OpenAICompatProvider._build_params(
        "gpt-4o",
        [{"role": "user", "content": "hi"}],
        0.5,
        -1,
        None,
        None,
    )
    assert params["model"] == "gpt-4o"
    assert params["temperature"] == 0.5
    assert "max_tokens" not in params
    assert "seed" not in params
    assert "response_format" not in params


def test_build_params_json_schema():
    """JSON schema output_format builds json_schema response."""
    from agents7.llm.openai_compat import OpenAICompatProvider

    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    params = OpenAICompatProvider._build_params(
        "m",
        [],
        0,
        100,
        schema,
        42,
    )
    assert params["max_tokens"] == 100
    assert params["seed"] == 42
    rf = params["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == schema


def test_build_params_json_string():
    """String 'json' output_format uses json_object type."""
    from agents7.llm.openai_compat import OpenAICompatProvider

    params = OpenAICompatProvider._build_params(
        "m",
        [],
        0,
        -1,
        "json",
        None,
    )
    assert params["response_format"]["type"] == "json_object"


def test_build_params_stream():
    """Stream mode includes stream_options."""
    from agents7.llm.openai_compat import OpenAICompatProvider

    params = OpenAICompatProvider._build_params(
        "m",
        [],
        0,
        -1,
        None,
        None,
        stream=True,
    )
    assert params["stream"] is True
    assert params["stream_options"] == {"include_usage": True}


# --- Chat sync ---


@patch("agents7.llm.openai_compat._openai")
def test_chat_sync(mock_openai):
    """chat() calls client and returns parsed dict."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_response("ok")
    mock_openai.OpenAI.return_value = mock_client
    mock_openai.AsyncOpenAI.return_value = MagicMock()

    from agents7.llm.openai_compat import OpenAICompatProvider

    p = OpenAICompatProvider("m")
    result = p.chat(
        [{"role": "user", "content": "test"}],
        temperature=0,
    )
    assert result["response"] == "ok"
    mock_client.chat.completions.create.assert_called_once()


# --- Chat async ---


@patch("agents7.llm.openai_compat._openai")
@pytest.mark.asyncio
async def test_chat_async(mock_openai):
    """chat_async() calls async client and returns parsed dict."""
    mock_async_client = MagicMock()
    mock_async_client.chat.completions.create = AsyncMock(
        return_value=_make_response("async_ok")
    )
    mock_openai.OpenAI.return_value = MagicMock()
    mock_openai.AsyncOpenAI.return_value = mock_async_client

    from agents7.llm.openai_compat import OpenAICompatProvider

    p = OpenAICompatProvider("m")
    result = await p.chat_async(
        [{"role": "user", "content": "test"}],
        temperature=0,
    )
    assert result["response"] == "async_ok"


# --- Generate delegates to chat ---


@patch("agents7.llm.openai_compat._openai")
def test_generate_delegates_to_chat(mock_openai):
    """generate() wraps prompt in messages and removes role."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_response("gen")
    mock_openai.OpenAI.return_value = mock_client
    mock_openai.AsyncOpenAI.return_value = MagicMock()

    from agents7.llm.openai_compat import OpenAICompatProvider

    p = OpenAICompatProvider("m")
    result = p.generate("hello")
    assert result["response"] == "gen"
    assert "role" not in result


# --- Stream adapter ---


def test_oai_chunk_adapter():
    """_OAIChunk adapts OpenAI stream chunks correctly."""
    from agents7.llm.openai_compat import _OAIChunk

    chunk = _make_stream_chunk(content="hi", role="assistant")
    adapted = _OAIChunk(chunk)
    assert adapted.message.content == "hi"
    assert adapted.message.role == "assistant"
    assert adapted.done is False


def test_oai_chunk_adapter_with_usage():
    """_OAIChunk sets done=True when usage is present."""
    from agents7.llm.openai_compat import _OAIChunk

    chunk = _make_stream_chunk(content="", usage=_make_usage(5, 15))
    adapted = _OAIChunk(chunk)
    assert adapted.done is True
    assert adapted.eval_count == 15
    assert adapted.prompt_eval_count == 5


# --- LLMEngine integration ---


def test_engine_creates_openai_compat_provider():
    """LLMEngine with provider='openai_compat' instantiates
    OpenAICompatProvider."""
    with patch("agents7.llm.openai_compat.OpenAICompatProvider") as mock_cls:
        mock_cls.return_value = Mock()
        from agents7.llm.base import LLMEngine

        engine = LLMEngine(
            "openai_compat",
            "gpt-4o",
            base_url="http://localhost:8000/v1",
            api_key="sk-test",
        )
        mock_cls.assert_called_once_with(
            "gpt-4o",
            base_url="http://localhost:8000/v1",
            api_key="sk-test",
        )
        assert engine.provider == "openai_compat"


def test_engine_reads_model_from_env(monkeypatch):
    """LLMEngine resolves the model from OPENAI_COMPAT_MODEL_NAME
    when model_name is omitted for openai_compat."""
    monkeypatch.setenv("OPENAI_COMPAT_MODEL_NAME", "deepseek-chat")
    with patch("agents7.llm.openai_compat.OpenAICompatProvider") as mock_cls:
        mock_cls.return_value = Mock()
        from agents7.llm.base import LLMEngine

        engine = LLMEngine("openai_compat")
        assert engine.model_name == "deepseek-chat"
        assert engine.model_id == "openai_compat::deepseek-chat"
        mock_cls.assert_called_once_with(
            "deepseek-chat",
            base_url=None,
            api_key=None,
        )


def test_engine_missing_model_raises(monkeypatch):
    """LLMEngine with no model and no env var raises ValueError."""
    monkeypatch.delenv("OPENAI_COMPAT_MODEL_NAME", raising=False)
    from agents7.llm.base import LLMEngine

    with pytest.raises(ValueError, match="model_name"):
        LLMEngine("openai_compat")


def test_engine_openai_compat_with_fallback():
    """LLMEngine creates fallback provider for openai_compat."""
    with patch("agents7.llm.openai_compat.OpenAICompatProvider") as mock_cls:
        mock_cls.return_value = Mock()
        from agents7.llm.base import LLMEngine

        LLMEngine(
            "openai_compat",
            "gpt-4o",
            fallback_model="gpt-3.5-turbo",
            base_url="http://x",
        )
        assert mock_cls.call_count == 2
        assert mock_cls.call_args_list[1][0][0] == "gpt-3.5-turbo"


# --- Error handling ---


@patch("agents7.llm.openai_compat._openai")
def test_chat_sync_propagates_api_error(mock_openai):
    """chat() propagates exceptions from the API client."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("api down")
    mock_openai.OpenAI.return_value = mock_client
    mock_openai.AsyncOpenAI.return_value = MagicMock()

    from agents7.llm.openai_compat import OpenAICompatProvider

    p = OpenAICompatProvider("m")
    with pytest.raises(RuntimeError, match="api down"):
        p.chat([{"role": "user", "content": "hi"}])


@patch("agents7.llm.openai_compat._openai")
@pytest.mark.asyncio
async def test_chat_async_propagates_api_error(mock_openai):
    """chat_async() propagates exceptions from the async client."""
    mock_async_client = MagicMock()
    mock_async_client.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("async api down")
    )
    mock_openai.OpenAI.return_value = MagicMock()
    mock_openai.AsyncOpenAI.return_value = mock_async_client

    from agents7.llm.openai_compat import OpenAICompatProvider

    p = OpenAICompatProvider("m")
    with pytest.raises(RuntimeError, match="async api down"):
        await p.chat_async([{"role": "user", "content": "hi"}])


# --- Stream adapter edge cases ---


def test_oai_chunk_empty_choices():
    """_OAIChunk with empty choices list yields empty content."""
    from agents7.llm.openai_compat import _OAIChunk

    chunk = SimpleNamespace(choices=[], usage=None)
    adapted = _OAIChunk(chunk)
    assert adapted.message.content == ""
    assert adapted.message.role is None
    assert adapted.done is False


def test_oai_chunk_delta_with_reasoning():
    """_OAIChunk adapts reasoning_content from the delta."""
    from agents7.llm.openai_compat import _OAIChunk

    chunk = _make_stream_chunk(content="ans", role="assistant")
    chunk.choices[0].delta.reasoning_content = "  thought  "
    adapted = _OAIChunk(chunk)
    assert adapted.message.thinking == "  thought  "


def test_adapt_stream_sync_filters_empty_chunks():
    """_adapt_stream_sync skips chunks with no choices and no usage."""
    from agents7.llm.openai_compat import _adapt_stream_sync

    empty = SimpleNamespace(choices=[], usage=None)
    content_chunk = _make_stream_chunk(content="hello")
    results = list(_adapt_stream_sync([empty, content_chunk]))
    assert len(results) == 1
    assert results[0].message.content == "hello"


def test_adapt_stream_sync_yields_usage_chunk():
    """_adapt_stream_sync yields a no-choices chunk when it carries usage."""
    from agents7.llm.openai_compat import _adapt_stream_sync

    usage_chunk = SimpleNamespace(choices=[], usage=_make_usage(5, 10))
    results = list(_adapt_stream_sync([usage_chunk]))
    assert len(results) == 1
    assert results[0].done is True
    assert results[0].eval_count == 10


# --- Streaming integration ---


@patch("agents7.llm.openai_compat._openai")
def test_chat_stream_returns_stream_result(mock_openai):
    """chat_stream() returns a StreamResult that yields text chunks."""
    from agents7.llm.streaming import StreamResult

    chunks = [
        _make_stream_chunk(content="hello"),
        _make_stream_chunk(content="", usage=_make_usage(5, 10)),
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter(chunks)
    mock_openai.OpenAI.return_value = mock_client
    mock_openai.AsyncOpenAI.return_value = MagicMock()

    from agents7.llm.openai_compat import OpenAICompatProvider

    p = OpenAICompatProvider("m")
    result = p.chat_stream([{"role": "user", "content": "hi"}])
    assert isinstance(result, StreamResult)
    assert "".join(result) == "hello"
    assert result.metrics["tokens"] == 10


@patch("agents7.llm.openai_compat._openai")
@pytest.mark.asyncio
async def test_chat_stream_async_returns_async_stream_result(mock_openai):
    """chat_stream_async() returns an AsyncStreamResult that yields chunks."""
    from agents7.llm.streaming import AsyncStreamResult

    async def fake_raw_stream():
        yield _make_stream_chunk(content="world")
        yield _make_stream_chunk(content="", usage=_make_usage(3, 7))

    mock_async_client = MagicMock()
    mock_async_client.chat.completions.create = AsyncMock(
        return_value=fake_raw_stream()
    )
    mock_openai.OpenAI.return_value = MagicMock()
    mock_openai.AsyncOpenAI.return_value = mock_async_client

    from agents7.llm.openai_compat import OpenAICompatProvider

    p = OpenAICompatProvider("m")
    result = await p.chat_stream_async([{"role": "user", "content": "hi"}])
    assert isinstance(result, AsyncStreamResult)
    collected = []
    async for text in result:
        collected.append(text)
    assert "".join(collected) == "world"
    assert result.metrics["tokens"] == 7


# --- Tool calling ---


def _tool_call(name, arguments, id_):
    return SimpleNamespace(
        id=id_,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _bare_provider():
    """An OpenAICompatProvider without a real client (tool helpers
    never touch the client)."""
    with patch("agents7.llm.openai_compat._openai") as mock_openai:
        mock_openai.OpenAI.return_value = MagicMock()
        mock_openai.AsyncOpenAI.return_value = MagicMock()
        from agents7.llm.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider("m")


def test_openai_supports_tools():
    """OpenAI-compat provider advertises tool support."""
    assert _bare_provider().supports_tools() is True


def test_build_params_passes_tools_through():
    """Tools are forwarded unchanged (registry == OpenAI format)."""
    from agents7.llm.openai_compat import OpenAICompatProvider

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    params = OpenAICompatProvider._build_params(
        "m", [], 0, -1, None, None, tools=tools
    )
    assert params["tools"] == tools


def test_build_params_no_tools_key_when_absent():
    """No tools key when none provided."""
    from agents7.llm.openai_compat import OpenAICompatProvider

    params = OpenAICompatProvider._build_params("m", [], 0, -1, None, None)
    assert "tools" not in params


def test_parse_tool_calls_decodes_json_arguments():
    """function.arguments JSON string decodes to a dict."""
    from agents7.llm.openai_compat import OpenAICompatProvider

    msg = SimpleNamespace(
        content=None,
        role="assistant",
        tool_calls=[_tool_call("get_weather", '{"city": "Paris"}', "call_1")],
    )
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg)], usage=_make_usage()
    )
    parsed = OpenAICompatProvider._parse_response(resp)
    assert parsed["tool_calls"] == [
        {
            "name": "get_weather",
            "arguments": {"city": "Paris"},
            "id": "call_1",
        }
    ]
    assert parsed["response"] == ""  # null content is null-safe


def test_parse_tool_calls_malformed_arguments_empty_dict():
    """Malformed argument strings degrade to an empty dict."""
    from agents7.llm.openai_compat import OpenAICompatProvider

    msg = SimpleNamespace(
        tool_calls=[_tool_call("ping", "not json", "call_x")],
    )
    calls = OpenAICompatProvider._parse_tool_calls(msg)
    assert calls == [{"name": "ping", "arguments": {}, "id": "call_x"}]


def test_parse_response_no_tools_empty_list():
    """A text-only response yields an empty tool_calls list."""
    from agents7.llm.openai_compat import OpenAICompatProvider

    parsed = OpenAICompatProvider._parse_response(_make_response("hi"))
    assert parsed["tool_calls"] == []


def test_build_tool_turn_messages_shape():
    """Single call: assistant tool_calls + tool message by id."""
    provider = _bare_provider()
    calls = [{"name": "add", "arguments": {"a": 1, "b": 2}, "id": "call_x"}]
    msgs = provider.build_tool_turn_messages("thinking...", calls, ["3"])

    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"] == "thinking..."
    tc = msgs[0]["tool_calls"][0]
    assert tc == {
        "id": "call_x",
        "type": "function",
        "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
    }
    assert msgs[1] == {
        "role": "tool",
        "tool_call_id": "call_x",
        "content": "3",
    }


def test_build_tool_turn_messages_content_none_when_empty():
    """Empty assistant text becomes None, not an empty string."""
    provider = _bare_provider()
    calls = [{"name": "add", "arguments": {"a": 1}, "id": "call_x"}]
    msgs = provider.build_tool_turn_messages("", calls, ["1"])
    assert msgs[0]["content"] is None


def test_build_tool_turn_messages_multi_call_pairing():
    """zip pairing is correct by index for multiple tool calls."""
    provider = _bare_provider()
    calls = [
        {"name": "add", "arguments": {"a": 1}, "id": "call_a"},
        {"name": "mul", "arguments": {"a": 2}, "id": "call_b"},
    ]
    msgs = provider.build_tool_turn_messages("", calls, ["1", "4"])

    assert len(msgs[0]["tool_calls"]) == 2
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "call_a"
    assert tool_msgs[0]["content"] == "1"
    assert tool_msgs[1]["tool_call_id"] == "call_b"
    assert tool_msgs[1]["content"] == "4"


@patch("agents7.llm.openai_compat._openai")
def test_chat_forwards_tools_to_client(mock_openai):
    """chat() passes the tools schema into the API call."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_response("ok")
    mock_openai.OpenAI.return_value = mock_client
    mock_openai.AsyncOpenAI.return_value = MagicMock()

    from agents7.llm.openai_compat import OpenAICompatProvider

    tools = [{"type": "function", "function": {"name": "f"}}]
    p = OpenAICompatProvider("m")
    p.chat([{"role": "user", "content": "hi"}], temperature=0, tools=tools)
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["tools"] == tools
