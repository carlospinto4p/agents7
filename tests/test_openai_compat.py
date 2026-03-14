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
