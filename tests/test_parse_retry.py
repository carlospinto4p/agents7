"""Unit tests for max_parse_retries on Labeler."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel, create_model

from agents7.agents.labeler import Labeler
from agents7.agents.parsing import ResponseParser
from agents7.agents.retry import ParseRetryHandler
from agents7.exceptions import ParseError
from agents7.llm import LLMEngine

from helpers import DUMMY_METRICS, make_labeler


def _chat_response(value: str) -> dict:
    return {
        "role": "assistant",
        "response": json.dumps({"value": value}),
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }


def _bad_response() -> dict:
    return {
        "role": "assistant",
        "response": "not valid json at all",
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }


# --- No retries (default) ---


@patch.object(LLMEngine, "chat")
def test_no_retries_default(mock_chat):
    """With max_parse_retries=0, parse failure returns None."""
    mock_chat.return_value = _bad_response()
    agent = make_labeler()
    result = agent.act({"text": "hello"})
    assert result.model is None
    assert mock_chat.call_count == 1


# --- Retry succeeds on second attempt ---


@patch.object(LLMEngine, "chat")
def test_retry_succeeds_on_second_attempt(mock_chat):
    """Parse retry re-prompts and succeeds on second call."""
    mock_chat.side_effect = [
        _bad_response(),
        _chat_response("fixed"),
    ]
    agent = make_labeler(max_parse_retries=1)
    result = agent.act({"text": "hello"})
    assert result.model is not None
    assert result.model.value == "fixed"
    assert mock_chat.call_count == 2


# --- Retry exhausted ---


@patch.object(LLMEngine, "chat")
def test_retry_exhausted_returns_none(mock_chat):
    """All retries fail, model is None."""
    mock_chat.side_effect = [
        _bad_response(),
        _bad_response(),
        _bad_response(),
    ]
    agent = make_labeler(max_parse_retries=2)
    result = agent.act({"text": "hello"})
    assert result.model is None
    # 1 initial + 2 retries = 3 calls
    assert mock_chat.call_count == 3


# --- Retry message content ---


@patch.object(LLMEngine, "chat")
def test_retry_appends_error_message(mock_chat):
    """Retry call includes the parse error in messages."""
    mock_chat.side_effect = [
        _bad_response(),
        _chat_response("ok"),
    ]
    agent = make_labeler(max_parse_retries=1)
    agent.act({"text": "hello"})

    # Second call should have 3 messages (system, user, retry)
    retry_call_messages = mock_chat.call_args_list[1][1]["messages"]
    assert len(retry_call_messages) == 3
    retry_msg = retry_call_messages[2]
    assert retry_msg["role"] == "user"
    assert "could not be parsed" in retry_msg["content"]
    assert "valid JSON" in retry_msg["content"]


# --- Multiple retries accumulate messages ---


@patch.object(LLMEngine, "chat")
def test_multiple_retries_accumulate_messages(mock_chat):
    """Each retry appends another error message."""
    mock_chat.side_effect = [
        _bad_response(),
        _bad_response(),
        _chat_response("finally"),
    ]
    agent = make_labeler(max_parse_retries=2)
    result = agent.act({"text": "hello"})
    assert result.model is not None
    assert result.model.value == "finally"
    # Third call should have 4 messages
    retry_messages = mock_chat.call_args_list[2][1]["messages"]
    assert len(retry_messages) == 4


# --- Retries are paid for, so they are reported ---


@patch.object(LLMEngine, "chat")
def test_parse_retries_are_added_to_the_response_metrics(mock_chat):
    """A parse retry is a whole extra LLM call, with its own bill.

    Unlike a transport retry, which raises before the provider
    reports anything, this one succeeds — it just produces text
    that will not parse. Its metrics were discarded, so an agent
    that needed two retries reported one call's tokens and cost
    while having paid for three.
    """
    mock_chat.side_effect = [
        _bad_response(),
        _bad_response(),
        _chat_response("finally"),
    ]
    agent = make_labeler(max_parse_retries=2)

    result = agent.act({"text": "hello"})

    assert mock_chat.call_count == 3
    assert result.metrics.tokens == 3 * DUMMY_METRICS["tokens"]
    assert result.metrics.prompt_tokens == 3 * DUMMY_METRICS["prompt_tokens"]
    assert result.metrics.total_duration == pytest.approx(
        3 * DUMMY_METRICS["total_duration"]
    )


@patch.object(LLMEngine, "chat")
def test_a_call_that_parsed_reports_only_itself(mock_chat):
    """No retry, nothing to add — the common case is unchanged."""
    mock_chat.return_value = _chat_response("fine")
    agent = make_labeler(max_parse_retries=2)

    result = agent.act({"text": "hello"})

    assert result.metrics.tokens == DUMMY_METRICS["tokens"]


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_async_parse_retries_are_added_too(mock_chat):
    """The async path bills the same as the sync one."""
    mock_chat.side_effect = [
        _bad_response(),
        _chat_response("fixed"),
    ]
    agent = make_labeler(max_parse_retries=1)

    result = await agent.act_async({"text": "hello"})

    assert result.metrics.tokens == 2 * DUMMY_METRICS["tokens"]


# --- Async retry ---


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_async_retry_succeeds(mock_chat):
    """Async parse retry works correctly."""
    mock_chat.side_effect = [
        _bad_response(),
        _chat_response("async_fixed"),
    ]
    agent = make_labeler(max_parse_retries=1)
    result = await agent.act_async({"text": "hello"})
    assert result.model is not None
    assert result.model.value == "async_fixed"
    assert mock_chat.call_count == 2


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_async_retry_exhausted(mock_chat):
    """Async retries exhausted returns None model."""
    mock_chat.side_effect = [
        _bad_response(),
        _bad_response(),
        _bad_response(),
    ]
    agent = make_labeler(max_parse_retries=2)
    result = await agent.act_async({"text": "hello"})
    assert result.model is None
    assert mock_chat.call_count == 3


# --- YAML loading ---


def test_yaml_max_parse_retries(tmp_path):
    """max_parse_retries is loaded from YAML config."""
    yaml_content = (
        "name: RetryAgent\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Test.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
        "max_parse_retries: 3\n"
    )
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content)
    agent = Labeler.from_yaml(str(yaml_file))
    assert agent.max_parse_retries == 3


def test_yaml_no_max_parse_retries(tmp_path):
    """Without max_parse_retries in YAML, defaults to 0."""
    yaml_content = (
        "name: NoRetry\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Test.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
    )
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content)
    agent = Labeler.from_yaml(str(yaml_file))
    assert agent.max_parse_retries == 0


# --- Static method ---


def test_parse_retry_message_content():
    """ParseRetryHandler._retry_message builds correct retry message."""

    error = ParseError("bad json")
    msg = ParseRetryHandler._retry_message(error)
    assert msg["role"] == "user"
    assert "bad json" in msg["content"]
    assert "valid JSON" in msg["content"]


# --- Direct ParseRetryHandler unit tests ---


class _Out(BaseModel):
    value: str


_OutList = create_model("RetryTestOutList", items=(list[_Out], ...))


def _make_handler(max_retries: int = 1, chat_return: dict | None = None):
    engine = Mock()
    if chat_return is not None:
        engine.chat.return_value = chat_return
    parser = ResponseParser(
        output_model=_Out,
        output_model_list=_OutList,
        name="test",
    )
    handler = ParseRetryHandler(
        engine=engine,
        parser=parser,
        name="test",
        max_retries=max_retries,
        output_schema=_Out.model_json_schema(),
    )
    return handler, engine


def test_handler_zero_retries_skips_engine():
    """max_retries=0 returns None immediately without calling engine."""
    handler, engine = _make_handler(max_retries=0)
    result = handler.retry_sync(
        error=ParseError("bad"), messages=[], chat_kwargs={}
    )
    assert result is None
    engine.chat.assert_not_called()


def test_handler_succeeds_on_retry():
    """retry_sync returns parsed model when engine returns valid JSON."""
    handler, engine = _make_handler(
        max_retries=1,
        chat_return={
            "role": "assistant",
            "response": json.dumps({"value": "ok"}),
        },
    )
    result = handler.retry_sync(
        error=ParseError("bad"), messages=[], chat_kwargs={}
    )
    assert result is not None
    assert result.value == "ok"


def test_handler_all_retries_fail_returns_none():
    """retry_sync returns None when all retry attempts fail."""
    handler, engine = _make_handler(max_retries=2)
    engine.chat.return_value = {"role": "assistant", "response": "not json"}
    result = handler.retry_sync(
        error=ParseError("first"), messages=[], chat_kwargs={}
    )
    assert result is None
    assert engine.chat.call_count == 2


def test_handler_chat_uses_temperature_zero():
    """retry_sync always calls engine with temperature=0."""
    handler, engine = _make_handler(
        max_retries=1,
        chat_return={
            "role": "assistant",
            "response": json.dumps({"value": "ok"}),
        },
    )
    handler.retry_sync(error=ParseError("err"), messages=[], chat_kwargs={})
    assert engine.chat.call_args.kwargs["temperature"] == 0


@pytest.mark.asyncio
async def test_handler_async_zero_retries_skips_engine():
    """async: max_retries=0 returns None without calling engine."""
    handler, engine = _make_handler(max_retries=0)
    engine.chat_async = AsyncMock()
    result = await handler.retry_async(
        error=ParseError("bad"), messages=[], chat_kwargs={}
    )
    assert result is None
    engine.chat_async.assert_not_called()


@pytest.mark.asyncio
async def test_handler_async_succeeds_on_retry():
    """async: retry_async returns parsed model on success."""
    handler, engine = _make_handler(max_retries=1)
    engine.chat_async = AsyncMock(
        return_value={
            "role": "assistant",
            "response": json.dumps({"value": "async_ok"}),
        }
    )
    result = await handler.retry_async(
        error=ParseError("bad"), messages=[], chat_kwargs={}
    )
    assert result is not None
    assert result.value == "async_ok"
