"""Unit tests for max_parse_retries on Labeler."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from agents7.agents.labeler import Labeler
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
    """_parse_retry_message builds correct retry message."""
    from agents7.exceptions import ParseError

    error = ParseError("bad json")
    msg = Labeler._parse_retry_message(error)
    assert msg["role"] == "user"
    assert "bad json" in msg["content"]
    assert "valid JSON" in msg["content"]
