"""Unit tests for structured call logging."""

import json
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents7.call_log import (
    CallRecord,
    JsonCallLogger,
    generate_trace_id,
)
from agents7.llm import LLMEngine

from helpers import DUMMY_METRICS, make_labeler


def _chat_response(value="ok"):
    return {
        "role": "assistant",
        "response": json.dumps({"value": value}),
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }


def _bad_response():
    return {
        "role": "assistant",
        "response": "not json",
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }


# --- CallRecord ---


def test_call_record_to_dict():
    """CallRecord.to_dict returns all fields."""
    record = CallRecord(
        trace_id="abc123",
        agent_name="test",
        agent_version=1,
        model_id="ollama::qwen3",
        success=True,
        tokens=10,
        prompt_tokens=5,
        total_duration=0.1,
    )
    d = record.to_dict()
    assert d["trace_id"] == "abc123"
    assert d["agent_name"] == "test"
    assert d["success"] is True
    assert d["error"] is None


def test_call_record_with_error():
    """CallRecord captures error messages."""
    record = CallRecord(
        trace_id="x",
        agent_name="a",
        agent_version=1,
        model_id="m",
        success=False,
        error="parse failed",
    )
    assert record.error == "parse failed"


# --- generate_trace_id ---


def test_trace_id_unique():
    """Each call returns a unique 12-char hex string."""
    ids = {generate_trace_id() for _ in range(100)}
    assert len(ids) == 100
    for tid in ids:
        assert len(tid) == 12


# --- JsonCallLogger ---


def test_json_logger_to_file(tmp_path):
    """JsonCallLogger writes JSON lines to a file."""
    path = tmp_path / "calls.jsonl"
    logger = JsonCallLogger(path=path)

    record = CallRecord(
        trace_id="abc",
        agent_name="test",
        agent_version=1,
        model_id="ollama::qwen3",
        success=True,
    )
    logger(record)
    logger(record)

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    obj = json.loads(lines[0])
    assert obj["agent_name"] == "test"
    assert obj["success"] is True


def test_json_logger_to_stream():
    """JsonCallLogger writes to a stream."""
    stream = StringIO()
    logger = JsonCallLogger(stream=stream)

    record = CallRecord(
        trace_id="abc",
        agent_name="test",
        agent_version=1,
        model_id="m",
        success=True,
    )
    logger(record)

    output = stream.getvalue().strip()
    obj = json.loads(output)
    assert obj["trace_id"] == "abc"


def test_json_logger_defaults_to_stdout(capsys):
    """JsonCallLogger defaults to stdout."""
    logger = JsonCallLogger()

    record = CallRecord(
        trace_id="abc",
        agent_name="test",
        agent_version=1,
        model_id="m",
        success=True,
    )
    logger(record)

    captured = capsys.readouterr()
    obj = json.loads(captured.out.strip())
    assert obj["agent_name"] == "test"


# --- Integration with Labeler ---


@patch.object(LLMEngine, "chat")
def test_act_emits_call_log(mock_chat):
    """act() calls the call_logger with a CallRecord."""
    mock_chat.return_value = _chat_response("ok")
    mock_logger = MagicMock()

    agent = make_labeler(call_logger=mock_logger)
    agent.act({"text": "hello"})

    mock_logger.assert_called_once()
    record = mock_logger.call_args[0][0]
    assert isinstance(record, CallRecord)
    assert record.agent_name == "test"
    assert record.success is True
    assert record.model_id == "ollama::qwen3"
    assert record.tokens == 10


@patch.object(LLMEngine, "chat")
def test_act_logs_failure(mock_chat):
    """act() logs success=False when parsing fails."""
    mock_chat.return_value = _bad_response()
    mock_logger = MagicMock()

    agent = make_labeler(call_logger=mock_logger)
    agent.act({"text": "hello"})

    record = mock_logger.call_args[0][0]
    assert record.success is False


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_act_async_emits_call_log(mock_chat):
    """act_async() calls the call_logger."""
    mock_chat.return_value = _chat_response("ok")
    mock_logger = MagicMock()

    agent = make_labeler(call_logger=mock_logger)
    await agent.act_async({"text": "hello"})

    mock_logger.assert_called_once()
    record = mock_logger.call_args[0][0]
    assert record.success is True


@patch.object(LLMEngine, "chat")
def test_no_logger_no_error(mock_chat):
    """No call_logger set does not raise."""
    mock_chat.return_value = _chat_response("ok")
    agent = make_labeler()
    result = agent.act({"text": "hello"})
    assert result.model is not None


@patch.object(LLMEngine, "chat")
def test_json_logger_file_integration(mock_chat, tmp_path):
    """Full integration: act() → JsonCallLogger → file."""
    mock_chat.side_effect = [
        _chat_response("ok"),
        _chat_response("ok"),
    ]
    path = tmp_path / "calls.jsonl"
    logger = JsonCallLogger(path=path)

    agent = make_labeler(call_logger=logger)
    agent.act({"text": "hello"})
    agent.act({"text": "world"})

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert obj["agent_name"] == "test"
        assert obj["success"] is True
        assert len(obj["trace_id"]) == 12
