"""Unit tests for structured call logging."""

import json
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents7.call_log import (
    CallLogSummary,
    CallRecord,
    CallStats,
    JsonCallLogger,
    generate_trace_id,
    summarize_call_log,
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


def test_json_logger_reuses_file_handle(tmp_path):
    """Repeated calls reuse one persistent append handle."""
    path = tmp_path / "calls.jsonl"
    logger = JsonCallLogger(path=path)

    record = CallRecord(
        trace_id="abc",
        agent_name="test",
        agent_version=1,
        model_id="m",
        success=True,
    )
    logger(record)
    fh_after_first = logger._fh
    logger(record)

    assert fh_after_first is logger._fh
    assert len(path.read_text().strip().split("\n")) == 2


def test_json_logger_lazy_file_creation(tmp_path):
    """No file is created until the first record is written."""
    path = tmp_path / "calls.jsonl"
    JsonCallLogger(path=path)
    assert not path.exists()


def test_json_logger_close_and_context_manager(tmp_path):
    """close() releases the handle; context manager closes it."""
    path = tmp_path / "calls.jsonl"
    record = CallRecord(
        trace_id="abc",
        agent_name="test",
        agent_version=1,
        model_id="m",
        success=True,
    )

    with JsonCallLogger(path=path) as logger:
        logger(record)
        assert logger._fh is not None
    assert logger._fh is None

    # Idempotent: closing again does not raise.
    logger.close()


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


# --- summarize_call_log ---


def _record(
    agent="a",
    model="m",
    success=True,
    tokens=10,
    prompt_tokens=5,
    duration=0.1,
    error=None,
):
    return CallRecord(
        trace_id="t",
        agent_name=agent,
        agent_version=1,
        model_id=model,
        success=success,
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        total_duration=duration,
        error=error,
    )


def test_summarize_empty_iterable():
    """An empty source yields zeroed stats."""
    summary = summarize_call_log([])
    assert isinstance(summary, CallLogSummary)
    assert summary.overall.calls == 0
    assert summary.overall.success_rate == 0.0
    assert summary.overall.avg_duration == 0.0
    assert summary.by_agent == {}
    assert summary.by_model == {}
    assert summary.errors == {}


def test_summarize_from_records():
    """Aggregates totals across CallRecord objects."""
    summary = summarize_call_log(
        [
            _record(tokens=10, prompt_tokens=5, duration=0.2),
            _record(tokens=20, prompt_tokens=7, duration=0.4),
        ]
    )
    o = summary.overall
    assert o.calls == 2
    assert o.successes == 2
    assert o.failures == 0
    assert o.tokens == 30
    assert o.prompt_tokens == 12
    assert abs(o.total_duration - 0.6) < 1e-9
    assert abs(o.avg_duration - 0.3) < 1e-9
    assert o.success_rate == 1.0


def test_summarize_from_dicts():
    """Accepts plain dicts (as parsed from JSONL)."""
    summary = summarize_call_log(
        [
            {
                "agent_name": "a",
                "model_id": "m",
                "success": True,
                "tokens": 3,
            },
            {"agent_name": "a", "model_id": "m", "success": False},
        ]
    )
    assert summary.overall.calls == 2
    assert summary.overall.successes == 1
    assert summary.overall.failures == 1
    assert summary.overall.tokens == 3


def test_summarize_from_json_strings():
    """Accepts raw JSON-line strings."""
    line = json.dumps(_record().to_dict())
    summary = summarize_call_log([line, line])
    assert summary.overall.calls == 2


def test_summarize_breakdowns():
    """Splits stats by agent and by model."""
    summary = summarize_call_log(
        [
            _record(agent="x", model="m1", success=True),
            _record(agent="x", model="m2", success=False),
            _record(agent="y", model="m1", success=True),
        ]
    )
    assert summary.by_agent["x"].calls == 2
    assert summary.by_agent["x"].successes == 1
    assert summary.by_agent["y"].calls == 1
    assert summary.by_model["m1"].calls == 2
    assert summary.by_model["m1"].success_rate == 1.0
    assert summary.by_model["m2"].calls == 1


def test_summarize_error_counts():
    """Counts error messages for failed calls only."""
    summary = summarize_call_log(
        [
            _record(success=False, error="parse failed"),
            _record(success=False, error="parse failed"),
            _record(success=False, error="timeout"),
            _record(success=True),
        ]
    )
    assert summary.errors == {"parse failed": 2, "timeout": 1}


def test_summarize_ignores_success_without_error():
    """Successful records never contribute to error counts."""
    summary = summarize_call_log([_record(success=True, error=None)])
    assert summary.errors == {}


def test_summarize_invalid_item_raises():
    """A non-record item raises TypeError."""
    with pytest.raises(TypeError):
        summarize_call_log([123])


def test_summarize_from_jsonl_file(tmp_path):
    """Reads and summarizes a JSONL file path, skipping blanks."""
    path = tmp_path / "calls.jsonl"
    with JsonCallLogger(path=path) as logger:
        logger(_record(agent="a", success=True))
        logger(_record(agent="b", success=False, error="boom"))
    # Inject a blank line to confirm it is skipped.
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n")

    summary = summarize_call_log(path)
    assert summary.overall.calls == 2
    assert summary.by_agent["a"].successes == 1
    assert summary.errors == {"boom": 1}


def test_call_stats_zero_division_safe():
    """Properties on empty stats do not divide by zero."""
    s = CallStats()
    assert s.success_rate == 0.0
    assert s.avg_duration == 0.0


def test_summary_format_renders_sections():
    """format() includes overall, breakdown, and error lines."""
    summary = summarize_call_log(
        [
            _record(agent="x", model="m1", success=True),
            _record(
                agent="x", model="m1", success=False, error="parse failed"
            ),
        ]
    )
    text = summary.format()
    assert "Call log summary: 2 calls" in text
    assert "By agent:" in text
    assert "By model:" in text
    assert "Errors:" in text
    assert "parse failed" in text


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
