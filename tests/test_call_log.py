"""Unit tests for structured call logging."""

import json
import sqlite3
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents7.call_log import (
    CallLogSummary,
    CallRecord,
    CallStats,
    JsonCallLogger,
    SqliteCallLogger,
    generate_trace_id,
    read_call_log,
    summarize_call_log,
)
from agents7.llm import LLMEngine
from agents7.llm.streaming import AsyncStreamResult, StreamResult

from helpers import DUMMY_METRICS, async_iter, chat_chunk, make_labeler


def _chat_response(value="ok"):
    return {
        "role": "assistant",
        "response": json.dumps({"value": value}),
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }


def _batch_chat_response(values):
    """A chunk response wrapping one item per value."""
    return {
        "role": "assistant",
        "response": json.dumps({"items": [{"value": v} for v in values]}),
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


# --- Failed chat calls ---
#
# A chat call that raises used to leave nothing in the call log, so a
# run could lose conversations to `Connection error.` while its log
# showed no failed call.


def _failed(mock_logger):
    """The records marked as failures, in order."""
    return [
        call[0][0]
        for call in mock_logger.call_args_list
        if call[0][0].success is False
    ]


@patch.object(LLMEngine, "chat")
def test_a_chat_call_that_raises_is_logged(mock_chat):
    """The failure is recorded, and still reaches the caller."""
    from agents7.exceptions import ProviderError

    from helpers import make_chatter

    mock_chat.side_effect = ProviderError("Connection error.")
    mock_logger = MagicMock()
    agent = make_chatter(call_logger=mock_logger)

    with pytest.raises(ProviderError):
        agent.chat("Hola")

    (record,) = _failed(mock_logger)
    assert record.agent_name == "test-chatter"
    assert "Connection error." in record.error
    assert "Hola" in json.loads(record.prompt)[-1]["content"]
    assert record.tokens == 0
    assert record.cost_cents is None
    assert record.total_duration >= 0


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_an_async_chat_call_that_raises_is_logged(mock_chat):
    from agents7.exceptions import ProviderError

    from helpers import make_chatter

    mock_chat.side_effect = ProviderError("Connection error.")
    mock_logger = MagicMock()
    agent = make_chatter(call_logger=mock_logger)

    with pytest.raises(ProviderError):
        await agent.chat_async("Hola")

    (record,) = _failed(mock_logger)
    assert "Connection error." in record.error


@patch.object(LLMEngine, "chat")
def test_a_failed_record_carries_the_retries_the_error_carries(mock_chat):
    """What the retry loop counted survives onto the record."""
    from agents7.exceptions import ProviderError

    from helpers import make_chatter

    error = ProviderError("All 2 retries exhausted.")
    error.retries = 3
    error.retry_duration = 3.5
    mock_chat.side_effect = error
    mock_logger = MagicMock()
    agent = make_chatter(call_logger=mock_logger)

    with pytest.raises(ProviderError):
        agent.chat("Hola")

    (record,) = _failed(mock_logger)
    assert record.retries == 3
    assert record.retry_duration == 3.5


@patch.object(LLMEngine, "chat")
def test_a_guarded_reply_that_raises_is_logged(mock_chat):
    from agents7.exceptions import ProviderError

    from helpers import make_guarded_chatter

    mock_chat.side_effect = ProviderError("Connection error.")
    mock_logger = MagicMock()
    agent = make_guarded_chatter(call_logger=mock_logger)

    with pytest.raises(ProviderError):
        agent.chat("Hola")

    (record,) = _failed(mock_logger)
    assert record.agent_name == "test-chatter"


@patch("agents7.agents.context_guard.ContextGuard.act")
@patch.object(LLMEngine, "chat")
def test_a_guard_check_that_raises_is_logged_as_the_guard(
    mock_chat, mock_guard
):
    """The reply succeeded; the guard's own call is what failed."""
    from agents7.exceptions import ProviderError

    from helpers import chat_response_dict, make_guarded_chatter

    mock_chat.return_value = chat_response_dict("¡Buenas!")
    mock_guard.side_effect = ProviderError("Connection error.")
    mock_logger = MagicMock()
    agent = make_guarded_chatter(call_logger=mock_logger)

    with pytest.raises(ProviderError):
        agent.chat("Hola")

    (record,) = _failed(mock_logger)
    assert record.agent_name == "test-guard"
    assert "¡Buenas!" in record.prompt


@patch.object(LLMEngine, "chat_stream")
def test_a_stream_that_fails_to_open_is_logged(mock_stream):
    from agents7.exceptions import ProviderError

    from helpers import make_chatter

    mock_stream.side_effect = ProviderError("Connection error.")
    mock_logger = MagicMock()
    agent = make_chatter(call_logger=mock_logger)

    with pytest.raises(ProviderError):
        agent.chat_stream("Hola")

    (record,) = _failed(mock_logger)
    assert "Connection error." in record.error


@patch.object(LLMEngine, "chat_stream")
def test_a_stream_that_fails_part_way_is_logged(mock_stream):
    from agents7.exceptions import ProviderError

    from helpers import make_chatter

    def chunks():
        yield chat_chunk(content="Hol")
        raise ProviderError("Connection error.")

    mock_stream.return_value = StreamResult(chunks(), "chat")
    mock_logger = MagicMock()
    agent = make_chatter(call_logger=mock_logger)

    result = agent.chat_stream("Hola")
    with pytest.raises(ProviderError):
        list(result)

    (record,) = _failed(mock_logger)
    assert "Connection error." in record.error


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
def test_act_in_batch_emits_call_log(mock_chat):
    """Batch runs emit one record per LLM call, not zero."""
    mock_chat.return_value = _batch_chat_response(["a", "b"])
    mock_logger = MagicMock()

    agent = make_labeler(call_logger=mock_logger)
    agent.act_in_batch([{"text": "one"}, {"text": "two"}], iter_func=None)

    # One chunk -> one LLM call -> one record, even though the
    # chunk expanded to two item responses.
    mock_logger.assert_called_once()
    record = mock_logger.call_args[0][0]
    assert isinstance(record, CallRecord)
    assert record.success is True
    assert record.tokens == 10


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_act_in_batch_async_emits_call_log(mock_chat):
    """Async batch runs emit call records too."""
    mock_chat.return_value = _batch_chat_response(["a", "b"])
    mock_logger = MagicMock()

    agent = make_labeler(call_logger=mock_logger)
    await agent.act_in_batch_async(
        [{"text": "one"}, {"text": "two"}], with_tqdm=False
    )

    mock_logger.assert_called_once()
    assert mock_logger.call_args[0][0].success is True


# --- The prompt on a failed call ---


@patch.object(LLMEngine, "chat")
def test_successful_call_records_no_prompt(mock_chat):
    """A call that parsed carries no prompt.

    Its prompt is derivable from the agent's config, and
    recording it on every row would dwarf the row.
    """
    mock_chat.return_value = _chat_response("ok")
    mock_logger = MagicMock()

    agent = make_labeler(call_logger=mock_logger)
    agent.act({"text": "hello"})

    assert mock_logger.call_args[0][0].prompt is None


@patch.object(LLMEngine, "chat")
def test_failed_call_records_the_messages_it_sent(mock_chat):
    """A failed call carries the messages, so it can be rerun.

    This is the whole point of the field: a re-run at
    temperature produces a different call, so a prompt not
    captured during the run cannot be recovered afterwards.
    """
    mock_chat.return_value = _bad_response()
    mock_logger = MagicMock()

    agent = make_labeler(call_logger=mock_logger)
    agent.act({"text": "hello"})

    record = mock_logger.call_args[0][0]
    assert record.success is False
    messages = json.loads(record.prompt)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Test prompt." in messages[0]["content"]
    assert "hello" in messages[1]["content"]


@patch.object(LLMEngine, "chat")
def test_recorded_prompt_survives_a_json_round_trip(mock_chat):
    """The prompt is a JSON string, so the record serializes.

    `to_dict` feeds `JsonCallLogger`, which json-dumps the
    whole record — a non-serializable prompt would break
    every failed row rather than just its own field.
    """
    mock_chat.return_value = _bad_response()
    mock_logger = MagicMock()

    agent = make_labeler(call_logger=mock_logger)
    agent.act({"text": "acentuación ñ"})

    record = mock_logger.call_args[0][0]
    reloaded = json.loads(json.dumps(record.to_dict()))
    assert "acentuación ñ" in json.loads(reloaded["prompt"])[1]["content"]


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_failed_async_call_records_the_prompt(mock_chat):
    """The async path records the prompt the sync path does."""
    mock_chat.return_value = _bad_response()
    mock_logger = MagicMock()

    agent = make_labeler(call_logger=mock_logger)
    await agent.act_async({"text": "hello"})

    record = mock_logger.call_args[0][0]
    assert record.success is False
    assert "hello" in json.loads(record.prompt)[1]["content"]


@patch.object(LLMEngine, "chat")
def test_failed_batch_chunk_records_its_own_prompt(mock_chat):
    """A batch chunk records the chunk it sent, not the item.

    The batch path builds its messages separately, so it is
    the one that regressed silently when the direct path was
    the only one wired up.
    """
    mock_chat.return_value = _bad_response()
    mock_logger = MagicMock()

    agent = make_labeler(call_logger=mock_logger)
    agent.act_in_batch([{"text": "one"}, {"text": "two"}], iter_func=None)

    record = mock_logger.call_args[0][0]
    assert record.success is False
    user = json.loads(record.prompt)[1]["content"]
    assert "one" in user and "two" in user


@patch.object(LLMEngine, "chat")
def test_successful_batch_chunk_records_no_prompt(mock_chat):
    """A batch chunk that parsed carries no prompt either."""
    mock_chat.return_value = _batch_chat_response(["a", "b"])
    mock_logger = MagicMock()

    agent = make_labeler(call_logger=mock_logger)
    agent.act_in_batch([{"text": "one"}, {"text": "two"}], iter_func=None)

    assert mock_logger.call_args[0][0].prompt is None


@patch.object(LLMEngine, "chat")
def test_no_logger_no_error(mock_chat):
    """No call_logger set does not raise."""
    mock_chat.return_value = _chat_response("ok")
    agent = make_labeler()
    result = agent.act({"text": "hello"})
    assert result.model is not None


# --- A parse retry is its own row ---


@patch.object(LLMEngine, "chat")
def test_each_parse_retry_gets_its_own_record(mock_chat):
    """Three LLM calls write three rows, not one.

    The retries' cost reached the response, so nothing went
    unreported — but all three calls were folded into a single
    record, so a log said one call where three were made, and
    a per-model average duration divided three calls' seconds
    by one.
    """
    mock_chat.side_effect = [
        _bad_response(),
        _bad_response(),
        _chat_response("finally"),
    ]
    records = []

    agent = make_labeler(max_parse_retries=2, call_logger=records.append)
    agent.act({"text": "hello"})

    assert mock_chat.call_count == 3
    assert len(records) == 3


@patch.object(LLMEngine, "chat")
def test_a_retry_record_carries_its_own_call_not_the_total(mock_chat):
    """Each row reports the call it is, so the rows add up.

    The response's own metrics are the total over all three
    calls. If a row carried that total too, a consumer summing
    the log would count the same tokens three times.
    """
    mock_chat.side_effect = [
        _bad_response(),
        _chat_response("fixed"),
    ]
    records = []

    agent = make_labeler(max_parse_retries=1, call_logger=records.append)
    result = agent.act({"text": "hello"})

    assert [r.tokens for r in records] == [
        DUMMY_METRICS["tokens"],
        DUMMY_METRICS["tokens"],
    ]
    assert result.metrics.tokens == 2 * DUMMY_METRICS["tokens"]


@patch.object(LLMEngine, "chat")
def test_the_calls_of_one_answer_share_a_trace_id(mock_chat):
    """One answer, one trace — as a guarded chat turn already did.

    Without it the three rows are unrelatable, and there is no
    way to ask what one answer cost.
    """
    mock_chat.side_effect = [
        _bad_response(),
        _bad_response(),
        _chat_response("finally"),
    ]
    records = []

    agent = make_labeler(max_parse_retries=2, call_logger=records.append)
    agent.act({"text": "hello"})

    assert len(records) == 3
    assert len({r.trace_id for r in records}) == 1


@patch.object(LLMEngine, "chat")
def test_only_the_answering_call_is_a_success(mock_chat):
    """The calls that would not parse are failures, and say so.

    The retry loop stops at the first call that parses, so
    every earlier one failed.
    """
    mock_chat.side_effect = [
        _bad_response(),
        _bad_response(),
        _chat_response("finally"),
    ]
    records = []

    agent = make_labeler(max_parse_retries=2, call_logger=records.append)
    agent.act({"text": "hello"})

    assert [r.success for r in records] == [False, False, True]


@patch.object(LLMEngine, "chat")
def test_an_answer_that_never_parsed_logs_only_failures(mock_chat):
    """Retries exhausted: every call failed, the last included."""
    mock_chat.return_value = _bad_response()
    records = []

    agent = make_labeler(max_parse_retries=1, call_logger=records.append)
    agent.act({"text": "hello"})

    assert [r.success for r in records] == [False, False]


@patch.object(LLMEngine, "chat")
def test_a_failed_retry_records_the_messages_that_retry_sent(mock_chat):
    """A retry's prompt is not the first call's.

    It carries the parse error fed back to the model, which is
    what has to be reproduced to see why the retry failed too.
    """
    mock_chat.side_effect = [
        _bad_response(),
        _bad_response(),
    ]
    records = []

    agent = make_labeler(max_parse_retries=1, call_logger=records.append)
    agent.act({"text": "hello"})

    first = json.loads(records[0].prompt)
    retry = json.loads(records[1].prompt)
    assert [m["role"] for m in first] == ["system", "user"]
    assert [m["role"] for m in retry] == ["system", "user", "user"]
    assert "could not be parsed" in retry[2]["content"]


@patch.object(LLMEngine, "chat")
def test_summarizing_a_retried_answer_counts_every_call(mock_chat):
    """`CallStats` reports three calls and three calls' seconds."""
    mock_chat.side_effect = [
        _bad_response(),
        _bad_response(),
        _chat_response("finally"),
    ]
    records = []

    agent = make_labeler(max_parse_retries=2, call_logger=records.append)
    agent.act({"text": "hello"})

    overall = summarize_call_log(records).overall
    assert overall.calls == 3
    assert overall.successes == 1
    assert overall.failures == 2
    assert overall.avg_duration == pytest.approx(
        DUMMY_METRICS["total_duration"]
    )


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_async_parse_retries_get_their_own_records(mock_chat):
    """The async path logs the rows the sync path does."""
    mock_chat.side_effect = [
        _bad_response(),
        _chat_response("fixed"),
    ]
    records = []

    agent = make_labeler(max_parse_retries=1, call_logger=records.append)
    await agent.act_async({"text": "hello"})

    assert [r.success for r in records] == [False, True]
    assert len({r.trace_id for r in records}) == 1


# --- Streamed calls ---


def _stream_chunks(text: str) -> list:
    """Chunks that stream *text* then report final metrics."""
    return [
        chat_chunk(content=text),
        chat_chunk(
            done=True,
            eval_count=10,
            prompt_eval_count=5,
            total_duration=2_000_000_000,
        ),
    ]


@patch.object(LLMEngine, "chat_stream")
def test_act_stream_emits_call_log(mock_stream):
    """A streamed call is logged like a plain one.

    It was not: `act_stream` returned the stream and never
    emitted a record, so a run made entirely of streamed calls
    wrote an empty log file.
    """
    mock_stream.return_value = StreamResult(
        iter(_stream_chunks('{"value": "ok"}')), "chat"
    )
    mock_logger = MagicMock()

    agent = make_labeler(call_logger=mock_logger)
    list(agent.act_stream({"text": "hello"}))

    mock_logger.assert_called_once()
    record = mock_logger.call_args[0][0]
    assert isinstance(record, CallRecord)
    assert record.success is True
    assert record.tokens == 10
    assert record.prompt_tokens == 5
    assert record.total_duration == 2.0


@patch.object(LLMEngine, "chat_stream")
def test_act_stream_logs_nothing_until_iteration_completes(mock_stream):
    """The record lands when the outcome is known, not before.

    Whether the call succeeded depends on parsing the streamed
    text, which cannot happen until the caller has finished
    reading it.
    """
    mock_stream.return_value = StreamResult(
        iter(_stream_chunks('{"value": "ok"}')), "chat"
    )
    mock_logger = MagicMock()

    agent = make_labeler(call_logger=mock_logger)
    stream = agent.act_stream({"text": "hello"})

    mock_logger.assert_not_called()
    list(stream)
    mock_logger.assert_called_once()


@patch.object(LLMEngine, "chat_stream")
def test_failed_stream_records_its_prompt(mock_stream):
    """An unparseable stream logs a failure with its messages."""
    mock_stream.return_value = StreamResult(
        iter(_stream_chunks("not json at all")), "chat"
    )
    mock_logger = MagicMock()

    agent = make_labeler(call_logger=mock_logger)
    list(agent.act_stream({"text": "hello"}))

    record = mock_logger.call_args[0][0]
    assert record.success is False
    messages = json.loads(record.prompt)
    assert [m["role"] for m in messages] == ["system", "user"]


@patch.object(LLMEngine, "chat_stream_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_act_stream_async_emits_call_log(mock_stream):
    """The async stream logs the same record the sync one does."""
    mock_stream.return_value = AsyncStreamResult(
        async_iter(_stream_chunks('{"value": "ok"}')), "chat"
    )
    mock_logger = MagicMock()

    agent = make_labeler(call_logger=mock_logger)
    stream = await agent.act_stream_async({"text": "hello"})
    async for _ in stream:
        pass

    mock_logger.assert_called_once()
    record = mock_logger.call_args[0][0]
    assert record.success is True
    assert record.tokens == 10


# --- summarize_call_log ---


def _record(
    agent="a",
    model="m",
    success=True,
    tokens=10,
    prompt_tokens=5,
    duration=0.1,
    error=None,
    cached=False,
    run=None,
):
    return CallRecord(
        trace_id="t",
        run_id=run,
        agent_name=agent,
        agent_version=1,
        model_id=model,
        success=success,
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        total_duration=duration,
        error=error,
        cached=cached,
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


def test_summarize_splits_by_run():
    """Two conversations in one log are two rows, not one.

    The breakdown a log written per-turn could not produce
    before records carried the run they belong to.
    """
    summary = summarize_call_log(
        [
            _record(run="cold", tokens=10),
            _record(run="cold", tokens=10),
            _record(run="warm", tokens=7),
        ]
    )
    assert summary.by_run["cold"].calls == 2
    assert summary.by_run["cold"].tokens == 20
    assert summary.by_run["warm"].calls == 1


def test_summarize_leaves_unnamed_runs_out_of_the_breakdown():
    """A log nobody named runs in has no runs to report.

    Bucketing the records that carried no run under a single
    key would invent a run that never existed, and would read
    as every call belonging to one.
    """
    summary = summarize_call_log([_record(), _record(run="warm")])

    assert summary.overall.calls == 2
    assert list(summary.by_run) == ["warm"]


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


def test_summarize_counts_cache_hits():
    """A hit's tokens are summed but were never billed.

    The count is what lets a reader tell a run that spent its
    token total from one that replayed most of it for free.
    """
    summary = summarize_call_log(
        [
            _record(cached=False),
            _record(cached=True),
            _record(cached=True),
        ]
    )

    assert summary.overall.calls == 3
    assert summary.overall.cached_calls == 2
    assert summary.by_agent["a"].cached_calls == 2


def test_format_says_unpriced_rather_than_zero():
    """No record carried a cost, so the run was not free."""
    text = summarize_call_log([_record()]).format()

    assert "Cost: unpriced" in text


def test_format_reports_cache_hits_when_there_are_any():
    """A clean run says nothing; a cached one says how many."""
    clean = summarize_call_log([_record()]).format()
    cached = summarize_call_log([_record(), _record(cached=True)]).format()

    assert "Cache:" not in clean
    assert "Cache: 1 of 2 calls served from cache" in cached


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


# ── SqliteCallLogger Tests ─────────────────────────────────


def test_sqlite_logger_writes_no_file_until_a_call_is_logged():
    """A logger that never writes leaves no database behind.

    Same contract as `JsonCallLogger`: a run that logged nothing
    should not leave an empty artefact for someone to wonder
    about later.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nested" / "calls.db"
        SqliteCallLogger(path).close()
        assert not path.exists()


def test_sqlite_logger_roundtrips_a_record(tmp_path):
    """Every field written comes back with its own type.

    Booleans go to SQLite as integers, so the read side has to
    convert them back — a `success` of 1 is not a `success` of
    True to anything comparing them.
    """
    path = tmp_path / "calls.db"
    record = CallRecord(
        trace_id="t1",
        agent_name="Extractor",
        agent_version=3,
        model_id="ollama::qwen3",
        success=False,
        tokens=9,
        prompt_tokens=6239,
        cached_prompt_tokens=6100,
        total_duration=1.5,
        cost_cents=0.0124,
        cached=True,
        retries=2,
        retry_duration=4.0,
        error="ParseError: nope",
        prompt='[{"role": "user", "content": "hi"}]',
        run_id="cold-read",
        recorded_at="2026-08-30T10:00:00+00:00",
    )
    with SqliteCallLogger(path) as logger:
        logger(record)

    assert list(read_call_log(path)) == [record]


def test_sqlite_logger_keeps_the_order_calls_were_made_in(tmp_path):
    """Rows come back oldest first, as the JSONL sink does."""
    path = tmp_path / "calls.db"
    with SqliteCallLogger(path) as logger:
        for i in range(3):
            logger(_record(agent=f"a{i}", run="r"))

    assert [r.agent_name for r in read_call_log(path)] == ["a0", "a1", "a2"]


def test_an_unpriced_call_stays_unpriced_through_sqlite(tmp_path):
    """`None` must not become `0.0` on the way to disk.

    A nullable column is the whole reason the cost is stored as
    one: an engine that declares no pricing has to read back as
    unpriced rather than as free.
    """
    path = tmp_path / "calls.db"
    with SqliteCallLogger(path) as logger:
        logger(_record())

    read = list(read_call_log(path))
    assert read[0].cost_cents is None
    assert summarize_call_log(path).overall.cost_cents is None


def test_summarize_reads_a_database_as_readily_as_a_file(tmp_path):
    """The summary does not care which sink wrote the log.

    The two are told apart by SQLite's file header, not by the
    file's name — a log is named by whoever configured it, and
    parsing a database as JSON lines fails with a decoding error
    that says nothing about the real cause.
    """
    db, jsonl = tmp_path / "calls.db", tmp_path / "calls.jsonl"
    records = [_record(run="cold"), _record(run="warm", success=False)]
    with SqliteCallLogger(db) as sqlite_logger:
        with JsonCallLogger(jsonl) as json_logger:
            for record in records:
                sqlite_logger(record)
                json_logger(record)

    from_db = summarize_call_log(db)
    from_file = summarize_call_log(jsonl)

    assert from_db.overall == from_file.overall
    assert from_db.by_run["cold"].calls == 1
    assert from_db.overall.failures == 1


def test_reading_a_missing_call_log_says_so(tmp_path):
    """A clearer failure than a decode error on an absent file."""
    with pytest.raises(FileNotFoundError):
        list(read_call_log(tmp_path / "nope.db"))


def test_a_second_logger_appends_to_an_existing_database(tmp_path):
    """Reopening a log adds to it rather than replacing it.

    A chat agent logging over months opens the file once per
    process, so every run after the first is an append.
    """
    path = tmp_path / "calls.db"
    with SqliteCallLogger(path) as first:
        first(_record(agent="first"))
    with SqliteCallLogger(path) as second:
        second(_record(agent="second"))

    assert [r.agent_name for r in read_call_log(path)] == [
        "first",
        "second",
    ]


# ── Opening a database an older version wrote ──────────────────────────────


def _legacy_schema() -> str:
    """The `calls` table as it stood before `model` was added.

    Written out rather than derived, because the point is to
    reproduce a file some earlier version actually created.
    """
    return """
    CREATE TABLE IF NOT EXISTS calls (
        row_id               INTEGER PRIMARY KEY AUTOINCREMENT,
        trace_id             TEXT    NOT NULL,
        agent_name           TEXT    NOT NULL,
        agent_version        INTEGER NOT NULL,
        model_id             TEXT    NOT NULL,
        success              INTEGER NOT NULL,
        tokens               INTEGER NOT NULL,
        prompt_tokens        INTEGER NOT NULL,
        cached_prompt_tokens INTEGER NOT NULL,
        total_duration       REAL    NOT NULL,
        cost_cents           REAL,
        cached               INTEGER NOT NULL,
        retries              INTEGER NOT NULL,
        retry_duration       REAL    NOT NULL,
        error                TEXT,
        prompt               TEXT,
        run_id               TEXT,
        recorded_at          TEXT
    );
    """


def test_an_older_database_gains_the_columns_it_predates(tmp_path):
    """A log written before a field existed keeps accepting rows.

    ``CREATE TABLE IF NOT EXISTS`` leaves an existing table
    exactly as it was, so adding a field made every insert name a
    column the file had never heard of and fail with
    ``table calls has no column named model``. A chat agent
    logging every turn for months must not stop logging because
    the framework learned to measure one more thing.
    """
    path = tmp_path / "calls.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(_legacy_schema())
    conn.commit()
    conn.close()

    logger = SqliteCallLogger(path)
    logger(
        CallRecord(
            trace_id="t1",
            agent_name="Extractor",
            agent_version=3,
            model_id="ollama::qwen3",
            success=True,
            model="ollama::qwen2.5:3b",
        )
    )
    logger.close()

    records = list(read_call_log(path))
    assert len(records) == 1
    assert records[0].model == "ollama::qwen2.5:3b"


def test_rows_written_before_a_field_existed_read_back_as_unmeasured(
    tmp_path,
):
    """An old row reports `None`, never a stand-in value.

    Nothing measured the model when those rows were written, and
    ``None`` is what says so. A default of the configured name
    would make an unmeasured row indistinguishable from one that
    really was answered by that model.

    Read without opening the file for writing, because reading
    someone's log must not require write access to it and an
    archived log should not be rewritten by being looked at.
    """
    path = tmp_path / "calls.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(_legacy_schema())
    conn.execute(
        "INSERT INTO calls (trace_id, agent_name, agent_version, "
        "model_id, success, tokens, prompt_tokens, "
        "cached_prompt_tokens, total_duration, cached, retries, "
        "retry_duration) "
        "VALUES ('t0', 'Old', 1, 'ollama::qwen3', 1, 9, 5, 0, 1.5, 0, 0, 0)"
    )
    conn.commit()
    conn.close()

    records = list(read_call_log(path))

    assert len(records) == 1
    assert records[0].model is None
