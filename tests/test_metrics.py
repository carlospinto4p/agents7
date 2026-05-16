"""Unit tests for agents7.metrics (core evaluation module)."""

import json
from enum import StrEnum
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from agents7.agents.labeler import Labeler
from agents7.eval.metrics import (
    EvalResult,
    _build_eval_result,
    _compare_field,
    _extract_fields,
    _fallback_sequential,
    build_confusion_matrix,
    compute_metrics,
    run_evaluation,
    run_evaluation_async,
)
from agents7.models.base import AgentResponse, LLMMetrics

from helpers import DUMMY_METRICS, DummyOutput, make_labeler

# --- _compare_field ---


def test_compare_same_values():
    """Identical strings match."""
    assert _compare_field("POSITIVE", "POSITIVE") is True


def test_compare_case_insensitive():
    """Comparison is case-insensitive."""
    assert _compare_field("positive", "POSITIVE") is True


def test_compare_different_values():
    """Different strings do not match."""
    assert _compare_field("POSITIVE", "NEGATIVE") is False


def test_compare_both_none():
    """Both None is a match."""
    assert _compare_field(None, None) is True


def test_compare_one_none():
    """One None does not match."""
    assert _compare_field(None, "POSITIVE") is False
    assert _compare_field("POSITIVE", None) is False


# --- compute_metrics ---


def test_metrics_perfect_score():
    """All passed gives 1.0 accuracy."""
    results = [
        EvalResult(
            input_data={},
            expected={"cat": "A"},
            actual={"cat": "A"},
            passed=True,
            field_results={"cat": True},
        ),
        EvalResult(
            input_data={},
            expected={"cat": "B"},
            actual={"cat": "B"},
            passed=True,
            field_results={"cat": True},
        ),
    ]
    m = compute_metrics(results, ["cat"])
    assert m["total"] == 2
    assert m["passed"] == 2
    assert m["accuracy"] == 1.0
    assert m["per_field"]["cat"] == 1.0


def test_metrics_half_score():
    """Half passed gives 0.5 accuracy."""
    results = [
        EvalResult(
            input_data={},
            expected={"cat": "A"},
            passed=True,
            field_results={"cat": True},
        ),
        EvalResult(
            input_data={},
            expected={"cat": "B"},
            passed=False,
            field_results={"cat": False},
        ),
    ]
    m = compute_metrics(results, ["cat"])
    assert m["accuracy"] == 0.5
    assert m["failed"] == 1


def test_metrics_empty_results():
    """Empty results give zeros."""
    m = compute_metrics([], ["cat"])
    assert m["total"] == 0
    assert m["accuracy"] == 0.0


def test_metrics_timing_included():
    """compute_metrics includes timing stats from response metrics."""
    metrics = LLMMetrics(tokens=5, prompt_tokens=3, total_duration=0.4)
    response = AgentResponse(response="{}", model=None, metrics=metrics)
    results = [
        EvalResult(
            input_data={},
            expected={"cat": "A"},
            passed=True,
            field_results={"cat": True},
            response=response,
        ),
    ]
    m = compute_metrics(results, ["cat"])
    assert m["timing"]["total_s"] == pytest.approx(0.4)
    assert m["timing"]["avg_s"] == pytest.approx(0.4)
    assert m["timing"]["min_s"] == pytest.approx(0.4)
    assert m["timing"]["max_s"] == pytest.approx(0.4)
    assert m["timing"]["per_case"] == [pytest.approx(0.4)]


def test_metrics_timing_no_response_uses_zero():
    """Rows without a response contribute 0.0 to timing."""
    results = [
        EvalResult(
            input_data={},
            expected={"cat": "A"},
            passed=False,
            error="Model parsing failed",
        ),
    ]
    m = compute_metrics(results, ["cat"])
    assert m["timing"]["total_s"] == 0.0
    assert m["timing"]["per_case"] == [0.0]


# --- build_confusion_matrix ---


def test_confusion_matrix_basic():
    """Basic 2x2 confusion matrix."""
    results = [
        EvalResult(
            input_data={},
            expected={"cat": "A"},
            actual={"cat": "A"},
        ),
        EvalResult(
            input_data={},
            expected={"cat": "A"},
            actual={"cat": "B"},
        ),
        EvalResult(
            input_data={},
            expected={"cat": "B"},
            actual={"cat": "B"},
        ),
    ]
    cm = build_confusion_matrix(results, "cat")
    assert "A" in cm["labels"]
    assert "B" in cm["labels"]
    assert len(cm["matrix"]) == 2


def test_confusion_matrix_empty():
    """Empty results give empty matrix."""
    cm = build_confusion_matrix([], "cat")
    assert cm["labels"] == []
    assert cm["matrix"] == []


# --- _build_eval_result ---


def test_build_eval_result_model_none_records_error():
    """When response.model is None, error is set to 'Model parsing failed'."""
    metrics = LLMMetrics(tokens=5, prompt_tokens=3, total_duration=0.1)
    response = AgentResponse(response="{}", model=None, metrics=metrics)
    row = {"input": {"text": "x"}, "cat": "A"}
    result = _build_eval_result(row, ["cat"], response)
    assert result.error == "Model parsing failed"
    assert result.passed is False


def test_build_eval_result_with_error_arg():
    """When error is supplied directly, it propagates to EvalResult."""
    row = {"input": {"text": "x"}, "cat": "A"}
    result = _build_eval_result(row, ["cat"], None, error="timeout")
    assert result.error == "timeout"
    assert result.passed is False
    assert result.response is None


# --- _extract_fields ---


def test_extract_fields_plain_values():
    class _Out(BaseModel):
        value: str
        score: int

    out = _Out(value="hello", score=42)
    extracted = _extract_fields(out, ["value", "score"])
    assert extracted == {"value": "hello", "score": 42}


def test_extract_fields_enum_uses_name():
    class _Label(StrEnum):
        YES = "Yes it is."
        NO = "No it is not."

    class _Out(BaseModel):
        label: _Label

    out = _Out(label=_Label.YES)
    extracted = _extract_fields(out, ["label"])
    assert extracted["label"] == "YES"


def test_extract_fields_missing_field_returns_none():
    class _Out(BaseModel):
        value: str

    out = _Out(value="x")
    extracted = _extract_fields(out, ["value", "nonexistent"])
    assert extracted["nonexistent"] is None


# --- _fallback_sequential ---


@patch("agents7.llm.base.LLMEngine.chat")
def test_fallback_sequential_row_exception_records_error(mock_chat):
    """A per-row exception in sequential fallback sets EvalResult.error."""
    mock_chat.side_effect = RuntimeError("row failed")
    agent = make_labeler()
    data = [{"input": {"text": "x"}, "value": "a"}]
    results = _fallback_sequential(agent, data, ["value"])
    assert len(results) == 1
    assert results[0].error == "row failed"
    assert results[0].passed is False


@patch("agents7.llm.base.LLMEngine.chat")
def test_fallback_sequential_partial_failure(mock_chat):
    """Only failing rows get errors; passing rows still succeed."""
    mock_chat.side_effect = [
        RuntimeError("row 0 failed"),
        {
            "role": "assistant",
            "response": json.dumps({"value": "ok"}),
            "thinking": None,
            "metrics": DUMMY_METRICS,
        },
    ]
    agent = make_labeler()
    data = [
        {"input": {"text": "x"}, "value": "a"},
        {"input": {"text": "y"}, "value": "ok"},
    ]
    results = _fallback_sequential(agent, data, ["value"])
    assert results[0].error is not None
    assert results[1].passed is True


# --- run_evaluation (batch-based) ---


def _make_batch_response(values):
    """Build a mock batch chat response for act_in_batch."""
    return {
        "role": "assistant",
        "response": json.dumps({"items": [{"value": v} for v in values]}),
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }


@patch("agents7.llm.base.LLMEngine.chat")
def test_run_evaluation_uses_batch(mock_chat):
    """run_evaluation uses act_in_batch internally."""
    mock_chat.return_value = _make_batch_response(["a", "b"])
    agent = make_labeler()

    data = [
        {"input": {"text": "x"}, "value": "a"},
        {"input": {"text": "y"}, "value": "b"},
    ]
    results = run_evaluation(agent, data, ["value"])

    assert len(results) == 2
    assert results[0].passed is True
    assert results[1].passed is True


@patch("agents7.llm.base.LLMEngine.chat")
def test_run_evaluation_partial_match(mock_chat):
    """Mismatched fields produce passed=False."""
    mock_chat.return_value = _make_batch_response(["a", "wrong"])
    agent = make_labeler()

    data = [
        {"input": {"text": "x"}, "value": "a"},
        {"input": {"text": "y"}, "value": "b"},
    ]
    results = run_evaluation(agent, data, ["value"])

    assert results[0].passed is True
    assert results[1].passed is False


@patch("agents7.llm.base.LLMEngine.chat")
def test_run_evaluation_batch_failure_fallback(mock_chat):
    """Falls back to sequential act() when batch fails."""
    mock_chat.side_effect = [
        RuntimeError("batch failed"),
        {
            "role": "assistant",
            "response": json.dumps({"value": "ok"}),
            "thinking": None,
            "metrics": DUMMY_METRICS,
        },
    ]
    agent = make_labeler()

    data = [{"input": {"text": "x"}, "value": "ok"}]
    results = run_evaluation(agent, data, ["value"])

    assert len(results) == 1
    assert results[0].passed is True


@patch("agents7.llm.base.LLMEngine.chat")
def test_run_evaluation_progress_callback(mock_chat):
    """Progress callback is called with (total, total)."""
    mock_chat.return_value = _make_batch_response(["a"])
    agent = make_labeler()

    callback = MagicMock()
    data = [{"input": {"text": "x"}, "value": "a"}]
    run_evaluation(agent, data, ["value"], progress_callback=callback)

    callback.assert_called_once_with(1, 1)


@patch("agents7.llm.base.LLMEngine.chat")
def test_run_evaluation_empty_data(mock_chat):
    """Empty eval_data returns empty list without calling the agent."""
    agent = make_labeler()
    results = run_evaluation(agent, [], ["value"])
    assert results == []
    mock_chat.assert_not_called()


# --- run_evaluation_async ---


@patch("agents7.llm.base.LLMEngine.chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_run_evaluation_async_uses_batch(mock_chat):
    """Async evaluation uses act_in_batch_async."""
    mock_chat.return_value = _make_batch_response(["a", "b"])
    agent = make_labeler()

    data = [
        {"input": {"text": "x"}, "value": "a"},
        {"input": {"text": "y"}, "value": "b"},
    ]
    results = await run_evaluation_async(agent, data, ["value"])

    assert len(results) == 2
    assert results[0].passed is True
    assert results[1].passed is True


@patch("agents7.llm.base.LLMEngine.chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_run_evaluation_async_progress_callback(mock_chat):
    """Async progress callback is called."""
    mock_chat.return_value = _make_batch_response(["a"])
    agent = make_labeler()

    callback = MagicMock()
    data = [{"input": {"text": "x"}, "value": "a"}]
    await run_evaluation_async(
        agent, data, ["value"], progress_callback=callback
    )

    callback.assert_called_once_with(1, 1)


@patch("agents7.llm.base.LLMEngine.chat")
@patch("agents7.llm.base.LLMEngine.chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_run_evaluation_async_batch_failure_falls_back(
    mock_chat_async, mock_chat
):
    """Async batch failure triggers sequential fallback (sync act())."""
    mock_chat_async.side_effect = RuntimeError("async batch failed")
    mock_chat.return_value = {
        "role": "assistant",
        "response": json.dumps({"value": "ok"}),
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }
    agent = make_labeler()

    data = [{"input": {"text": "x"}, "value": "ok"}]
    results = await run_evaluation_async(agent, data, ["value"])

    assert len(results) == 1
    assert results[0].passed is True


@patch("agents7.llm.base.LLMEngine.chat")
@patch("agents7.llm.base.LLMEngine.chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_run_evaluation_async_row_error_recorded(
    mock_chat_async, mock_chat
):
    """Per-row exception in async sequential fallback sets error on result."""
    mock_chat_async.side_effect = RuntimeError("batch boom")
    mock_chat.side_effect = RuntimeError("row boom")
    agent = make_labeler()

    data = [{"input": {"text": "x"}, "value": "a"}]
    results = await run_evaluation_async(agent, data, ["value"])

    assert len(results) == 1
    assert results[0].error == "row boom"
    assert results[0].passed is False


@patch("agents7.llm.base.LLMEngine.chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_run_evaluation_async_empty_data(mock_chat):
    """Empty eval_data returns empty list without calling the agent."""
    agent = make_labeler()
    results = await run_evaluation_async(agent, [], ["value"])
    assert results == []
    mock_chat.assert_not_called()


@patch("agents7.llm.base.LLMEngine.chat")
@patch("agents7.llm.base.LLMEngine.chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_run_evaluation_async_progress_callback_called_on_fallback(
    mock_chat_async, mock_chat
):
    """Progress callback fires even when async batch fails and falls back."""
    mock_chat_async.side_effect = RuntimeError("batch boom")
    mock_chat.return_value = {
        "role": "assistant",
        "response": json.dumps({"value": "a"}),
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }
    agent = make_labeler()
    callback = MagicMock()
    data = [{"input": {"text": "x"}, "value": "a"}]
    await run_evaluation_async(
        agent, data, ["value"], progress_callback=callback
    )
    callback.assert_called_once_with(1, 1)


# --- partial batch completion ---


@patch.object(Labeler, "act_in_batch")
def test_run_evaluation_partial_batch_model_none(mock_batch):
    """Batch rows with model=None are individually marked as errors."""
    metrics = LLMMetrics(tokens=5, prompt_tokens=3, total_duration=0.1)
    good = AgentResponse(
        response=json.dumps({"value": "ok"}),
        model=DummyOutput(value="ok"),
        thinking=None,
        metrics=metrics,
    )
    bad = AgentResponse(
        response="{}",
        model=None,
        thinking=None,
        metrics=metrics,
    )
    mock_batch.return_value = [good, bad]

    agent = make_labeler()
    data = [
        {"input": {"text": "x"}, "value": "ok"},
        {"input": {"text": "y"}, "value": "a"},
    ]
    results = run_evaluation(agent, data, ["value"])

    assert len(results) == 2
    assert results[0].passed is True
    assert results[0].error is None
    assert results[1].passed is False
    assert results[1].error == "Model parsing failed"
