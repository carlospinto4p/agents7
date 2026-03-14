"""Unit tests for agents7.metrics (core evaluation module)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents7.metrics import (
    EvalResult,
    _compare_field,
    build_confusion_matrix,
    compute_metrics,
    run_evaluation,
    run_evaluation_async,
)

from helpers import DUMMY_METRICS, make_labeler

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
