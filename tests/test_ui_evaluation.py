"""Unit tests for agents7.ui.evaluation (runner + metrics)."""

from unittest.mock import MagicMock

from agents7.models.base import AgentResponse
from agents7.ui.evaluation.runner import (
    EvalResult,
    run_evaluation,
    _compare_field,
)
from agents7.ui.evaluation.metrics import (
    build_confusion_matrix,
    compute_metrics,
)

from helpers import make_agent_mock, make_response, mock_metrics


# --- _compare_field ---


def test_same_values():
    assert _compare_field("POSITIVE", "POSITIVE")


def test_case_insensitive():
    assert _compare_field("positive", "POSITIVE")


def test_different_values():
    assert not _compare_field("POSITIVE", "NEGATIVE")


def test_none_equal():
    assert _compare_field(None, None)


def test_none_not_equal():
    assert not _compare_field(None, "POSITIVE")


# --- run_evaluation ---


def test_all_pass():
    responses = [
        make_response("POSITIVE", "HAPPY"),
        make_response("NEGATIVE", "SAD"),
    ]
    agent = make_agent_mock(responses)
    eval_data = [
        {
            "input": {"text": "great"},
            "category": "POSITIVE",
            "reason": "HAPPY",
        },
        {
            "input": {"text": "awful"},
            "category": "NEGATIVE",
            "reason": "SAD",
        },
    ]
    results = run_evaluation(agent, eval_data, ["category", "reason"])
    assert len(results) == 2
    assert all(r.passed for r in results)


def test_partial_fail():
    responses = [
        make_response("NEGATIVE", "HAPPY"),
    ]
    agent = make_agent_mock(responses)
    eval_data = [
        {
            "input": {"text": "great"},
            "category": "POSITIVE",
            "reason": "HAPPY",
        },
    ]
    results = run_evaluation(agent, eval_data, ["category", "reason"])
    assert not results[0].passed
    assert results[0].field_results["category"] is False
    assert results[0].field_results["reason"] is True


def test_model_none():
    response = AgentResponse(
        response="{}",
        model=None,
        metrics=mock_metrics(),
    )
    agent = MagicMock()
    agent.act_in_batch = MagicMock(return_value=[response])
    eval_data = [
        {
            "input": {"text": "bad"},
            "category": "POSITIVE",
        },
    ]
    results = run_evaluation(agent, eval_data, ["category"])
    assert not results[0].passed
    assert results[0].error == "Model parsing failed"


def test_agent_exception():
    agent = MagicMock()
    agent.act_in_batch = MagicMock(side_effect=RuntimeError("batch error"))
    agent.act = MagicMock(side_effect=RuntimeError("connection error"))
    eval_data = [
        {
            "input": {"text": "test"},
            "category": "X",
        },
    ]
    results = run_evaluation(agent, eval_data, ["category"])
    assert not results[0].passed
    assert "connection error" in results[0].error


def test_progress_callback():
    responses = [make_response(), make_response()]
    agent = make_agent_mock(responses)
    callback = MagicMock()
    eval_data = [
        {"input": {"text": "a"}, "category": "POSITIVE"},
        {"input": {"text": "b"}, "category": "POSITIVE"},
    ]
    run_evaluation(
        agent,
        eval_data,
        ["category"],
        progress_callback=callback,
    )
    callback.assert_called_once_with(2, 2)


def test_empty_eval_data():
    agent = MagicMock()
    results = run_evaluation(agent, [], ["category"])
    assert results == []


# --- compute_metrics ---


def test_perfect_score():
    results = [
        EvalResult(
            input_data={},
            expected={"category": "A"},
            actual={"category": "A"},
            passed=True,
            field_results={"category": True},
        ),
        EvalResult(
            input_data={},
            expected={"category": "B"},
            actual={"category": "B"},
            passed=True,
            field_results={"category": True},
        ),
    ]
    metrics = compute_metrics(results, ["category"])
    assert metrics["total"] == 2
    assert metrics["passed"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["per_field"]["category"] == 1.0


def test_half_score():
    results = [
        EvalResult(
            input_data={},
            expected={"category": "A"},
            actual={"category": "A"},
            passed=True,
            field_results={"category": True},
        ),
        EvalResult(
            input_data={},
            expected={"category": "B"},
            actual={"category": "A"},
            passed=False,
            field_results={"category": False},
        ),
    ]
    metrics = compute_metrics(results, ["category"])
    assert metrics["accuracy"] == 0.5
    assert metrics["per_field"]["category"] == 0.5


def test_empty_results():
    metrics = compute_metrics([], ["category"])
    assert metrics["total"] == 0
    assert metrics["accuracy"] == 0.0


def test_multiple_fields():
    results = [
        EvalResult(
            input_data={},
            expected={"cat": "A", "rsn": "X"},
            actual={"cat": "A", "rsn": "Y"},
            passed=False,
            field_results={"cat": True, "rsn": False},
        ),
    ]
    metrics = compute_metrics(results, ["cat", "rsn"])
    assert metrics["per_field"]["cat"] == 1.0
    assert metrics["per_field"]["rsn"] == 0.0


# --- build_confusion_matrix ---


def test_basic_matrix():
    results = [
        EvalResult(
            input_data={},
            expected={"category": "A"},
            actual={"category": "A"},
            passed=True,
            field_results={"category": True},
        ),
        EvalResult(
            input_data={},
            expected={"category": "A"},
            actual={"category": "B"},
            passed=False,
            field_results={"category": False},
        ),
        EvalResult(
            input_data={},
            expected={"category": "B"},
            actual={"category": "B"},
            passed=True,
            field_results={"category": True},
        ),
    ]
    cm = build_confusion_matrix(results, "category")
    assert "A" in cm["labels"]
    assert "B" in cm["labels"]
    # Matrix is labels x labels
    assert len(cm["matrix"]) == 2
    assert len(cm["matrix"][0]) == 2


def test_confusion_empty_results():
    cm = build_confusion_matrix([], "category")
    assert cm["labels"] == []
    assert cm["matrix"] == []


def test_none_actual():
    results = [
        EvalResult(
            input_data={},
            expected={"category": "A"},
            actual=None,
            passed=False,
            field_results={"category": False},
        ),
    ]
    cm = build_confusion_matrix(results, "category")
    assert "_NONE_" in cm["labels"]


def test_missing_expected_field():
    results = [
        EvalResult(
            input_data={},
            expected={"other": "X"},
            actual={"category": "A"},
            passed=False,
            field_results={},
        ),
    ]
    cm = build_confusion_matrix(results, "category")
    assert cm["labels"] == []
