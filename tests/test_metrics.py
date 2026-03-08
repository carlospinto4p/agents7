"""Unit tests for agents7.metrics (core evaluation module)."""

from unittest.mock import MagicMock

from agents7.metrics import (
    EvalResult,
    _compare_field,
    build_confusion_matrix,
    compute_metrics,
    run_evaluation,
)
from agents7.models.base import AgentResponse


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


# --- run_evaluation ---


def test_run_evaluation_all_pass():
    """Evaluation with all correct predictions."""
    mock_model = MagicMock()
    mock_model.category = MagicMock(name="POSITIVE")
    mock_model.category.name = "POSITIVE"

    mock_response = MagicMock(spec=AgentResponse)
    mock_response.model = mock_model

    mock_agent = MagicMock()
    mock_agent.act.return_value = mock_response

    data = [{"input": {"text": "great"}, "category": "POSITIVE"}]
    results = run_evaluation(mock_agent, data, ["category"])

    assert len(results) == 1
    assert results[0].passed is True


def test_run_evaluation_agent_error():
    """Evaluation handles agent exceptions."""
    mock_agent = MagicMock()
    mock_agent.act.side_effect = RuntimeError("boom")

    data = [{"input": {"text": "test"}, "category": "A"}]
    results = run_evaluation(mock_agent, data, ["category"])

    assert len(results) == 1
    assert results[0].error == "boom"
    assert results[0].passed is False
