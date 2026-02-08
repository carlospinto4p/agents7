"""Unit tests for agents7.ui.evaluation (runner + metrics)."""

from unittest.mock import MagicMock

from pydantic import BaseModel

from agents7.models.base import AgentResponse, LLMMetrics
from agents7.ui.evaluation.runner import (
    EvalResult,
    run_evaluation,
    _compare_field,
)
from agents7.ui.evaluation.metrics import (
    build_confusion_matrix,
    compute_metrics,
)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


class _MockModel(BaseModel):
    category: str = "POSITIVE"
    reason: str = "HAPPY"


def _mock_metrics() -> LLMMetrics:
    return LLMMetrics(tokens=10, prompt_tokens=5, total_duration=0.1)


def _make_response(
    category: str = "POSITIVE", reason: str = "HAPPY"
) -> AgentResponse:
    model = _MockModel(category=category, reason=reason)
    return AgentResponse(
        response="{}",
        model=model,
        metrics=_mock_metrics(),
    )


def _make_agent_mock(responses: list[AgentResponse]):
    """Return a mock agent that returns responses in order."""
    agent = MagicMock()
    agent.act = MagicMock(side_effect=responses)
    return agent


# ================================================================
# _compare_field
# ================================================================


class TestCompareField:
    def test_same_values(self):
        assert _compare_field("POSITIVE", "POSITIVE")

    def test_case_insensitive(self):
        assert _compare_field("positive", "POSITIVE")

    def test_different_values(self):
        assert not _compare_field("POSITIVE", "NEGATIVE")

    def test_none_equal(self):
        assert _compare_field(None, None)

    def test_none_not_equal(self):
        assert not _compare_field(None, "POSITIVE")


# ================================================================
# run_evaluation
# ================================================================


class TestRunEvaluation:
    def test_all_pass(self):
        responses = [
            _make_response("POSITIVE", "HAPPY"),
            _make_response("NEGATIVE", "SAD"),
        ]
        agent = _make_agent_mock(responses)
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

    def test_partial_fail(self):
        responses = [
            _make_response("NEGATIVE", "HAPPY"),
        ]
        agent = _make_agent_mock(responses)
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

    def test_model_none(self):
        response = AgentResponse(
            response="{}",
            model=None,
            metrics=_mock_metrics(),
        )
        agent = _make_agent_mock([response])
        eval_data = [
            {
                "input": {"text": "bad"},
                "category": "POSITIVE",
            },
        ]
        results = run_evaluation(agent, eval_data, ["category"])
        assert not results[0].passed
        assert results[0].error == "Model parsing failed"

    def test_agent_exception(self):
        agent = MagicMock()
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

    def test_progress_callback(self):
        responses = [_make_response(), _make_response()]
        agent = _make_agent_mock(responses)
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
        assert callback.call_count == 2
        callback.assert_any_call(1, 2)
        callback.assert_any_call(2, 2)

    def test_empty_eval_data(self):
        agent = MagicMock()
        results = run_evaluation(agent, [], ["category"])
        assert results == []


# ================================================================
# compute_metrics
# ================================================================


class TestComputeMetrics:
    def test_perfect_score(self):
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

    def test_half_score(self):
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

    def test_empty_results(self):
        metrics = compute_metrics([], ["category"])
        assert metrics["total"] == 0
        assert metrics["accuracy"] == 0.0

    def test_multiple_fields(self):
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


# ================================================================
# build_confusion_matrix
# ================================================================


class TestBuildConfusionMatrix:
    def test_basic_matrix(self):
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

    def test_empty_results(self):
        cm = build_confusion_matrix([], "category")
        assert cm["labels"] == []
        assert cm["matrix"] == []

    def test_none_actual(self):
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

    def test_missing_expected_field(self):
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
