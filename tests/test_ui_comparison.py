"""Unit tests for agents7.ui.comparison (models + runner)."""

from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from agents7.models.base import AgentResponse, LLMMetrics
from agents7.ui.comparison.models import PromptVersion
from agents7.ui.comparison.runner import (
    ComparisonResult,
    _build_version_state,
    run_comparison,
)
from agents7.ui.state import AgentBuilderState


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


class _MockModel(BaseModel):
    category: str = "POSITIVE"


def _mock_metrics() -> LLMMetrics:
    return LLMMetrics(tokens=10, prompt_tokens=5, total_duration=0.1)


def _make_response(
    category: str = "POSITIVE",
) -> AgentResponse:
    model = _MockModel(category=category)
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
# PromptVersion
# ================================================================


class TestPromptVersion:
    def test_creation_defaults(self):
        v = PromptVersion(name="v1")
        assert v.name == "v1"
        assert v.system_prompt == ""
        assert v.examples == []
        assert v.model_name == "qwen3"

    def test_creation_all_fields(self):
        v = PromptVersion(
            name="v2",
            system_prompt="Classify this.",
            examples=[{"input": "hi", "output": "pos"}],
            model_name="llama3",
        )
        assert v.name == "v2"
        assert v.system_prompt == "Classify this."
        assert len(v.examples) == 1
        assert v.model_name == "llama3"


# ================================================================
# _build_version_state
# ================================================================


class TestBuildVersionState:
    def test_replaces_prompt_fields(self):
        base = AgentBuilderState(
            agent_type="classifier",
            agent_name="test_agent",
            system_prompt="original",
            examples=[],
            model_name="qwen3",
        )
        version = PromptVersion(
            name="v1",
            system_prompt="new prompt",
            examples=[{"input": "x", "output": "y"}],
            model_name="llama3",
        )
        result = _build_version_state(base, version)
        assert result.system_prompt == "new prompt"
        assert result.examples == [{"input": "x", "output": "y"}]
        assert result.model_name == "llama3"

    def test_preserves_non_prompt_fields(self):
        base = AgentBuilderState(
            agent_type="detector",
            agent_name="my_agent",
            system_prompt="orig",
            model_name="qwen3",
        )
        version = PromptVersion(
            name="v1",
            system_prompt="changed",
            model_name="llama3",
        )
        result = _build_version_state(base, version)
        assert result.agent_type == "detector"
        assert result.agent_name == "my_agent"


# ================================================================
# ComparisonResult
# ================================================================


class TestComparisonResult:
    def test_defaults(self):
        r = ComparisonResult(input_data={"text": "hi"})
        assert r.input_data == {"text": "hi"}
        assert r.output_a is None
        assert r.output_b is None
        assert r.metrics_a == {}
        assert r.metrics_b == {}
        assert r.error_a is None
        assert r.error_b is None
        assert r.outputs_match is False

    def test_outputs_match_true(self):
        r = ComparisonResult(
            input_data={"text": "hi"},
            output_a={"category": "POS"},
            output_b={"category": "POS"},
            outputs_match=True,
        )
        assert r.outputs_match is True


# ================================================================
# run_comparison
# ================================================================


class TestRunComparison:
    @patch("agents7.ui.comparison.runner.build_agent_from_state")
    def test_matching_outputs(self, mock_build):
        resp = _make_response("POSITIVE")
        agent = _make_agent_mock([resp, resp])
        mock_build.return_value = agent

        base = AgentBuilderState()
        va = PromptVersion(name="A")
        vb = PromptVersion(name="B")

        results = run_comparison(base, va, vb, [{"text": "great"}])
        assert len(results) == 1
        assert results[0].outputs_match is True
        assert results[0].output_a == {"category": "POSITIVE"}
        assert results[0].output_b == {"category": "POSITIVE"}

    @patch("agents7.ui.comparison.runner.build_agent_from_state")
    def test_different_outputs(self, mock_build):
        resp_a = _make_response("POSITIVE")
        resp_b = _make_response("NEGATIVE")

        agent_a = _make_agent_mock([resp_a])
        agent_b = _make_agent_mock([resp_b])
        mock_build.side_effect = [agent_a, agent_b]

        base = AgentBuilderState()
        va = PromptVersion(name="A")
        vb = PromptVersion(name="B")

        results = run_comparison(base, va, vb, [{"text": "mixed"}])
        assert len(results) == 1
        assert results[0].outputs_match is False
        assert results[0].output_a == {"category": "POSITIVE"}
        assert results[0].output_b == {"category": "NEGATIVE"}

    @patch("agents7.ui.comparison.runner.build_agent_from_state")
    def test_error_in_one_version(self, mock_build):
        resp_a = _make_response("POSITIVE")
        agent_a = _make_agent_mock([resp_a])

        agent_b = MagicMock()
        agent_b.act = MagicMock(side_effect=RuntimeError("connection error"))

        mock_build.side_effect = [agent_a, agent_b]

        base = AgentBuilderState()
        va = PromptVersion(name="A")
        vb = PromptVersion(name="B")

        results = run_comparison(base, va, vb, [{"text": "test"}])
        assert results[0].error_a is None
        assert "connection error" in results[0].error_b
        assert results[0].outputs_match is False

    @patch("agents7.ui.comparison.runner.build_agent_from_state")
    def test_progress_callback(self, mock_build):
        resp = _make_response("POSITIVE")
        agent = _make_agent_mock([resp, resp, resp, resp])
        mock_build.return_value = agent

        callback = MagicMock()
        base = AgentBuilderState()
        va = PromptVersion(name="A")
        vb = PromptVersion(name="B")

        run_comparison(
            base,
            va,
            vb,
            [{"text": "a"}, {"text": "b"}],
            progress_callback=callback,
        )
        assert callback.call_count == 2
        callback.assert_any_call(1, 2)
        callback.assert_any_call(2, 2)

    @patch("agents7.ui.comparison.runner.build_agent_from_state")
    def test_empty_inputs(self, mock_build):
        agent = MagicMock()
        mock_build.return_value = agent

        base = AgentBuilderState()
        va = PromptVersion(name="A")
        vb = PromptVersion(name="B")

        results = run_comparison(base, va, vb, [])
        assert results == []

    @patch("agents7.ui.comparison.runner.build_agent_from_state")
    def test_none_model_output(self, mock_build):
        resp = AgentResponse(
            response="{}",
            model=None,
            metrics=_mock_metrics(),
        )
        agent = _make_agent_mock([resp, resp])
        mock_build.return_value = agent

        base = AgentBuilderState()
        va = PromptVersion(name="A")
        vb = PromptVersion(name="B")

        results = run_comparison(base, va, vb, [{"text": "bad"}])
        assert results[0].output_a is None
        assert results[0].output_b is None
        assert results[0].outputs_match is False
