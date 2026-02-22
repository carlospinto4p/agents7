"""Unit tests for agents7.ui.comparison (models + runner)."""

from unittest.mock import MagicMock, patch

from agents7.models.base import AgentResponse
from agents7.ui.comparison.models import PromptVersion
from agents7.ui.comparison.runner import (
    ComparisonResult,
    _build_version_state,
    run_comparison,
)
from agents7.ui.state import AgentBuilderState

from conftest import make_agent_mock, make_response, mock_metrics


# --- PromptVersion ---


def test_creation_defaults():
    v = PromptVersion(name="v1")
    assert v.name == "v1"
    assert v.system_prompt == ""
    assert v.examples == []
    assert v.model_name == "qwen3"


def test_creation_all_fields():
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


# --- _build_version_state ---


def test_replaces_prompt_fields():
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


def test_preserves_non_prompt_fields():
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


# --- ComparisonResult ---


def test_comparison_result_defaults():
    r = ComparisonResult(input_data={"text": "hi"})
    assert r.input_data == {"text": "hi"}
    assert r.output_a is None
    assert r.output_b is None
    assert r.metrics_a == {}
    assert r.metrics_b == {}
    assert r.error_a is None
    assert r.error_b is None
    assert r.outputs_match is False


def test_outputs_match_true():
    r = ComparisonResult(
        input_data={"text": "hi"},
        output_a={"category": "POS"},
        output_b={"category": "POS"},
        outputs_match=True,
    )
    assert r.outputs_match is True


# --- run_comparison ---


@patch("agents7.ui.comparison.runner.build_agent_from_state")
def test_matching_outputs(mock_build):
    resp = make_response("POSITIVE")
    agent = make_agent_mock([resp, resp])
    mock_build.return_value = agent

    base = AgentBuilderState()
    va = PromptVersion(name="A")
    vb = PromptVersion(name="B")

    results = run_comparison(base, va, vb, [{"text": "great"}])
    assert len(results) == 1
    assert results[0].outputs_match is True
    assert results[0].output_a == {
        "category": "POSITIVE",
        "reason": "HAPPY",
    }
    assert results[0].output_b == {
        "category": "POSITIVE",
        "reason": "HAPPY",
    }


@patch("agents7.ui.comparison.runner.build_agent_from_state")
def test_different_outputs(mock_build):
    resp_a = make_response("POSITIVE")
    resp_b = make_response("NEGATIVE")

    agent_a = make_agent_mock([resp_a])
    agent_b = make_agent_mock([resp_b])
    mock_build.side_effect = [agent_a, agent_b]

    base = AgentBuilderState()
    va = PromptVersion(name="A")
    vb = PromptVersion(name="B")

    results = run_comparison(base, va, vb, [{"text": "mixed"}])
    assert len(results) == 1
    assert results[0].outputs_match is False
    assert results[0].output_a == {
        "category": "POSITIVE",
        "reason": "HAPPY",
    }
    assert results[0].output_b == {
        "category": "NEGATIVE",
        "reason": "HAPPY",
    }


@patch("agents7.ui.comparison.runner.build_agent_from_state")
def test_error_in_one_version(mock_build):
    resp_a = make_response("POSITIVE")
    agent_a = make_agent_mock([resp_a])

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
def test_progress_callback(mock_build):
    resp = make_response("POSITIVE")
    agent = make_agent_mock([resp, resp, resp, resp])
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
def test_empty_inputs(mock_build):
    agent = MagicMock()
    mock_build.return_value = agent

    base = AgentBuilderState()
    va = PromptVersion(name="A")
    vb = PromptVersion(name="B")

    results = run_comparison(base, va, vb, [])
    assert results == []


@patch("agents7.ui.comparison.runner.build_agent_from_state")
def test_none_model_output(mock_build):
    resp = AgentResponse(
        response="{}",
        model=None,
        metrics=mock_metrics(),
    )
    agent = make_agent_mock([resp, resp])
    mock_build.return_value = agent

    base = AgentBuilderState()
    va = PromptVersion(name="A")
    vb = PromptVersion(name="B")

    results = run_comparison(base, va, vb, [{"text": "bad"}])
    assert results[0].output_a is None
    assert results[0].output_b is None
    assert results[0].outputs_match is False
