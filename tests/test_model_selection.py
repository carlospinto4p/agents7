"""Unit tests for the model_selection module."""

from enum import StrEnum
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from agents7.model_selection import (
    BenchmarkResult,
    CaseResult,
    ModelScore,
    ModelSelectionError,
    TestCase,
    _benchmark_model,
    _compare_field,
    _compare_output,
    _create_agent,
    _format_size,
    _resolve_candidates,
    find_best_model,
)


# --- Test fixtures ---


class Reason(StrEnum):
    TOO_COLD = "Too cold."
    TOO_HOT = "Too hot."
    COMFORTABLE = "Comfortable."
    NOT_APPLICABLE = "Not applicable."


class MockOutput(BaseModel):
    value: float | None = None
    reason: Reason


# --- _compare_field ---


def test_compare_field_enum_match():
    assert _compare_field(Reason.TOO_COLD, "TOO_COLD") is True


def test_compare_field_enum_mismatch():
    assert _compare_field(Reason.TOO_COLD, "TOO_HOT") is False


def test_compare_field_none_both():
    assert _compare_field(None, None) is True


def test_compare_field_none_actual_only():
    assert _compare_field(None, 42) is False


def test_compare_field_none_expected_only():
    assert _compare_field(42, None) is False


def test_compare_field_numeric_match():
    assert _compare_field(25.0, 25.0) is True


def test_compare_field_numeric_mismatch():
    assert _compare_field(25.0, 30.0) is False


def test_compare_field_string_match():
    assert _compare_field("hello", "hello") is True


# --- _compare_output ---


def test_compare_output_all_match():
    model = MockOutput(value=25.0, reason=Reason.COMFORTABLE)
    passed, actual = _compare_output(
        model, {"reason": "COMFORTABLE", "value": 25.0}
    )
    assert passed is True
    assert actual == {"reason": "COMFORTABLE", "value": 25.0}


def test_compare_output_partial_match():
    model = MockOutput(value=25.0, reason=Reason.COMFORTABLE)
    passed, actual = _compare_output(model, {"reason": "COMFORTABLE"})
    assert passed is True
    assert actual == {"reason": "COMFORTABLE"}


def test_compare_output_mismatch():
    model = MockOutput(value=25.0, reason=Reason.TOO_HOT)
    passed, actual = _compare_output(model, {"reason": "COMFORTABLE"})
    assert passed is False
    assert actual == {"reason": "TOO_HOT"}


def test_compare_output_none_model_empty_expected():
    passed, actual = _compare_output(None, {})
    assert passed is True
    assert actual is None


def test_compare_output_none_model_nonempty_expected():
    passed, actual = _compare_output(None, {"reason": "TOO_COLD"})
    assert passed is False
    assert actual is None


def test_compare_output_missing_field():
    model = MockOutput(value=25.0, reason=Reason.COMFORTABLE)
    passed, actual = _compare_output(model, {"nonexistent": "value"})
    assert passed is False
    assert actual == {"nonexistent": "<missing>"}


def test_compare_output_none_value():
    model = MockOutput(value=None, reason=Reason.NOT_APPLICABLE)
    passed, actual = _compare_output(
        model,
        {"value": None, "reason": "NOT_APPLICABLE"},
    )
    assert passed is True


# --- _format_size ---


def test_format_size_gb():
    assert _format_size(3_500_000_000) == "3.5 GB"


def test_format_size_mb():
    assert _format_size(500_000_000) == "500.0 MB"


def test_format_size_bytes():
    assert _format_size(1_000) == "1,000 B"


# --- _resolve_candidates ---


@patch("agents7.model_selection._list_ollama_models")
def test_resolve_candidates_all(mock_list):
    mock_list.return_value = [
        {"name": "small:latest", "size": 100},
        {"name": "big:latest", "size": 500},
    ]
    result = _resolve_candidates(None)
    assert len(result) == 2
    assert result[0]["name"] == "small:latest"


@patch("agents7.model_selection._list_ollama_models")
def test_resolve_candidates_filtered(mock_list):
    mock_list.return_value = [
        {"name": "small:latest", "size": 100},
        {"name": "big:latest", "size": 500},
        {"name": "medium:latest", "size": 300},
    ]
    result = _resolve_candidates(["big:latest"])
    assert len(result) == 1
    assert result[0]["name"] == "big:latest"


@patch("agents7.model_selection._list_ollama_models")
def test_resolve_candidates_with_latest_suffix(mock_list):
    mock_list.return_value = [
        {"name": "qwen2.5:3b", "size": 100},
    ]
    # Exact match first
    result = _resolve_candidates(["qwen2.5:3b"])
    assert len(result) == 1


@patch("agents7.model_selection._list_ollama_models")
def test_resolve_candidates_missing_model(mock_list):
    mock_list.return_value = [
        {"name": "small:latest", "size": 100},
    ]
    result = _resolve_candidates(["nonexistent"])
    assert len(result) == 0


@patch("agents7.model_selection._list_ollama_models")
def test_resolve_candidates_sorted_by_size(mock_list):
    mock_list.return_value = [
        {"name": "small:latest", "size": 100},
        {"name": "big:latest", "size": 500},
        {"name": "medium:latest", "size": 300},
    ]
    result = _resolve_candidates(["big:latest", "small:latest"])
    assert result[0]["name"] == "small:latest"
    assert result[1]["name"] == "big:latest"


# --- TestCase ---


def test_testcase_creation():
    tc = TestCase(
        input={"text": "hello"},
        expected={"reason": "GREETING"},
    )
    assert tc.input == {"text": "hello"}
    assert tc.expected == {"reason": "GREETING"}


def test_testcase_frozen():
    tc = TestCase(input={}, expected={})
    with pytest.raises(AttributeError):
        tc.input = {"new": "value"}


def test_testcase_empty_fields():
    """TestCase with empty dicts is valid."""
    tc = TestCase(input={}, expected={})
    assert tc.input == {}
    assert tc.expected == {}


def test_testcase_nested_input():
    """TestCase handles nested input dicts."""
    tc = TestCase(
        input={"data": {"key": "value"}, "count": 3},
        expected={"label": "X"},
    )
    assert tc.input["data"]["key"] == "value"


# --- ModelScore ---


def test_modelscore_creation():
    score = ModelScore(
        model_name="test:1b",
        model_size=1_000_000,
        passed=True,
        pass_rate=1.0,
    )
    assert score.passed is True
    assert score.error is None


# --- BenchmarkResult ---


def test_benchmarkresult_no_best():
    result = BenchmarkResult(
        best_model=None,
        best_model_size=None,
        scores=[],
    )
    assert result.best_model is None


def test_benchmarkresult_with_best():
    """BenchmarkResult stores best model, size, and scores."""
    score = ModelScore(
        model_name="small:1b",
        model_size=1_000_000,
        passed=True,
        pass_rate=1.0,
    )
    result = BenchmarkResult(
        best_model="small:1b",
        best_model_size=1_000_000,
        scores=[score],
    )
    assert result.best_model == "small:1b"
    assert result.best_model_size == 1_000_000
    assert len(result.scores) == 1
    assert result.scores[0].passed is True


def test_benchmarkresult_frozen():
    """BenchmarkResult is immutable."""
    result = BenchmarkResult(best_model=None, best_model_size=None, scores=[])
    with pytest.raises(AttributeError):
        result.best_model = "something"


# --- find_best_model ---


def test_find_best_model_no_test_cases():
    with pytest.raises(ModelSelectionError):
        find_best_model("some.yml", test_cases=[])


@patch("agents7.model_selection._resolve_candidates")
@patch("agents7.model_selection.load_yaml_settings")
def test_find_best_model_no_candidates(mock_config, mock_resolve):
    mock_config.return_value = {"engine": MagicMock()}
    mock_resolve.return_value = []
    with pytest.raises(ModelSelectionError):
        find_best_model(
            "some.yml",
            test_cases=[TestCase(input={}, expected={})],
        )


@patch("agents7.model_selection._benchmark_model")
@patch("agents7.model_selection._resolve_candidates")
@patch("agents7.model_selection.load_yaml_settings")
def test_find_best_model_first_passes(
    mock_config, mock_resolve, mock_benchmark
):
    mock_config.return_value = {"engine": MagicMock()}
    mock_resolve.return_value = [
        {"name": "small:1b", "size": 1000},
        {"name": "big:7b", "size": 7000},
    ]
    mock_benchmark.return_value = ModelScore(
        model_name="small:1b",
        model_size=1000,
        passed=True,
        pass_rate=1.0,
    )
    result = find_best_model(
        "some.yml",
        test_cases=[TestCase(input={}, expected={})],
    )
    assert result.best_model == "small:1b"
    assert len(result.scores) == 1  # stopped on first


@patch("agents7.model_selection._benchmark_model")
@patch("agents7.model_selection._resolve_candidates")
@patch("agents7.model_selection.load_yaml_settings")
def test_find_best_model_second_passes(
    mock_config, mock_resolve, mock_benchmark
):
    mock_config.return_value = {"engine": MagicMock()}
    mock_resolve.return_value = [
        {"name": "small:1b", "size": 1000},
        {"name": "big:7b", "size": 7000},
    ]
    mock_benchmark.side_effect = [
        ModelScore(
            model_name="small:1b",
            model_size=1000,
            passed=False,
            pass_rate=0.5,
        ),
        ModelScore(
            model_name="big:7b",
            model_size=7000,
            passed=True,
            pass_rate=1.0,
        ),
    ]
    result = find_best_model(
        "some.yml",
        test_cases=[TestCase(input={}, expected={})],
    )
    assert result.best_model == "big:7b"
    assert len(result.scores) == 2


@patch("agents7.model_selection._benchmark_model")
@patch("agents7.model_selection._resolve_candidates")
@patch("agents7.model_selection.load_yaml_settings")
def test_find_best_model_none_pass(mock_config, mock_resolve, mock_benchmark):
    mock_config.return_value = {"engine": MagicMock()}
    mock_resolve.return_value = [
        {"name": "small:1b", "size": 1000},
    ]
    mock_benchmark.return_value = ModelScore(
        model_name="small:1b",
        model_size=1000,
        passed=False,
        pass_rate=0.5,
    )
    result = find_best_model(
        "some.yml",
        test_cases=[TestCase(input={}, expected={})],
    )
    assert result.best_model is None


@patch("agents7.model_selection._benchmark_model")
@patch("agents7.model_selection._resolve_candidates")
@patch("agents7.model_selection.load_yaml_settings")
def test_find_best_model_stop_on_first_false(
    mock_config, mock_resolve, mock_benchmark
):
    mock_config.return_value = {"engine": MagicMock()}
    mock_resolve.return_value = [
        {"name": "small:1b", "size": 1000},
        {"name": "big:7b", "size": 7000},
    ]
    mock_benchmark.side_effect = [
        ModelScore(
            model_name="small:1b",
            model_size=1000,
            passed=True,
            pass_rate=1.0,
        ),
        ModelScore(
            model_name="big:7b",
            model_size=7000,
            passed=True,
            pass_rate=1.0,
        ),
    ]
    result = find_best_model(
        "some.yml",
        test_cases=[TestCase(input={}, expected={})],
        stop_on_first=False,
    )
    assert result.best_model == "small:1b"
    assert len(result.scores) == 2  # tested all


@patch("agents7.model_selection._benchmark_model")
@patch("agents7.model_selection._resolve_candidates")
@patch("agents7.model_selection.load_yaml_settings")
def test_find_best_model_writes_report(
    mock_config, mock_resolve, mock_benchmark, tmp_path
):
    mock_config.return_value = {"engine": MagicMock()}
    mock_resolve.return_value = [{"name": "small:1b", "size": 1000}]
    mock_benchmark.return_value = ModelScore(
        model_name="small:1b",
        model_size=1000,
        passed=True,
        pass_rate=1.0,
    )
    report = tmp_path / "bench.md"
    find_best_model(
        "some.yml",
        test_cases=[TestCase(input={}, expected={})],
        report_path=str(report),
    )
    assert report.exists()
    assert "Benchmark Report" in report.read_text(encoding="utf-8")


# --- _create_agent ---


def test_create_agent_labeler():
    from agents7.agents import Labeler

    config = {
        "name": "Test",
        "version": 1,
        "engine": MagicMock(),
        "system_prompt": "Test prompt",
        "output_model": MockOutput,
        "examples": None,
        "validators": None,
        "prompt_params": None,
    }
    agent = _create_agent(config)
    assert isinstance(agent, Labeler)


# --- CaseResult ---


def test_case_result_creation():
    tc = TestCase(input={}, expected={})
    cr = CaseResult(test_case=tc, passed=True, actual={}, run=1)
    assert cr.passed is True
    assert cr.run == 1


def test_case_result_failed_with_none_actual():
    """CaseResult with passed=False and actual=None."""
    tc = TestCase(input={}, expected={})
    cr = CaseResult(test_case=tc, passed=False, actual=None, run=1)
    assert cr.passed is False
    assert cr.actual is None


def test_case_result_run_number():
    """CaseResult stores run number correctly for multi-run benchmarks."""
    tc = TestCase(input={}, expected={})
    cr = CaseResult(test_case=tc, passed=True, actual={"x": 1}, run=3)
    assert cr.run == 3


def test_case_result_frozen():
    """CaseResult is immutable."""
    tc = TestCase(input={}, expected={})
    cr = CaseResult(test_case=tc, passed=True, actual={}, run=1)
    with pytest.raises(AttributeError):
        cr.passed = False


# --- _benchmark_model ---


@patch("agents7.model_selection.LLMEngine")
def test_benchmark_model_engine_init_failure(mock_engine_cls):
    mock_engine_cls.side_effect = RuntimeError("connection refused")
    score = _benchmark_model(
        config={},
        model_name="bad:1b",
        model_size=1_000_000,
        test_cases=[TestCase(input={}, expected={})],
        runs=1,
        include_batch=False,
    )
    assert score.passed is False
    assert score.model_name == "bad:1b"
    assert score.error is not None
    assert "Engine init failed" in score.error


def test_benchmark_model_runs_zero_raises():
    """runs=0 raises ValueError — vacuous passes hide bugs."""
    with pytest.raises(ValueError, match="runs must be >= 1"):
        _benchmark_model(
            config={},
            model_name="any:1b",
            model_size=1_000_000,
            test_cases=[TestCase(input={}, expected={})],
            runs=0,
            include_batch=False,
        )


@patch("agents7.model_selection._benchmark_model")
@patch("agents7.model_selection._resolve_candidates")
@patch("agents7.model_selection.load_yaml_settings")
def test_find_best_model_runs_zero_passed_through(
    mock_config, mock_resolve, mock_benchmark
):
    """runs=0 is forwarded to _benchmark_model."""
    mock_config.return_value = {"engine": MagicMock()}
    mock_resolve.return_value = [{"name": "small:1b", "size": 1000}]
    mock_benchmark.return_value = ModelScore(
        model_name="small:1b",
        model_size=1000,
        passed=True,
        pass_rate=0.0,
        case_results=[],
    )
    tc = TestCase(input={}, expected={})
    find_best_model("some.yml", test_cases=[tc], runs=0)
    mock_benchmark.assert_called_once_with(
        mock_config.return_value,
        "small:1b",
        1000,
        [tc],
        runs=0,
        include_batch=False,
    )
