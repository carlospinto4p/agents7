"""Unit tests for agents7.eval.export (JSONL and CSV export)."""

import json

from pydantic import BaseModel

from agents7.eval.export import (
    export_csv,
    export_jsonl,
)
from agents7.models.base import AgentResponse, LLMMetrics


# --- Fixtures ---


class DummyModel(BaseModel):
    name: str
    score: int


def _make_response(name: str, score: int) -> AgentResponse:
    return AgentResponse(
        response="raw text",
        model=DummyModel(name=name, score=score),
        metrics=LLMMetrics(tokens=10, prompt_tokens=5, total_duration=1.0),
    )


def _make_none_response() -> AgentResponse:
    return AgentResponse(
        response="bad json",
        model=None,
        metrics=LLMMetrics(tokens=5, prompt_tokens=3, total_duration=0.5),
    )


# --- JSONL export ---


def test_export_jsonl_basic(tmp_path):
    """Exports responses to valid JSONL."""
    path = tmp_path / "out.jsonl"
    responses = [_make_response("Alice", 90)]
    export_jsonl(responses, path)

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["model"]["name"] == "Alice"
    assert obj["model"]["score"] == 90
    assert obj["metrics"]["tokens"] == 10


def test_export_jsonl_none_model(tmp_path):
    """None model is serialized as null."""
    path = tmp_path / "out.jsonl"
    export_jsonl([_make_none_response()], path)

    obj = json.loads(path.read_text().strip())
    assert obj["model"] is None


def test_export_jsonl_multiple(tmp_path):
    """Multiple responses produce one line each."""
    path = tmp_path / "out.jsonl"
    responses = [
        _make_response("Alice", 90),
        _make_response("Bob", 80),
    ]
    export_jsonl(responses, path)

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2


# --- CSV export ---


def test_export_csv_basic(tmp_path):
    """Exports responses to CSV with detected fields."""
    path = tmp_path / "out.csv"
    responses = [_make_response("Alice", 90)]
    export_csv(responses, path)

    text = path.read_text()
    lines = text.strip().split("\n")
    assert len(lines) == 2
    assert "response" in lines[0]
    assert "name" in lines[0]
    assert "score" in lines[0]
    assert "Alice" in lines[1]


def test_export_csv_explicit_fields(tmp_path):
    """Only specified fields are included."""
    path = tmp_path / "out.csv"
    responses = [_make_response("Alice", 90)]
    export_csv(responses, path, fields=["name"])

    header = path.read_text().strip().split("\n")[0]
    assert "name" in header
    assert "score" not in header


def test_export_csv_none_model(tmp_path):
    """None model produces empty field columns."""
    path = tmp_path / "out.csv"
    responses = [_make_none_response()]
    export_csv(responses, path, fields=["name", "score"])

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    # Data row should have empty values for name and score
    assert lines[1].startswith("bad json,,")


def test_export_csv_empty_responses(tmp_path):
    """Empty list produces header-only CSV."""
    path = tmp_path / "out.csv"
    export_csv([], path, fields=["name"])

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1
    assert "response" in lines[0]


# --- Benchmark report export ---


def _make_benchmark_result():
    from agents7.model_selection import (
        BenchmarkResult,
        CaseResult,
        ModelScore,
        TestCase,
    )

    tc = TestCase(input={"text": "hi"}, expected={"label": "A"})
    cr_pass = CaseResult(
        test_case=tc, passed=True, actual={"label": "A"}, run=1
    )
    cr_fail = CaseResult(
        test_case=tc, passed=False, actual={"label": "B"}, run=1
    )
    scores = [
        ModelScore(
            model_name="small:1b",
            model_size=1_000_000_000,
            passed=False,
            pass_rate=0.5,
            case_results=[cr_pass, cr_fail],
            avg_duration=0.5,
        ),
        ModelScore(
            model_name="big:4b",
            model_size=4_000_000_000,
            passed=True,
            pass_rate=1.0,
            case_results=[cr_pass, cr_pass],
            avg_duration=1.2,
        ),
    ]
    return BenchmarkResult(
        best_model="big:4b",
        best_model_size=4_000_000_000,
        scores=scores,
    )


def test_export_benchmark_report_basic(tmp_path):
    """Writes a markdown report with scores and per-case tables."""
    from agents7.eval.export import export_benchmark_report

    path = tmp_path / "bench.md"
    export_benchmark_report(
        _make_benchmark_result(), path, yaml_path="cfg.yml"
    )

    text = path.read_text(encoding="utf-8")
    assert "# Benchmark Report" in text
    assert "cfg.yml" in text
    assert "big:4b" in text
    assert "small:1b" in text
    assert "Per-case Results" in text


def test_export_benchmark_report_no_best(tmp_path):
    """Handles the case where no model passed."""
    from agents7.eval.export import export_benchmark_report
    from agents7.model_selection import BenchmarkResult

    path = tmp_path / "bench.md"
    export_benchmark_report(
        BenchmarkResult(best_model=None, best_model_size=None, scores=[]),
        path,
    )

    text = path.read_text(encoding="utf-8")
    assert "none passed" in text
