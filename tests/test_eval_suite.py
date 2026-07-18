"""Unit tests for the composed eval route (agents7.eval.suite)."""

import json

import pytest

from agents7.agents.classifier import Classifier
from agents7.eval.suite import EvalRun, resolve_eval_fields, run_eval_suite
from agents7.models.demo import (
    Sentiment,
    SentimentCategory,
    SentimentReason,
)
from agents7.testing import create_mock_engine

_CONFIG = "examples/configs/demo_sentiment_classifier.yml"


@pytest.fixture
def mock_agent():
    """A Classifier backed by a mock engine, always POSITIVE."""
    engine = create_mock_engine(
        json.dumps(
            {
                "category": SentimentCategory.POSITIVE.value,
                "reason": SentimentReason.ENTHUSIASTIC.value,
            }
        )
    )
    return Classifier(
        name="MockSentiment",
        version=1,
        system_prompt="Classify sentiment.",
        output_model=Sentiment,
        engine=engine,
    )


@pytest.fixture
def golden_set(tmp_path):
    """A two-row flat golden set: one match, one mismatch."""
    path = tmp_path / "golden_2.jsonl"
    path.write_text(
        json.dumps(
            {
                "text": "I love it",
                "category": "POSITIVE",
                "reason": "ENTHUSIASTIC",
            }
        )
        + "\n"
        + json.dumps(
            {
                "text": "I hate it",
                "category": "NEGATIVE",
                "reason": "ANGRY",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def patched_build(monkeypatch, mock_agent):
    """Make `run_eval_suite` build the mock agent, not a real one."""
    monkeypatch.setattr(
        "agents7.eval.suite.build_agent",
        lambda config, agent_type=None: mock_agent,
    )
    return mock_agent


# ── resolve_eval_fields ────────────────────────────────────────


def test_resolve_eval_fields_derives_from_output_model():
    """Fields come from the config's output_model when not given."""
    rows = [{"text": "a", "category": "POSITIVE", "note": "skip"}]
    eval_rows, fields = resolve_eval_fields(_CONFIG, rows)
    assert fields == ["category"]
    assert eval_rows == [{"input": "a", "category": "POSITIVE"}]


def test_resolve_eval_fields_explicit_overrides_derived():
    """An explicit field list wins over the derived one."""
    rows = [{"text": "a", "category": "POSITIVE", "reason": "ANGRY"}]
    _, fields = resolve_eval_fields(_CONFIG, rows, ["reason"])
    assert fields == ["reason"]


def test_resolve_eval_fields_empty_raises():
    """No determinable fields is an error, not a silent pass."""
    with pytest.raises(ValueError, match="expected output fields"):
        resolve_eval_fields(_CONFIG, [{"input": "a"}])


# ── run_eval_suite ─────────────────────────────────────────────


def test_run_eval_suite_scores_the_golden_set(patched_build, golden_set):
    """The route loads, runs, and scores in one call."""
    run = run_eval_suite(_CONFIG, golden_set)

    assert isinstance(run, EvalRun)
    assert run.agent_name == "MockSentiment"
    assert run.data_path == golden_set
    assert run.fields == ["category", "reason"]
    assert len(run.results) == 2
    # The mock always answers POSITIVE/ENTHUSIASTIC, so row 1
    # matches and row 2 does not.
    assert run.metrics["total"] == 2
    assert run.metrics["passed"] == 1
    assert run.metrics["accuracy"] == 0.5


def test_run_eval_suite_explicit_fields(patched_build, golden_set):
    """An explicit field list restricts what is scored."""
    run = run_eval_suite(_CONFIG, golden_set, fields=["category"])
    assert run.fields == ["category"]
    assert run.metrics["per_field"] == {"category": 0.5}


def test_run_eval_suite_empty_data_raises(patched_build, tmp_path):
    """An empty golden set is rejected up front."""
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Golden set is empty"):
        run_eval_suite(_CONFIG, path)


def test_run_eval_suite_writes_report(patched_build, golden_set, tmp_path):
    """A requested report is written and its path returned."""
    report = tmp_path / "report.md"
    run = run_eval_suite(_CONFIG, golden_set, report=report)

    assert run.report_path == report
    text = report.read_text(encoding="utf-8")
    assert "# Evaluation Report" in text
    assert "MockSentiment" in text
    assert "Confusion Matrix (category)" in text
    assert golden_set.name in text


def test_run_eval_suite_extra_notes_reach_the_report(
    patched_build, golden_set, tmp_path
):
    """Caller notes are appended after the standard ones."""
    report = tmp_path / "report.md"
    run_eval_suite(
        _CONFIG,
        golden_set,
        report=report,
        notes=["NOT_APPLICABLE is disabled on this model."],
    )
    text = report.read_text(encoding="utf-8")
    assert "NOT_APPLICABLE is disabled on this model." in text
    # The standard notes survive alongside the caller's.
    assert golden_set.name in text
    assert "Accuracy: 50%" in text


def test_run_eval_suite_no_report_leaves_path_none(patched_build, golden_set):
    """Without a report request, no file and no path."""
    run = run_eval_suite(_CONFIG, golden_set)
    assert run.report_path is None


def test_run_eval_suite_call_log_is_summarized(
    patched_build, golden_set, tmp_path
):
    """A requested call log is written and summarised."""
    log = tmp_path / "calls.jsonl"
    run = run_eval_suite(_CONFIG, golden_set, call_log=log)

    assert log.exists()
    assert run.call_log is not None
    assert run.call_log.overall.calls > 0
    assert run.call_log.overall.successes > 0


def test_run_eval_suite_call_log_reaches_the_report(
    patched_build, golden_set, tmp_path
):
    """The call-log summary lands in the Markdown report."""
    report = tmp_path / "report.md"
    run_eval_suite(
        _CONFIG,
        golden_set,
        report=report,
        call_log=tmp_path / "calls.jsonl",
    )
    text = report.read_text(encoding="utf-8")
    assert "## Call Log" in text
    assert "| Completion tokens |" in text


def test_run_eval_suite_without_call_log_has_no_summary(
    patched_build, golden_set
):
    """No call log requested means no summary attached."""
    run = run_eval_suite(_CONFIG, golden_set)
    assert run.call_log is None


def test_run_eval_suite_does_not_mutate_the_built_agent(
    patched_build, golden_set, tmp_path
):
    """Attaching a call logger clones rather than mutating."""
    run_eval_suite(_CONFIG, golden_set, call_log=tmp_path / "calls.jsonl")
    assert patched_build.call_logger is None


def test_run_eval_suite_progress_callback(patched_build, golden_set):
    """The progress callback is forwarded to run_evaluation."""
    seen: list[tuple[int, int]] = []
    run_eval_suite(
        _CONFIG,
        golden_set,
        progress_callback=lambda c, t: seen.append((c, t)),
    )
    assert seen == [(2, 2)]


# ── EvalRun.format ─────────────────────────────────────────────


def test_eval_run_format_includes_headline_numbers(patched_build, golden_set):
    """The summary block carries agent, data, and accuracy."""
    text = run_eval_suite(_CONFIG, golden_set).format()
    assert "MockSentiment" in text
    assert golden_set.name in text
    assert "50.0%" in text
    assert "Per-field accuracy:" in text


def test_eval_run_format_mentions_report_path(
    patched_build, golden_set, tmp_path
):
    """A written report is announced in the summary."""
    report = tmp_path / "report.md"
    text = run_eval_suite(_CONFIG, golden_set, report=report).format()
    assert f"Report written to {report}" in text
