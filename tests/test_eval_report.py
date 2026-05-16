"""Unit tests for export_eval_report."""

from agents7.eval.export import export_eval_report
from agents7.eval.metrics import EvalResult


def _make_results():
    return [
        EvalResult(
            input_data={"text": "great"},
            expected={"cat": "A"},
            actual={"cat": "A"},
            passed=True,
            field_results={"cat": True},
        ),
        EvalResult(
            input_data={"text": "bad"},
            expected={"cat": "B"},
            actual={"cat": "A"},
            passed=False,
            field_results={"cat": False},
        ),
    ]


def _make_metrics():
    return {
        "total": 2,
        "passed": 1,
        "failed": 1,
        "accuracy": 0.5,
        "per_field": {"cat": 0.5},
    }


# --- Basic report ---


def test_basic_report(tmp_path):
    """Generates a Markdown report with all sections."""
    path = tmp_path / "report.md"
    export_eval_report(_make_results(), _make_metrics(), path)

    text = path.read_text()
    assert "# Evaluation Report" in text
    assert "50.0%" in text
    assert "Case 1" in text
    assert "PASS" in text
    assert "FAIL" in text


# --- Agent and model info ---


def test_report_with_agent_info(tmp_path):
    """Report includes agent name and model ID."""
    path = tmp_path / "report.md"
    export_eval_report(
        _make_results(),
        _make_metrics(),
        path,
        agent_name="MyAgent",
        model_id="ollama::qwen3",
    )

    text = path.read_text()
    assert "MyAgent" in text
    assert "ollama::qwen3" in text


# --- Per-field accuracy ---


def test_report_per_field_accuracy(tmp_path):
    """Report includes per-field accuracy table."""
    path = tmp_path / "report.md"
    export_eval_report(_make_results(), _make_metrics(), path)

    text = path.read_text()
    assert "Per-Field Accuracy" in text
    assert "| cat |" in text


# --- Confusion matrix ---


def test_report_with_confusion_matrix(tmp_path):
    """Report includes confusion matrix when requested."""
    path = tmp_path / "report.md"
    export_eval_report(
        _make_results(),
        _make_metrics(),
        path,
        confusion_field="cat",
    )

    text = path.read_text()
    assert "Confusion Matrix (cat)" in text
    assert "Expected \\\\ Actual" in text


# --- No confusion matrix by default ---


def test_report_no_confusion_matrix(tmp_path):
    """Report omits confusion matrix when not requested."""
    path = tmp_path / "report.md"
    export_eval_report(_make_results(), _make_metrics(), path)

    text = path.read_text()
    assert "Confusion Matrix" not in text


# --- Error in results ---


def test_report_with_error(tmp_path):
    """Report includes error messages."""
    results = [
        EvalResult(
            input_data={"text": "x"},
            expected={"cat": "A"},
            passed=False,
            error="connection failed",
        ),
    ]
    metrics = {
        "total": 1,
        "passed": 0,
        "failed": 1,
        "accuracy": 0.0,
        "per_field": {},
    }
    path = tmp_path / "report.md"
    export_eval_report(results, metrics, path)

    text = path.read_text()
    assert "connection failed" in text


# --- Empty results ---


def test_report_empty_results(tmp_path):
    """Report handles empty results."""
    metrics = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "accuracy": 0.0,
        "per_field": {},
    }
    path = tmp_path / "report.md"
    export_eval_report([], metrics, path)

    text = path.read_text()
    assert "Total | 0" in text


# --- Date included ---


def test_report_includes_date(tmp_path):
    """Report includes a date stamp."""
    path = tmp_path / "report.md"
    export_eval_report(_make_results(), _make_metrics(), path)

    text = path.read_text()
    assert "Date" in text
    assert "UTC" in text


# --- Timing section ---


def _timing_metrics(**overrides):
    timing = {
        "total_s": 3.5,
        "avg_s": 1.75,
        "min_s": 1.2,
        "max_s": 2.3,
        "per_case": [],
    }
    timing.update(overrides)
    return {**_make_metrics(), "timing": timing}


def test_report_timing_section_rendered(tmp_path):
    """Report includes Response Times section when timing data is present."""
    path = tmp_path / "report.md"
    export_eval_report(_make_results(), _timing_metrics(), path)

    text = path.read_text()
    assert "## Response Times" in text
    assert "3.50s" in text
    assert "1.75s" in text
    assert "1.20s" in text
    assert "2.30s" in text


def test_report_per_case_timing(tmp_path):
    """Per-case timing is appended to each case header."""
    metrics = _timing_metrics(per_case=[1.0, 2.0])
    path = tmp_path / "report.md"
    export_eval_report(_make_results(), metrics, path)

    text = path.read_text()
    assert "PASS (1.00s)" in text
    assert "FAIL (2.00s)" in text


def test_report_timing_omitted_when_absent(tmp_path):
    """Timing section omitted when timing key is absent from metrics."""
    path = tmp_path / "report.md"
    export_eval_report(_make_results(), _make_metrics(), path)

    text = path.read_text()
    assert "## Response Times" not in text


def test_report_timing_omitted_when_zero(tmp_path):
    """Timing section omitted when total_s is zero."""
    metrics = _timing_metrics(total_s=0, avg_s=0, min_s=0, max_s=0)
    path = tmp_path / "report.md"
    export_eval_report(_make_results(), metrics, path)

    text = path.read_text()
    assert "## Response Times" not in text


# --- Missing metric fields ---


def test_report_missing_metric_fields(tmp_path):
    """Report uses zero defaults when metrics keys are absent."""
    path = tmp_path / "report.md"
    export_eval_report([], {}, path)

    text = path.read_text()
    assert "| Total | 0 |" in text
    assert "0.0%" in text
