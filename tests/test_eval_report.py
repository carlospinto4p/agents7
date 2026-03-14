"""Unit tests for export_eval_report."""

from agents7.export import export_eval_report
from agents7.metrics import EvalResult


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
