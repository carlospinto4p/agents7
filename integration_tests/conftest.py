"""Shared fixtures and configuration for integration tests."""

import os
import subprocess
from pathlib import Path

import pytest

from agents7.agents import Classifier, Detector, Regressor

# Project root is one level up from integration_tests/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _check_ollama() -> bool:
    try:
        return (
            subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=10,
            ).returncode
            == 0
        )
    except Exception:
        return False


_OLLAMA_AVAILABLE = _check_ollama()


_PER_TEST_TIMEOUT = 60  # seconds
_SESSION_TIMEOUT = 300  # 5 minutes total


def pytest_collection_modifyitems(items):
    """Skip all integration tests when Ollama is unavailable.
    Add a 60-second timeout to every test."""
    if not _OLLAMA_AVAILABLE:
        skip = pytest.mark.skip(reason="Ollama is not available")
        for item in items:
            item.add_marker(skip)
        return
    timeout_marker = pytest.mark.timeout(_PER_TEST_TIMEOUT)
    for item in items:
        item.add_marker(timeout_marker)


def pytest_sessionstart(session):
    """Record session start time for the global timeout."""
    session._start_time = __import__("time").monotonic()


def pytest_runtest_setup(item):
    """Fail fast if the session has exceeded 5 minutes."""
    import time

    elapsed = time.monotonic() - item.session._start_time
    if elapsed > _SESSION_TIMEOUT:
        pytest.fail(
            f"Session timeout: {elapsed:.0f}s exceeded "
            f"{_SESSION_TIMEOUT}s limit."
        )


@pytest.fixture(autouse=True, scope="session")
def _chdir_to_project_root():
    """Run all integration tests from the project root.

    YAML config paths like 'examples/configs/...' are relative to the
    project root, so we need to be in that directory.
    """
    original = os.getcwd()
    os.chdir(PROJECT_ROOT)
    yield
    os.chdir(original)


# --- Shared agent fixtures ---


@pytest.fixture(scope="module")
def temperature_detector():
    """Create a temperature detector instance (shared per module)."""
    return Detector.from_yaml(
        "examples/configs/demo_temperature_detector.yml"
    )


@pytest.fixture(scope="module")
def sentiment_classifier():
    """Create a sentiment classifier instance (shared per module)."""
    return Classifier.from_yaml(
        "examples/configs/demo_sentiment_classifier.yml"
    )


@pytest.fixture(scope="module")
def temperature_regressor():
    """Create a temperature regressor instance (shared per module)."""
    return Regressor.from_yaml(
        "examples/configs/demo_temperature_detector.yml"
    )


@pytest.fixture(scope="session")
def reports_dir():
    """Create and return the integration test reports dir."""
    path = PROJECT_ROOT / "integration_tests" / "reports"
    path.mkdir(exist_ok=True)
    return path


def write_eval_report_from_responses(
    agent,
    responses,
    test_data: list[dict],
    expected_field: str,
    expected_key: str,
    reports_dir: Path,
    filename: str,
    confusion_field: str | None = None,
) -> None:
    """Build an eval report from already-collected responses.

    Reuses LLM responses from batch tests — no extra LLM
    calls.

    :param agent: Agent instance (for name/model metadata).
    :param responses: List of AgentResponse from act_in_batch.
    :param test_data: Original test data list.
    :param expected_field: Model field to compare (e.g.
        ``"reason"``).
    :param expected_key: Key in test_data holding expected
        value (e.g. ``"expected_reason"``).
    :param reports_dir: Directory for the report file.
    :param filename: Report filename.
    :param confusion_field: Optional field for confusion
        matrix.
    """
    from agents7.export import export_eval_report
    from agents7.metrics import (
        EvalResult,
        _compare_field,
        _extract_fields,
        compute_metrics,
    )

    results: list[EvalResult] = []
    for case, resp in zip(test_data, responses):
        expected = {expected_field: case[expected_key]}
        result = EvalResult(
            input_data={"text": case["text"]},
            expected=expected,
        )
        result.response = resp
        if resp.model is not None:
            actual = _extract_fields(resp.model, [expected_field])
            result.actual = actual
            result.field_results = {
                expected_field: _compare_field(
                    actual.get(expected_field),
                    expected.get(expected_field),
                )
            }
            result.passed = all(result.field_results.values())
        else:
            result.error = "Model parsing failed"
        results.append(result)

    metrics = compute_metrics(results, [expected_field])
    export_eval_report(
        results,
        metrics,
        reports_dir / filename,
        agent_name=agent.name,
        model_id=agent.engine.model_id,
        confusion_field=confusion_field,
    )
