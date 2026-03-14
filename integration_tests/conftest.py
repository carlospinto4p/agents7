"""Shared fixtures and configuration for integration tests."""

import os
import subprocess
from pathlib import Path

import pytest

from agents7.agents import Classifier, Detector, Regressor
from agents7.export import export_eval_report
from agents7.metrics import compute_metrics, run_evaluation

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


def pytest_collection_modifyitems(items):
    """Skip all integration tests when Ollama is unavailable."""
    if _OLLAMA_AVAILABLE:
        return
    skip = pytest.mark.skip(reason="Ollama is not available")
    for item in items:
        item.add_marker(skip)


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


def generate_eval_report(
    agent,
    eval_data: list[dict],
    expected_fields: list[str],
    reports_dir: Path,
    filename: str,
    confusion_field: str | None = None,
) -> None:
    """Run evaluation and write a Markdown report.

    :param agent: Agent instance.
    :param eval_data: Evaluation data rows with ``input``
        and expected field keys.
    :param expected_fields: Fields to compare.
    :param reports_dir: Directory to write the report.
    :param filename: Report filename (e.g.
        ``"detector.md"``).
    :param confusion_field: Optional field for confusion
        matrix.
    """
    results = run_evaluation(agent, eval_data, expected_fields)
    metrics = compute_metrics(results, expected_fields)
    export_eval_report(
        results,
        metrics,
        reports_dir / filename,
        agent_name=agent.name,
        model_id=agent.engine.model_id,
        confusion_field=confusion_field,
    )
