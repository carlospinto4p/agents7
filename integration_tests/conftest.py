"""Shared fixtures and configuration for integration tests."""

import os
import subprocess
from pathlib import Path

import pytest
from integration_helpers import warm_model, warmup_seconds

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
    except (OSError, subprocess.TimeoutExpired):
        return False


_OLLAMA_AVAILABLE = _check_ollama()


_PER_TEST_TIMEOUT = 60  # seconds
_SESSION_TIMEOUT = 300  # 5 minutes total


def pytest_collection_modifyitems(items):
    """Skip all integration tests when Ollama is unavailable.
    Add a 60-second timeout to every test body. Fixture setup is
    left out of it, so loading a model there does not count."""
    if not _OLLAMA_AVAILABLE:
        skip = pytest.mark.skip(reason="Ollama is not available")
        for item in items:
            item.add_marker(skip)
        return
    timeout_marker = pytest.mark.timeout(_PER_TEST_TIMEOUT, func_only=True)
    for item in items:
        item.add_marker(timeout_marker)


def pytest_sessionstart(session):
    """Record session start time for the global timeout."""
    session._start_time = __import__("time").monotonic()


def pytest_runtest_setup(item):
    """Fail fast if the session has exceeded 5 minutes, not
    counting the time spent loading models."""
    import time

    elapsed = time.monotonic() - item.session._start_time - warmup_seconds()
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
    agent = Detector.from_yaml(
        "examples/configs/demo_temperature_detector.yml"
    )
    warm_model(agent.engine.model_name)
    return agent


@pytest.fixture(scope="module")
def sentiment_classifier():
    """Create a sentiment classifier instance (shared per module)."""
    agent = Classifier.from_yaml(
        "examples/configs/demo_sentiment_classifier.yml"
    )
    warm_model(agent.engine.model_name)
    return agent


@pytest.fixture(scope="module")
def temperature_regressor():
    """Create a temperature regressor instance (shared per module)."""
    agent = Regressor.from_yaml(
        "examples/configs/demo_temperature_detector.yml"
    )
    warm_model(agent.engine.model_name)
    return agent


@pytest.fixture(scope="session")
def reports_dir():
    """Create and return the integration test reports dir."""
    path = PROJECT_ROOT / "integration_tests" / "reports"
    path.mkdir(exist_ok=True)
    return path
