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


@pytest.fixture
def temperature_detector():
    """Create a temperature detector instance."""
    return Detector.from_yaml(
        "examples/configs/demo_temperature_detector.yml"
    )


@pytest.fixture
def sentiment_classifier():
    """Create a sentiment classifier instance."""
    return Classifier.from_yaml(
        "examples/configs/demo_sentiment_classifier.yml"
    )


@pytest.fixture
def temperature_regressor():
    """Create a temperature regressor instance."""
    return Regressor.from_yaml(
        "examples/configs/demo_temperature_detector.yml"
    )
