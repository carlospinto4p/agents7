"""Shared pytest fixtures for unit tests.

Only pytest fixtures belong here. Plain helper functions,
factory functions, constants, and mock models live in
``helpers.py``.
"""

import os
from pathlib import Path

import pytest

from agents7.agents.classifier import Classifier
from agents7.agents.detector import Detector
from agents7.agents.regressor import Regressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True, scope="session")
def _chdir_to_project_root():
    """Run all unit tests from the project root.

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
    """Create a temperature detector from YAML config."""
    return Detector.from_yaml(
        "examples/configs/demo_temperature_detector.yml"
    )


@pytest.fixture
def sentiment_classifier():
    """Create a sentiment classifier from YAML config."""
    return Classifier.from_yaml(
        "examples/configs/demo_sentiment_classifier.yml"
    )


@pytest.fixture
def temperature_regressor():
    """Create a temperature regressor from YAML config."""
    return Regressor.from_yaml(
        "examples/configs/demo_temperature_detector.yml"
    )
