"""Shared pytest fixtures and helpers for unit tests."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from agents7.agents.classifier import Classifier
from agents7.agents.detector import Detector
from agents7.agents.regressor import Regressor
from agents7.models.base import AgentResponse, LLMMetrics

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


# --- Shared UI test helpers ---


def mock_metrics() -> LLMMetrics:
    """Create standard LLMMetrics for UI tests."""
    return LLMMetrics(tokens=10, prompt_tokens=5, total_duration=0.1)


def make_agent_mock(
    responses: list[AgentResponse],
) -> MagicMock:
    """Return a mock agent that returns responses in order."""
    agent = MagicMock()
    agent.act = MagicMock(side_effect=responses)
    return agent


class MockModel(BaseModel):
    """Shared mock model for UI evaluation/comparison tests."""

    category: str = "POSITIVE"
    reason: str = "HAPPY"


def make_response(
    category: str = "POSITIVE",
    reason: str = "HAPPY",
) -> AgentResponse:
    """Build a mock AgentResponse for UI tests."""
    model = MockModel(category=category, reason=reason)
    return AgentResponse(
        response="{}",
        model=model,
        metrics=mock_metrics(),
    )
