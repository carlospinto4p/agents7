from enum import StrEnum

import pytest
from pydantic import BaseModel

from agents7.agents.detector import Detector
from agents7.exceptions import PropertyConfigurationError
from agents7.llm import LLMEngine
from agents7.models import Property
from agents7.models.base import NOT_APPLICABLE_DESCRIPTION
from agents7.models.demo import TemperaturePropertyReason


# --- Initialization & prompt ---


def test_detector_initialization(temperature_detector: Detector) -> None:
    """Test that the detector can be properly initialized from YAML config."""
    assert temperature_detector is not None
    assert hasattr(temperature_detector, "prompt")
    assert hasattr(temperature_detector, "act_in_batch")


def test_detector_prompt_exists(temperature_detector: Detector) -> None:
    """Test that the detector has a valid prompt."""
    assert temperature_detector.prompt is not None
    assert len(temperature_detector.prompt.strip()) > 0


# --- Edge cases ---


def test_empty_input(temperature_detector: Detector) -> None:
    """Test behavior with empty input."""
    response = temperature_detector.act_in_batch([])
    assert response == []


# --- Enum properties ---


def test_possible_reasons(temperature_detector: Detector) -> None:
    """Test that the detector can return all possible reasons."""
    expected_reasons = {
        k: v for k, v in TemperaturePropertyReason.__members__.items()
    }
    expected_reasons["NOT_APPLICABLE"] = NOT_APPLICABLE_DESCRIPTION
    assert temperature_detector.possible_reasons == expected_reasons


# --- NOT_APPLICABLE auto-extension ---


def test_not_applicable_added_automatically() -> None:
    """Test that NOT_APPLICABLE is automatically added to reason enums."""

    class SimpleReason(StrEnum):
        GOOD = "Good result."
        BAD = "Bad result."

    class SimpleProperty(Property[str]):
        reason: SimpleReason

    # Verify NOT_APPLICABLE was added to the annotation
    reason_enum = SimpleProperty.__annotations__["reason"]
    assert hasattr(reason_enum, "NOT_APPLICABLE")
    assert (
        reason_enum.NOT_APPLICABLE
        == "No applicable reason for this classification."
    )

    # Verify original members are preserved
    assert hasattr(reason_enum, "GOOD")
    assert hasattr(reason_enum, "BAD")


def test_explicit_not_applicable_preserved() -> None:
    """Test that explicit NOT_APPLICABLE is preserved and not overwritten."""

    class ExplicitReason(StrEnum):
        YES = "Affirmative."
        NO = "Negative."
        NOT_APPLICABLE = "Custom N/A message."

    class ExplicitProperty(Property[bool]):
        reason: ExplicitReason

    reason_enum = ExplicitProperty.__annotations__["reason"]
    assert reason_enum.NOT_APPLICABLE == "Custom N/A message."


# --- Validation ---


def test_detector_requires_property_subclass() -> None:
    """Test that Detector raises error for non-Property models."""

    class NotAProperty(BaseModel):
        value: str

    engine = LLMEngine(provider="ollama", model_name="qwen3")

    with pytest.raises(PropertyConfigurationError):
        Detector(
            name="InvalidDetector",
            version=1,
            engine=engine,
            system_prompt="Test prompt.",
            output_model=NotAProperty,
        )
