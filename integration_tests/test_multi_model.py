"""
Tests for multi-model support.

These tests verify that the library works correctly with different
Ollama models, including those that don't support structured output.
"""

import subprocess

import pytest

from agents7.agents import Detector
from agents7.llm import LLMEngine
from agents7.models.demo import Temperature

try:
    _OLLAMA_AVAILABLE = (
        subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=10,
        ).returncode
        == 0
    )
except Exception:
    _OLLAMA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _OLLAMA_AVAILABLE, reason="Ollama is not available"
)


def is_model_available(model_name: str) -> bool:
    """Check if an Ollama model is available locally."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return model_name in result.stdout
    except Exception:
        return False


# Models to test - add more as needed
MODELS_TO_TEST = [
    pytest.param(
        "qwen2.5:3b",
        id="qwen2.5-3b",
        marks=pytest.mark.skipif(
            not is_model_available("qwen2.5:3b"),
            reason="qwen2.5:3b model not available",
        ),
    ),
    pytest.param(
        "gemma3:4b",
        id="gemma3-4b",
        marks=pytest.mark.skipif(
            not is_model_available("gemma3:4b"),
            reason="gemma3:4b model not available",
        ),
    ),
]

# Simple test cases that should work across all models
SIMPLE_TEST_CASES = [
    {
        "text": "It is -10 degrees outside.",
        "expected_reason": "TOO_COLD",
    },
    {
        "text": "The temperature is 35 degrees in the shade.",
        "expected_reason": "TOO_HOT",
    },
    {
        "text": "She bought a new book.",
        "expected_reason": "NOT_APPLICABLE",
    },
]


@pytest.fixture
def make_detector():
    """Factory fixture to create a detector with a specific model."""

    def _make_detector(model_name: str) -> Detector:
        engine = LLMEngine(provider="ollama", model_name=model_name)
        return Detector(
            name="TestDetector",
            version=1,
            engine=engine,
            system_prompt=(
                "Detect temperature from text. Return "
                "temperature in Celsius.\n\n"
                "TOO_COLD: <=10 C. TOO_HOT: >=30 C. "
                "COMFORTABLE: 11-29 C. "
                "NOT_APPLICABLE: no temperature context."
            ),
            output_model=Temperature,
            examples=[
                {
                    "input": "It is freezing outside.",
                    "output": {"value": -5, "reason": "TOO_COLD"},
                },
                {
                    "input": "Nice weather for a walk.",
                    "output": {"value": 22, "reason": "COMFORTABLE"},
                },
                {
                    "input": "The asphalt is melting.",
                    "output": {"value": 40, "reason": "TOO_HOT"},
                },
                {
                    "input": "She bought a new book.",
                    "output": {"value": None, "reason": "NOT_APPLICABLE"},
                },
                {
                    "input": "The meeting is at 3 PM.",
                    "output": {"value": None, "reason": "NOT_APPLICABLE"},
                },
            ],
        )

    return _make_detector


@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
@pytest.mark.parametrize("test_case", SIMPLE_TEST_CASES)
def test_temperature_detection_across_models(
    make_detector, model_name: str, test_case: dict
) -> None:
    """Test temperature detection works across different models."""
    detector = make_detector(model_name)
    input_data = {"text": test_case["text"]}

    response = detector.act(input_data)

    assert response.model is not None, (
        f"Model {model_name} returned None for: {test_case['text']}"
    )
    assert response.model.reason.name == test_case["expected_reason"], (
        f"Model {model_name} returned wrong reason. "
        f"Expected {test_case['expected_reason']}, "
        f"got {response.model.reason.name}"
    )


# Models that support batch processing reliably
MODELS_FOR_BATCH = [
    pytest.param(
        "qwen2.5:3b",
        id="qwen2.5-3b",
        marks=pytest.mark.skipif(
            not is_model_available("qwen2.5:3b"),
            reason="qwen2.5:3b model not available",
        ),
    ),
]


@pytest.mark.parametrize("model_name", MODELS_FOR_BATCH)
def test_batch_processing_across_models(
    make_detector, model_name: str
) -> None:
    """Test that batch processing works across different models."""
    detector = make_detector(model_name)
    inputs = [{"text": case["text"]} for case in SIMPLE_TEST_CASES]

    responses = detector.act_in_batch(inputs)

    assert len(responses) == len(SIMPLE_TEST_CASES)
    for i, (response, test_case) in enumerate(
        zip(responses, SIMPLE_TEST_CASES)
    ):
        assert response.model is not None, (
            f"Model {model_name} returned None for case {i}"
        )
        assert response.model.reason.name == test_case["expected_reason"], (
            f"Model {model_name} batch case {i}: "
            f"expected {test_case['expected_reason']}, "
            f"got {response.model.reason.name}"
        )


@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_response_structure_across_models(
    make_detector, model_name: str
) -> None:
    """Test that response structure is consistent across models."""
    detector = make_detector(model_name)
    input_data = {"text": "The thermometer reads 25 degrees."}

    response = detector.act(input_data)

    # Verify response structure
    assert hasattr(response, "model")
    assert hasattr(response, "response")
    assert hasattr(response, "metrics")

    # Verify model has expected fields
    assert hasattr(response.model, "value")
    assert hasattr(response.model, "reason")

    # Verify metrics exist
    assert response.metrics.tokens > 0
    assert response.metrics.prompt_tokens > 0
