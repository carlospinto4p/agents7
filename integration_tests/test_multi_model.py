"""Multi-model integration tests.

Verify that every Ollama model referenced in
``examples/configs/*.yml`` works with the library.
Each model gets a single LLM call to confirm
connectivity, structured output parsing, and response
structure.

Add new models here when a new config references one.
"""

import subprocess

import pytest

from agents7.agents import Detector
from agents7.llm import LLMEngine
from agents7.models.demo import Temperature


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


# Every model referenced in examples/configs/*.yml.
MODELS_TO_TEST = [
    pytest.param(
        "qwen2.5:3b",
        id="qwen2.5-3b",
        marks=pytest.mark.skipif(
            not is_model_available("qwen2.5:3b"),
            reason="qwen2.5:3b not available",
        ),
    ),
    pytest.param(
        "qwen3:1.7b",
        id="qwen3-1.7b",
        marks=pytest.mark.skipif(
            not is_model_available("qwen3:1.7b"),
            reason="qwen3:1.7b not available",
        ),
    ),
    pytest.param(
        "qwen3:4b",
        id="qwen3-4b",
        marks=pytest.mark.skipif(
            not is_model_available("qwen3:4b"),
            reason="qwen3:4b not available",
        ),
    ),
    pytest.param(
        "gemma3:12b",
        id="gemma3-12b",
        marks=pytest.mark.skipif(
            not is_model_available("gemma3:12b"),
            reason="gemma3:12b not available",
        ),
    ),
    pytest.param(
        "gemma4:e4b",
        id="gemma4-e4b",
        marks=pytest.mark.skipif(
            not is_model_available("gemma4:e4b"),
            reason="gemma4:e4b not available",
        ),
    ),
]


@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_model_connectivity(model_name: str) -> None:
    """Single-call smoke test per model.

    Verifies connectivity, structured output parsing,
    and response structure in one LLM call.
    """
    engine = LLMEngine(provider="ollama", model_name=model_name)
    detector = Detector(
        name="TestDetector",
        version=1,
        engine=engine,
        system_prompt=(
            "Detect temperature from text. "
            "TOO_COLD: <=10 C. TOO_HOT: >=30 C. "
            "COMFORTABLE: 11-29 C. "
            "NOT_APPLICABLE: no temperature."
        ),
        output_model=Temperature,
        examples=[
            {
                "input": "It is freezing outside.",
                "output": {
                    "value": -5,
                    "reason": "TOO_COLD",
                },
            },
        ],
    )

    response = detector.act({"text": "It is -10 degrees outside."})

    # Structured output parsed into a valid model
    assert response.model is not None, f"{model_name}: model is None"
    assert hasattr(response.model, "reason"), (
        f"{model_name}: model missing 'reason' field"
    )
    assert hasattr(response.model, "value"), (
        f"{model_name}: model missing 'value' field"
    )

    # Response structure intact
    assert response.response, f"{model_name}: empty raw response"
    assert response.metrics.tokens > 0, f"{model_name}: no tokens reported"
    assert response.metrics.prompt_tokens > 0, (
        f"{model_name}: no prompt tokens reported"
    )
