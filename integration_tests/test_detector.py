import subprocess

import pytest

from agents7.agents import Detector

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


demo_temperature_data = [
    {
        "text": "This is a test text.",
        "expected_reason": "NOT_APPLICABLE",
    },
    {
        "text": "The water in the lake is starting to freeze.",
        "expected_reason": "TOO_COLD",
    },
    {
        "text": "It's scorching outside; I can barely stay in the sun.",
        "expected_reason": "TOO_HOT",
    },
    {
        "text": "A light breeze makes the evening feel very pleasant.",
        "expected_reason": "COMFORTABLE",
    },
]


@pytest.fixture
def temperature_detector():
    """Fixture to create a temperature detector instance."""
    return Detector.from_yaml(
        "examples/configs/demo_temperature_detector.yml"
    )


@pytest.mark.parametrize("test_case", demo_temperature_data)
def test_single_temperature_detection(
    temperature_detector: Detector, test_case: dict
) -> None:
    """Test single temperature detection cases."""
    input_data = {"text": test_case["text"]}
    response = temperature_detector.act(input_data)

    actual_reason = response.model.reason.name

    assert actual_reason == test_case["expected_reason"], (
        f"Expected reason '{test_case['expected_reason']}' but got "
        f"'{actual_reason}' for text: '{test_case['text']}'"
    )


def test_batch_temperature_detection(
    temperature_detector: Detector,
) -> None:
    """Test batch processing of all temperature detection cases."""
    input_list = [{"text": case["text"]} for case in demo_temperature_data]
    response = temperature_detector.act_in_batch(input_list)

    assert len(response) == len(demo_temperature_data), (
        f"Expected {len(demo_temperature_data)} responses "
        f"but got {len(response)}"
    )

    for i, (test_case, output) in enumerate(
        zip(demo_temperature_data, response)
    ):
        assert test_case["expected_reason"] == output.model.reason.name, (
            f"Expected reason '{test_case['expected_reason']}' but got "
            f"'{output.model.reason.name}' for case {i}:"
            f" '{test_case['text']}'"
        )
