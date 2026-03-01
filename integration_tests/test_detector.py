import pytest

from agents7.agents import Detector

from data import TEMPERATURE_DATA


@pytest.mark.parametrize("test_case", TEMPERATURE_DATA)
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
    input_list = [{"text": case["text"]} for case in TEMPERATURE_DATA]
    response = temperature_detector.act_in_batch(input_list)

    assert len(response) == len(TEMPERATURE_DATA), (
        f"Expected {len(TEMPERATURE_DATA)} responses but got {len(response)}"
    )

    for i, (test_case, output) in enumerate(zip(TEMPERATURE_DATA, response)):
        assert test_case["expected_reason"] == output.model.reason.name, (
            f"Expected reason '{test_case['expected_reason']}' but got "
            f"'{output.model.reason.name}' for case {i}:"
            f" '{test_case['text']}'"
        )
