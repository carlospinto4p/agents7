import pytest

from agents7.agents import Detector

from report_utils import write_eval_report_from_responses
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
    temperature_detector: Detector, reports_dir
) -> None:
    """Test batch processing and generate eval report."""
    input_list = [{"text": case["text"]} for case in TEMPERATURE_DATA]
    responses = temperature_detector.act_in_batch(input_list)

    assert len(responses) == len(TEMPERATURE_DATA), (
        f"Expected {len(TEMPERATURE_DATA)} responses but got {len(responses)}"
    )

    for i, (test_case, output) in enumerate(zip(TEMPERATURE_DATA, responses)):
        assert test_case["expected_reason"] == output.model.reason.name, (
            f"Expected reason '{test_case['expected_reason']}' but got "
            f"'{output.model.reason.name}' for case {i}:"
            f" '{test_case['text']}'"
        )

    # Generate report from already-collected responses
    write_eval_report_from_responses(
        temperature_detector,
        responses,
        TEMPERATURE_DATA,
        "reason",
        "expected_reason",
        reports_dir,
        "detector.md",
        notes=[
            "NOT_APPLICABLE is **enabled** on the output model "
            "and **tested** (included in test data).",
        ],
    )
