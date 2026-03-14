import pytest

from agents7.agents import Regressor

from report_utils import write_eval_report_from_responses
from data import TEMPERATURE_DATA


@pytest.mark.parametrize("test_case", TEMPERATURE_DATA)
def test_single_temperature_regression(
    temperature_regressor: Regressor, test_case: dict
) -> None:
    """Test single temperature regression cases."""
    input_data = {"text": test_case["text"]}
    response = temperature_regressor.act(input_data)

    actual_reason = response.model.reason.name

    assert actual_reason == test_case["expected_reason"], (
        f"Expected reason '{test_case['expected_reason']}' but got "
        f"'{actual_reason}' for text: '{test_case['text']}'"
    )


def test_batch_temperature_regression(
    temperature_regressor: Regressor, reports_dir
) -> None:
    """Test batch processing and generate eval report."""
    input_list = [{"text": case["text"]} for case in TEMPERATURE_DATA]
    responses = temperature_regressor.act_in_batch(input_list)

    assert len(responses) == len(TEMPERATURE_DATA), (
        f"Expected {len(TEMPERATURE_DATA)} responses but got {len(responses)}"
    )

    for i, (test_case, output) in enumerate(zip(TEMPERATURE_DATA, responses)):
        assert test_case["expected_reason"] == output.model.reason.name, (
            f"Expected reason '{test_case['expected_reason']}' "
            f"but got '{output.model.reason.name}' for case {i}:"
            f" '{test_case['text']}'"
        )

    # Generate report from already-collected responses
    write_eval_report_from_responses(
        temperature_regressor,
        responses,
        TEMPERATURE_DATA,
        "reason",
        "expected_reason",
        reports_dir,
        "regressor.md",
        notes=[
            "NOT_APPLICABLE is **enabled** on the output model "
            "and **tested** (included in test data).",
        ],
    )


def test_regressor_extracts_numeric_value(
    temperature_regressor: Regressor,
) -> None:
    """Test that regressor extracts numeric values when present."""
    response = temperature_regressor.act(
        {"text": ("The thermometer shows 25 degrees, quite a nice day.")}
    )

    assert response.model.value is not None
    assert 20 <= response.model.value <= 30
    assert response.model.reason.name == "COMFORTABLE"
