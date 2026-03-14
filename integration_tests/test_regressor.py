import pytest

from agents7.agents import Regressor

from conftest import generate_eval_report
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
    temperature_regressor: Regressor,
) -> None:
    """Test batch processing of temperature regression cases."""
    input_list = [{"text": case["text"]} for case in TEMPERATURE_DATA]
    response = temperature_regressor.act_in_batch(input_list)

    assert len(response) == len(TEMPERATURE_DATA), (
        f"Expected {len(TEMPERATURE_DATA)} responses but got {len(response)}"
    )

    for i, (test_case, output) in enumerate(zip(TEMPERATURE_DATA, response)):
        assert test_case["expected_reason"] == output.model.reason.name, (
            f"Expected reason '{test_case['expected_reason']}' "
            f"but got '{output.model.reason.name}' for case {i}:"
            f" '{test_case['text']}'"
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


def test_regressor_eval_report(
    temperature_regressor: Regressor, reports_dir
) -> None:
    """Generate an evaluation report for temperature regression."""
    eval_data = [
        {"input": {"text": c["text"]}, "reason": c["expected_reason"]}
        for c in TEMPERATURE_DATA
    ]
    generate_eval_report(
        temperature_regressor,
        eval_data,
        ["reason"],
        reports_dir,
        "regressor.md",
    )
