import pytest

from agents7.agents import Regressor

from data import TEMPERATURE_DATA
from integration_helpers import run_batch_and_report, run_single_case


@pytest.mark.parametrize("test_case", TEMPERATURE_DATA)
def test_single_temperature_regression(
    temperature_regressor: Regressor, test_case: dict
) -> None:
    """Test single temperature regression cases."""
    run_single_case(
        temperature_regressor,
        test_case,
        "reason",
        "expected_reason",
    )


def test_batch_temperature_regression(
    temperature_regressor: Regressor, reports_dir
) -> None:
    """Test batch processing and generate eval report."""
    run_batch_and_report(
        temperature_regressor,
        TEMPERATURE_DATA,
        "reason",
        "expected_reason",
        reports_dir,
        "regressor.md",
        notes=[
            "NOT_APPLICABLE is **enabled** on the output "
            "model and **tested** (included in test data).",
        ],
    )


def test_regressor_extracts_numeric_value(
    temperature_regressor: Regressor,
) -> None:
    """Test that regressor extracts numeric values."""
    response = temperature_regressor.act(
        {"text": ("The thermometer shows 25 degrees, quite a nice day.")}
    )

    assert response.model.value is not None
    assert 20 <= response.model.value <= 30
    assert response.model.reason.name == "COMFORTABLE"
