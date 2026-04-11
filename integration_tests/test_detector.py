import pytest

from agents7.agents import Detector

from data import TEMPERATURE_DATA
from integration_helpers import run_batch_and_report, run_single_case


@pytest.mark.parametrize("test_case", TEMPERATURE_DATA)
def test_single_temperature_detection(
    temperature_detector: Detector, test_case: dict
) -> None:
    """Test single temperature detection cases."""
    run_single_case(
        temperature_detector,
        test_case,
        "reason",
        "expected_reason",
    )


def test_batch_temperature_detection(
    temperature_detector: Detector, reports_dir
) -> None:
    """Test batch processing and generate eval report."""
    run_batch_and_report(
        temperature_detector,
        TEMPERATURE_DATA,
        "reason",
        "expected_reason",
        reports_dir,
        "detector.md",
        notes=[
            "NOT_APPLICABLE is **enabled** on the output "
            "model and **tested** (included in test data).",
        ],
    )
