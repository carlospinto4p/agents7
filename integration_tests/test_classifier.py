import pytest

from agents7.agents import Classifier

from data import SENTIMENT_DATA
from integration_helpers import run_batch_and_report, run_single_case


@pytest.mark.parametrize("test_case", SENTIMENT_DATA)
def test_single_sentiment_classification(
    sentiment_classifier: Classifier, test_case: dict
) -> None:
    """Test single sentiment classification cases."""
    run_single_case(
        sentiment_classifier,
        test_case,
        "category",
        "expected_category",
    )


def test_batch_sentiment_classification(
    sentiment_classifier: Classifier, reports_dir
) -> None:
    """Test batch processing and generate eval report."""
    run_batch_and_report(
        sentiment_classifier,
        SENTIMENT_DATA,
        "category",
        "expected_category",
        reports_dir,
        "classifier.md",
        confusion_field="category",
        notes=[
            "NOT_APPLICABLE is **disabled** on the "
            "Sentiment model "
            "(`NOT_APPLICABLE = False`). The NEUTRAL "
            "category covers inputs with no clear "
            "sentiment.",
        ],
    )
