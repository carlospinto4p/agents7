import pytest

from agents7.agents import Classifier

from data import SENTIMENT_DATA


@pytest.mark.parametrize("test_case", SENTIMENT_DATA)
def test_single_sentiment_classification(
    sentiment_classifier: Classifier, test_case: dict
) -> None:
    """Test single sentiment classification cases."""
    input_data = {"text": test_case["text"]}
    response = sentiment_classifier.act(input_data)

    actual_category = response.model.category.name

    assert actual_category == test_case["expected_category"], (
        f"Expected category '{test_case['expected_category']}' "
        f"but got '{actual_category}' for text: '{test_case['text']}'"
    )


def test_batch_sentiment_classification(
    sentiment_classifier: Classifier,
) -> None:
    """Test batch processing of sentiment classification cases."""
    input_list = [{"text": case["text"]} for case in SENTIMENT_DATA]
    response = sentiment_classifier.act_in_batch(input_list)

    assert len(response) == len(SENTIMENT_DATA), (
        f"Expected {len(SENTIMENT_DATA)} responses but got {len(response)}"
    )

    for i, (test_case, output) in enumerate(zip(SENTIMENT_DATA, response)):
        assert test_case["expected_category"] == output.model.category.name, (
            f"Expected category '{test_case['expected_category']}' "
            f"but got '{output.model.category.name}' for case {i}:"
            f" '{test_case['text']}'"
        )
