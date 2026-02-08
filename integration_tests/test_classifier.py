import pytest

from agents7.agents import Classifier


demo_sentiment_data = [
    {
        "text": "I absolutely love this! Best day ever!",
        "expected_category": "POSITIVE",
    },
    {
        "text": "This is terrible and I'm very disappointed.",
        "expected_category": "NEGATIVE",
    },
    {
        "text": "The meeting is at 3 PM in conference room B.",
        "expected_category": "NEUTRAL",
    },
]


@pytest.fixture
def sentiment_classifier():
    """Fixture to create a sentiment classifier instance."""
    return Classifier.from_yaml(
        "examples/configs/demo_sentiment_classifier.yml"
    )


@pytest.mark.parametrize("test_case", demo_sentiment_data)
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
    input_list = [{"text": case["text"]} for case in demo_sentiment_data]
    response = sentiment_classifier.act_in_batch(input_list)

    assert len(response) == len(demo_sentiment_data), (
        f"Expected {len(demo_sentiment_data)} responses "
        f"but got {len(response)}"
    )

    for i, (test_case, output) in enumerate(
        zip(demo_sentiment_data, response)
    ):
        assert test_case["expected_category"] == output.model.category.name, (
            f"Expected category '{test_case['expected_category']}' "
            f"but got '{output.model.category.name}' for case {i}:"
            f" '{test_case['text']}'"
        )
