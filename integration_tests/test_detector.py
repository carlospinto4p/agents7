import pytest

from agents7.agents import Detector


demo_temperature_data = [
    {
        "text": "This is a test text.",
        "expected_value": None,
        "expected_reason": "NOT_APPLICABLE",
    },
    {
        "text": "The water in the lake is starting to freeze.",
        "expected_value_min": -20,
        "expected_value_max": 0,
        "expected_reason": "TOO_COLD",
    },
    {
        "text": "It's scorching outside; I can barely stay in the sun.",
        "expected_value_min": 30,
        "expected_value_max": 40,
        "expected_reason": "TOO_HOT",
    },
    {
        "text": "A light breeze makes the evening feel very pleasant.",
        "expected_value_min": 15,
        "expected_value_max": 25,
        "expected_reason": "COMFORTABLE",
    },
    {
        "text": "The thermometer shows 38 degrees in the shade.",
        "expected_value_min": 35,
        "expected_value_max": 40,
        "expected_reason": "TOO_HOT",
    },
    {
        "text": "I walked outside and shivered immediately.",
        "expected_value_min": -20,
        "expected_value_max": 10,
        "expected_reason": "TOO_COLD",
    },
    {
        "text": "The book I ordered finally arrived today.",
        "expected_value": None,
        "expected_reason": "NOT_APPLICABLE",
    },
    {
        "text": "Summer in this city often exceeds 35 degrees.",
        "expected_value_min": 35,
        "expected_value_max": 40,
        "expected_reason": "TOO_HOT",
    },
    {
        "text": "She just bought a new pair of shoes.",
        "expected_value": None,
        "expected_reason": "NOT_APPLICABLE",
    },
    {
        "text": "The snow on the mountains hasn't melted yet.",
        "expected_value_min": -20,
        "expected_value_max": 10,
        "expected_reason": "TOO_COLD",
    },
    {
        "text": (
            "At 21 degrees, the room feels perfectly comfortable for working."
        ),
        "expected_value_min": 19,
        "expected_value_max": 23,
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
