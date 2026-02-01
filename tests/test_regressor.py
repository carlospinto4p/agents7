import pytest

from agents7.agents.regressor import Regressor
from agents7.models.demo import TemperaturePropertyReason


demo_temperature_data = [
    {
        "text": "This is a test text.",
        "expected_reason": "NOT_APPLICABLE",
    },
    {
        "text": "The water in the lake is starting to freeze.",
        "expected_reason": "TOO_COLD",
    },
    {
        "text": "It's scorching outside; I can barely stay in the sun.",
        "expected_reason": "TOO_HOT",
    },
    {
        "text": "A light breeze makes the evening feel very pleasant.",
        "expected_reason": "COMFORTABLE",
    },
    {
        "text": "The thermometer shows 38 degrees in the shade.",
        "expected_reason": "TOO_HOT",
    },
]


@pytest.fixture
def temperature_regressor():
    """Fixture to create a temperature regressor instance."""
    return Regressor.from_yaml(
        "examples/configs/demo_temperature_detector.yml"
    )


def test_regressor_initialization(temperature_regressor: Regressor) -> None:
    """Test that the regressor can be properly initialized from YAML config."""
    assert temperature_regressor is not None
    assert hasattr(temperature_regressor, "prompt")
    assert hasattr(temperature_regressor, "possible_reasons")


def test_regressor_prompt_exists(temperature_regressor: Regressor) -> None:
    """Test that the regressor has a valid prompt."""
    assert temperature_regressor.prompt is not None
    assert len(temperature_regressor.prompt.strip()) > 0
    assert "value" in temperature_regressor.prompt.lower()


def test_possible_reasons(temperature_regressor: Regressor) -> None:
    """Test that the regressor can return all possible reasons."""
    expected_reasons = {
        k: v for k, v in TemperaturePropertyReason.__members__.items()
    }
    assert temperature_regressor.possible_reasons == expected_reasons


@pytest.mark.parametrize("test_case", demo_temperature_data)
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
    input_list = [{"text": case["text"]} for case in demo_temperature_data]
    response = temperature_regressor.act_in_batch(input_list)

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


def test_empty_input(temperature_regressor: Regressor) -> None:
    """Test behavior with empty input."""
    response = temperature_regressor.act_in_batch([])
    assert response == []


def test_regressor_extracts_numeric_value(
    temperature_regressor: Regressor,
) -> None:
    """Test that regressor extracts numeric values when present."""
    response = temperature_regressor.act(
        {"text": "The temperature is exactly 25 degrees Celsius."}
    )

    # Should extract a value close to 25
    assert response.model.value is not None
    assert 20 <= response.model.value <= 30
    assert response.model.reason.name == "COMFORTABLE"
