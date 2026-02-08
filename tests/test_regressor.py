import pytest

from agents7.agents.regressor import Regressor
from agents7.models.demo import TemperaturePropertyReason


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


def test_empty_input(temperature_regressor: Regressor) -> None:
    """Test behavior with empty input."""
    response = temperature_regressor.act_in_batch([])
    assert response == []
