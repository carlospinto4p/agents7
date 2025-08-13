from enum import StrEnum

from pydantic import BaseModel

from agents7.models import Property


class DemoModel(BaseModel):
    """
    A simple demo model for testing purposes.
    This model can be used to demonstrate the functionality of an agent
    without requiring a complex setup.
    """

    name: str
    description: str
    value: int


class TemperaturePropertyReason(StrEnum):
    """
    Reasons for the temperature property.
    This is a specific reason for the Temperature property.
    """

    TOO_COLD = "The temperature is too cold."
    TOO_HOT = "The temperature is too hot."
    COMFORTABLE = "The temperature is comfortable."
    NOT_APPLICABLE = "No reference nor context for any temperature detection."


class Temperature(Property):
    value: float | None = None
    reason: TemperaturePropertyReason
