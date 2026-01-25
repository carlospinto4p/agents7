from enum import StrEnum

from pydantic import BaseModel

from agents7.models import Property


class PersonInfo(BaseModel):
    """
    Structured information about a person extracted from text.
    Used to demonstrate the Labeler's entity extraction capability.
    """

    name: str | None = None
    age: int | None = None
    occupation: str | None = None
    location: str | None = None


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
