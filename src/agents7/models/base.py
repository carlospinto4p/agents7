from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, Field


T = TypeVar("T")


class LLMMetrics(BaseModel):
    """
    LLM metrics class.
    """

    tokens: int = Field(description="The number of tokens in the response.")
    prompt_tokens: int = Field(
        description="The number of tokens in the prompt."
    )
    total_duration: float = Field(
        description="The total duration of the response, in seconds."
    )


class AgentResponse(BaseModel):
    """
    Agent response class.
    """

    response: str = Field(description="The generated text response.")
    model: BaseModel = Field(
        description="The structured output validated by the output model."
    )
    thinking: str | None = Field(
        default=None, description="The thinking part of the response, if any."
    )
    metrics: LLMMetrics = Field(description="Metrics of the LLM request.")
    idx: int | None = Field(
        default=None, description="Index for tracking purposes."
    )


class Property(BaseModel, Generic[T]):
    """
    A generic base class for properties used by Detector agents.

    Subclasses should specify the type parameter T for type-safe value
    extraction. The reason field must be a StrEnum with a NOT_APPLICABLE
    member for cases where the property cannot be detected.

    Example:
        class Temperature(Property[float]):
            reason: TemperaturePropertyReason
    """

    value: T | None = Field(
        default=None, description="The extracted value of the property."
    )
    reason: StrEnum = Field(
        description="The categorical reason for the classification."
    )
