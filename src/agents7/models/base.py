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


NOT_APPLICABLE_DESCRIPTION = "No applicable reason for this classification."


class Property(BaseModel, Generic[T]):
    """
    A generic base class for properties used by Detector agents.

    Subclasses should specify the type parameter T for type-safe value
    extraction. The reason field must be a StrEnum defining the possible
    classification reasons.

    A NOT_APPLICABLE member is automatically added to all reason enums
    for cases where the property cannot be detected from the input.

    Example:
        class TemperatureReason(StrEnum):
            TOO_COLD = "The temperature is too cold."
            TOO_HOT = "The temperature is too hot."
            COMFORTABLE = "The temperature is comfortable."
            # NOT_APPLICABLE is added automatically

        class Temperature(Property[float]):
            reason: TemperatureReason
    """

    value: T | None = Field(
        default=None, description="The extracted value of the property."
    )
    reason: StrEnum = Field(
        description="The categorical reason for the classification."
    )

    def __init_subclass__(cls, **kwargs):
        """
        Automatically extend the reason enum with NOT_APPLICABLE if not present.

        This hook runs before Pydantic processes the class, allowing us to
        modify the reason annotation to include the fallback member.
        """
        super().__init_subclass__(**kwargs)

        if "reason" not in cls.__annotations__:
            return

        reason_enum = cls.__annotations__["reason"]

        # Skip if not a StrEnum subclass
        if not (
            isinstance(reason_enum, type) and issubclass(reason_enum, StrEnum)
        ):
            return

        # Skip if already has NOT_APPLICABLE
        if hasattr(reason_enum, "NOT_APPLICABLE"):
            return

        # Create extended enum with NOT_APPLICABLE
        members = [(m.name, m.value) for m in reason_enum]
        members.append(("NOT_APPLICABLE", NOT_APPLICABLE_DESCRIPTION))

        extended_enum = StrEnum(reason_enum.__name__, members)

        # Replace the annotation with the extended enum
        cls.__annotations__["reason"] = extended_enum
