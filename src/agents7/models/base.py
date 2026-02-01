from enum import StrEnum
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel, Field, model_validator


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
    model: BaseModel | None = Field(
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


class ClassificationProperty(BaseModel):
    """
    A base class for classification outputs used by Classifier agents.

    Subclasses should define category and reason StrEnums, plus a
    CATEGORY_REASONS mapping that links valid reasons to each category.

    A NOT_APPLICABLE member is automatically added to both category and
    reason enums for cases where the input cannot be classified.

    Example:
        class SentimentCategory(StrEnum):
            POSITIVE = "Positive sentiment."
            NEGATIVE = "Negative sentiment."

        class SentimentReason(StrEnum):
            ENTHUSIASTIC = "Expresses enthusiasm."
            FRUSTRATED = "Expresses frustration."

        class Sentiment(ClassificationProperty):
            category: SentimentCategory
            reason: SentimentReason

            CATEGORY_REASONS = {
                SentimentCategory.POSITIVE: [SentimentReason.ENTHUSIASTIC],
                SentimentCategory.NEGATIVE: [SentimentReason.FRUSTRATED],
            }
    """

    category: StrEnum = Field(description="The classification category.")
    reason: StrEnum = Field(
        description="The justification for the classification."
    )

    # Subclasses define mapping of valid reasons per category
    CATEGORY_REASONS: ClassVar[dict[StrEnum, list[StrEnum]]] = {}

    def __init_subclass__(cls, **kwargs):
        """
        Automatically extend category and reason enums with NOT_APPLICABLE.

        This hook runs before Pydantic processes the class, allowing us to
        modify the annotations to include the fallback members.
        """
        super().__init_subclass__(**kwargs)

        # Extend category enum
        if "category" in cls.__annotations__:
            category_enum = cls.__annotations__["category"]
            if (
                isinstance(category_enum, type)
                and issubclass(category_enum, StrEnum)
                and not hasattr(category_enum, "NOT_APPLICABLE")
            ):
                members = [(m.name, m.value) for m in category_enum]
                members.append(("NOT_APPLICABLE", NOT_APPLICABLE_DESCRIPTION))
                extended_enum = StrEnum(category_enum.__name__, members)
                cls.__annotations__["category"] = extended_enum

        # Extend reason enum
        if "reason" in cls.__annotations__:
            reason_enum = cls.__annotations__["reason"]
            if (
                isinstance(reason_enum, type)
                and issubclass(reason_enum, StrEnum)
                and not hasattr(reason_enum, "NOT_APPLICABLE")
            ):
                members = [(m.name, m.value) for m in reason_enum]
                members.append(("NOT_APPLICABLE", NOT_APPLICABLE_DESCRIPTION))
                extended_enum = StrEnum(reason_enum.__name__, members)
                cls.__annotations__["reason"] = extended_enum

    @model_validator(mode="after")
    def validate_category_reason_link(self):
        """Ensure reason is valid for the selected category."""
        from agents7.exceptions import PropertyConfigurationError

        if not self.CATEGORY_REASONS:
            return self  # No validation if mapping not defined

        # NOT_APPLICABLE category can have any reason
        if self.category.name == "NOT_APPLICABLE":
            return self

        valid_reasons = self.CATEGORY_REASONS.get(self.category, [])
        # Allow NOT_APPLICABLE reason for any category
        if self.reason.name == "NOT_APPLICABLE":
            return self

        if valid_reasons and self.reason not in valid_reasons:
            raise PropertyConfigurationError(
                f"Reason '{self.reason.name}' is not valid for "
                f"category '{self.category.name}'. "
                f"Valid reasons: {[r.name for r in valid_reasons]}"
            )
        return self
