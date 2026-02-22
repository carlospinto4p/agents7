from enum import StrEnum

from pydantic import BaseModel

from agents7.models import ClassificationProperty, Property


class PersonInfo(BaseModel):
    """
    Structured information about a person extracted from text.
    Used to demonstrate the Labeler's entity extraction capability.
    """

    name: str | None = None
    age: int | None = None
    occupation: str | None = None
    location: str | None = None


class PersonBio(BaseModel):
    """
    A short biography generated from extracted person information.
    Used to demonstrate the Pipeline's multi-step chaining.
    """

    bio: str


class TemperaturePropertyReason(StrEnum):
    """
    Reasons for the temperature property.
    This is a specific reason for the Temperature property.

    Note: NOT_APPLICABLE is auto-added by the Detector/Regressor
    via `_extend_enum_with_not_applicable()` at agent init time.
    """

    TOO_COLD = "The temperature is too cold."
    TOO_HOT = "The temperature is too hot."
    COMFORTABLE = "The temperature is comfortable."


class Temperature(Property[float]):
    """
    Temperature property for detecting temperature-related content.
    Extracts a numeric temperature value (in Celsius) and classifies
    the reason based on predefined categories.
    """

    reason: TemperaturePropertyReason


# Sentiment Classification Example


class SentimentCategory(StrEnum):
    """Categories for sentiment classification."""

    POSITIVE = "Positive sentiment expressed."
    NEGATIVE = "Negative sentiment expressed."
    NEUTRAL = "Neutral or objective tone."


class SentimentReason(StrEnum):
    """Reasons justifying the sentiment classification."""

    # Positive reasons
    ENTHUSIASTIC = "Expresses enthusiasm or excitement."
    GRATEFUL = "Shows gratitude or appreciation."
    HOPEFUL = "Conveys hope or optimism."
    # Negative reasons
    FRUSTRATED = "Expresses frustration or annoyance."
    DISAPPOINTED = "Shows disappointment or sadness."
    ANGRY = "Conveys anger or hostility."
    # Neutral reasons
    FACTUAL = "Primarily factual, objective tone."
    INFORMATIVE = "Provides information without emotion."
    DESCRIPTIVE = "Describes without judgment."


class Sentiment(ClassificationProperty):
    """
    Sentiment classification property for text analysis.
    Classifies text into sentiment categories with justification.
    """

    category: SentimentCategory
    reason: SentimentReason

    CATEGORY_REASONS = {
        SentimentCategory.POSITIVE: [
            SentimentReason.ENTHUSIASTIC,
            SentimentReason.GRATEFUL,
            SentimentReason.HOPEFUL,
        ],
        SentimentCategory.NEGATIVE: [
            SentimentReason.FRUSTRATED,
            SentimentReason.DISAPPOINTED,
            SentimentReason.ANGRY,
        ],
        SentimentCategory.NEUTRAL: [
            SentimentReason.FACTUAL,
            SentimentReason.INFORMATIVE,
            SentimentReason.DESCRIPTIVE,
        ],
    }
