from enum import StrEnum

import pytest
from pydantic import BaseModel

from agents7.agents.classifier import Classifier
from agents7.exceptions import PropertyConfigurationError
from agents7.llm import LLMEngine
from agents7.models import ClassificationProperty
from agents7.models.demo import Sentiment


# --- Initialization & prompt ---


def test_classifier_initialization(sentiment_classifier: Classifier) -> None:
    """Test that the classifier initializes from YAML config."""
    assert sentiment_classifier is not None
    assert hasattr(sentiment_classifier, "prompt")
    assert hasattr(sentiment_classifier, "possible_categories")
    assert hasattr(sentiment_classifier, "possible_reasons")


def test_classifier_prompt_exists(sentiment_classifier: Classifier) -> None:
    """Test that the classifier has a valid prompt."""
    assert sentiment_classifier.prompt is not None
    assert len(sentiment_classifier.prompt.strip()) > 0
    assert "category" in sentiment_classifier.prompt.lower()


# --- Enum properties ---


def test_possible_categories(sentiment_classifier: Classifier) -> None:
    """Test that the classifier can return all possible categories."""
    categories = sentiment_classifier.possible_categories
    assert "POSITIVE" in categories
    assert "NEGATIVE" in categories
    assert "NEUTRAL" in categories
    # Sentiment model sets NOT_APPLICABLE = False
    assert "NOT_APPLICABLE" not in categories


def test_possible_reasons(sentiment_classifier: Classifier) -> None:
    """Test that the classifier can return all possible reasons."""
    reasons = sentiment_classifier.possible_reasons
    assert "ENTHUSIASTIC" in reasons
    assert "FRUSTRATED" in reasons
    assert "FACTUAL" in reasons
    # Sentiment model sets NOT_APPLICABLE = False
    assert "NOT_APPLICABLE" not in reasons


# --- Edge cases ---


def test_empty_input(sentiment_classifier: Classifier) -> None:
    """Test behavior with empty input."""
    response = sentiment_classifier.act_in_batch([])
    assert response == []


# --- NOT_APPLICABLE auto-extension ---


def test_not_applicable_added_to_category() -> None:
    """Test that NOT_APPLICABLE is automatically added to category enum."""

    class SimpleCategory(StrEnum):
        A = "Category A."
        B = "Category B."

    class SimpleReason(StrEnum):
        R1 = "Reason 1."
        R2 = "Reason 2."

    class SimpleClassification(ClassificationProperty):
        category: SimpleCategory
        reason: SimpleReason

    # Verify NOT_APPLICABLE was added to both enums
    category_enum = SimpleClassification.__annotations__["category"]
    reason_enum = SimpleClassification.__annotations__["reason"]

    assert hasattr(category_enum, "NOT_APPLICABLE")
    assert hasattr(reason_enum, "NOT_APPLICABLE")


# --- Validation ---


def test_category_reason_validation() -> None:
    """Test that invalid category-reason combinations raise errors."""

    # The Sentiment class has CATEGORY_REASONS defined
    # POSITIVE reasons: ENTHUSIASTIC, GRATEFUL, HOPEFUL
    # Trying to use a NEGATIVE reason with POSITIVE category should fail

    # Get the extended enums from the Sentiment class
    category_enum = Sentiment.__annotations__["category"]
    reason_enum = Sentiment.__annotations__["reason"]

    # Valid combination should work
    valid = Sentiment(
        category=category_enum.POSITIVE,
        reason=reason_enum.ENTHUSIASTIC,
    )
    assert valid.category.name == "POSITIVE"

    # Invalid combination should raise
    with pytest.raises(PropertyConfigurationError):
        Sentiment(
            category=category_enum.POSITIVE,
            reason=reason_enum.FRUSTRATED,  # FRUSTRATED is for NEGATIVE
        )


def test_classifier_requires_classification_property() -> None:
    """Test that Classifier raises error for non-CP models."""

    class NotAClassificationProperty(BaseModel):
        value: str

    engine = LLMEngine(provider="ollama", model_name="qwen3")

    with pytest.raises(PropertyConfigurationError):
        Classifier(
            name="InvalidClassifier",
            version=1,
            engine=engine,
            system_prompt="Test prompt.",
            output_model=NotAClassificationProperty,
        )
