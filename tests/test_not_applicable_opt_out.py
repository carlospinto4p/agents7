"""Unit tests for NOT_APPLICABLE opt-out on Property and
ClassificationProperty."""

from enum import StrEnum

import pytest

from agents7.agents.classifier import Classifier
from agents7.agents.detector import Detector
from agents7.models.base import ClassificationProperty, Property

from helpers import ENGINE


# --- Property opt-out ---


class Reason(StrEnum):
    HIGH = "High."
    LOW = "Low."


class WithNA(Property[float]):
    reason: Reason


class WithoutNA(Property[float]):
    NOT_APPLICABLE = False
    reason: Reason


def test_property_default_adds_na():
    """Property adds NOT_APPLICABLE by default."""
    reason_enum = WithNA.model_fields["reason"].annotation
    assert hasattr(reason_enum, "NOT_APPLICABLE")


def test_property_opt_out_no_na():
    """Property with NOT_APPLICABLE=False skips it."""
    reason_enum = WithoutNA.model_fields["reason"].annotation
    assert not hasattr(reason_enum, "NOT_APPLICABLE")


def test_property_opt_out_enum_members():
    """Opted-out enum only has original members."""
    reason_enum = WithoutNA.model_fields["reason"].annotation
    names = [m.name for m in reason_enum]
    assert names == ["HIGH", "LOW"]


# --- ClassificationProperty opt-out ---


class Cat(StrEnum):
    GOOD = "Good."
    BAD = "Bad."


class Rsn(StrEnum):
    CLEAR = "Clear."
    VAGUE = "Vague."


class ClassWithNA(ClassificationProperty):
    category: Cat
    reason: Rsn


class ClassWithoutNA(ClassificationProperty):
    NOT_APPLICABLE = False
    category: Cat
    reason: Rsn


def test_classification_default_adds_na():
    """ClassificationProperty adds NOT_APPLICABLE by default."""
    cat_enum = ClassWithNA.model_fields["category"].annotation
    rsn_enum = ClassWithNA.model_fields["reason"].annotation
    assert hasattr(cat_enum, "NOT_APPLICABLE")
    assert hasattr(rsn_enum, "NOT_APPLICABLE")


def test_classification_opt_out_no_na():
    """ClassificationProperty with NOT_APPLICABLE=False skips."""
    cat_enum = ClassWithoutNA.model_fields["category"].annotation
    rsn_enum = ClassWithoutNA.model_fields["reason"].annotation
    assert not hasattr(cat_enum, "NOT_APPLICABLE")
    assert not hasattr(rsn_enum, "NOT_APPLICABLE")


def test_classification_opt_out_validation_works():
    """Validation still works without NOT_APPLICABLE."""

    class StrictClass(ClassificationProperty):
        NOT_APPLICABLE = False
        category: Cat
        reason: Rsn
        CATEGORY_REASONS = {
            Cat.GOOD: [Rsn.CLEAR],
            Cat.BAD: [Rsn.VAGUE],
        }

    valid = StrictClass(category=Cat.GOOD, reason=Rsn.CLEAR)
    assert valid.category == Cat.GOOD

    from agents7.exceptions import PropertyConfigurationError

    with pytest.raises(PropertyConfigurationError):
        StrictClass(category=Cat.GOOD, reason=Rsn.VAGUE)


# --- Detector prompt ---


def test_detector_prompt_includes_na_guidance():
    """Detector prompt includes NOT_APPLICABLE guidance."""
    agent = Detector(
        name="t",
        version=1,
        engine=ENGINE,
        system_prompt="Test.",
        output_model=WithNA,
    )
    assert "NOT_APPLICABLE" in agent.prompt


def test_detector_prompt_excludes_na_guidance():
    """Detector prompt excludes guidance when opted out."""
    agent = Detector(
        name="t",
        version=1,
        engine=ENGINE,
        system_prompt="Test.",
        output_model=WithoutNA,
    )
    assert "NOT_APPLICABLE Guidance" not in agent.prompt


# --- Classifier prompt ---


def test_classifier_prompt_includes_na():
    """Classifier prompt includes NOT_APPLICABLE when enabled."""
    agent = Classifier(
        name="t",
        version=1,
        engine=ENGINE,
        system_prompt="Test.",
        output_model=ClassWithNA,
    )
    assert "NOT_APPLICABLE" in agent.prompt


def test_classifier_prompt_excludes_na():
    """Classifier prompt excludes NOT_APPLICABLE when opted out."""
    agent = Classifier(
        name="t",
        version=1,
        engine=ENGINE,
        system_prompt="Test.",
        output_model=ClassWithoutNA,
    )
    assert "NOT_APPLICABLE Guidance" not in agent.prompt
    assert "NOT_APPLICABLE: NOT_APPLICABLE" not in agent.prompt
    assert "cannot be classified" not in agent.prompt
