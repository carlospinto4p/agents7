"""Unit tests for agents7.models.base model classes."""

from enum import StrEnum

import pytest
from pydantic import ValidationError

from agents7.exceptions import PropertyConfigurationError
from agents7.models.base import (
    AgentResponse,
    ChatResponse,
    ClassificationProperty,
    LLMMetrics,
    Property,
    _extend_enum_with_not_applicable,
)


# ── LLMMetrics ─────────────────────────────────────────────────────────────


def test_llm_metrics_basic():
    m = LLMMetrics(tokens=10, prompt_tokens=5, total_duration=0.5)
    assert m.tokens == 10
    assert m.prompt_tokens == 5
    assert m.total_duration == 0.5
    assert m.cost_cents is None


def test_llm_metrics_with_cost():
    m = LLMMetrics(
        tokens=100, prompt_tokens=20, total_duration=1.0, cost_cents=0.05
    )
    assert m.cost_cents == 0.05


# ── AgentResponse ───────────────────────────────────────────────────────────


def test_agent_response_basic():
    metrics = LLMMetrics(tokens=5, prompt_tokens=3, total_duration=0.1)
    r = AgentResponse(response="hello", model=None, metrics=metrics)
    assert r.response == "hello"
    assert r.model is None
    assert r.thinking is None
    assert r.idx is None


def test_agent_response_with_thinking_and_idx():
    metrics = LLMMetrics(tokens=5, prompt_tokens=3, total_duration=0.1)
    r = AgentResponse(
        response="hi", model=None, thinking="thought", metrics=metrics, idx=3
    )
    assert r.thinking == "thought"
    assert r.idx == 3


# ── ChatResponse ────────────────────────────────────────────────────────────


def test_chat_response_basic():
    metrics = LLMMetrics(tokens=8, prompt_tokens=4, total_duration=0.2)
    r = ChatResponse(response="Hi there!", metrics=metrics)
    assert r.response == "Hi there!"
    assert r.thinking is None


def test_chat_response_with_thinking():
    metrics = LLMMetrics(tokens=8, prompt_tokens=4, total_duration=0.2)
    r = ChatResponse(
        response="Answer", thinking="Let me think", metrics=metrics
    )
    assert r.thinking == "Let me think"


# ── _extend_enum_with_not_applicable ────────────────────────────────────────


class _SimpleReason(StrEnum):
    YES = "Yes."
    NO = "No."


def test_extend_enum_adds_not_applicable():
    extended = _extend_enum_with_not_applicable(_SimpleReason)
    assert hasattr(extended, "NOT_APPLICABLE")


def test_extend_enum_preserves_originals():
    extended = _extend_enum_with_not_applicable(_SimpleReason)
    assert hasattr(extended, "YES")
    assert hasattr(extended, "NO")


def test_extend_enum_idempotent():
    """Extending an already-extended enum returns it unchanged."""
    extended = _extend_enum_with_not_applicable(_SimpleReason)
    extended_again = _extend_enum_with_not_applicable(extended)
    assert extended_again is extended


# ── Property ────────────────────────────────────────────────────────────────


class _PriceReason(StrEnum):
    CHEAP = "It is cheap."
    EXPENSIVE = "It is expensive."


class _PriceProperty(Property[float]):
    reason: _PriceReason


class _StrictPriceProperty(Property[float]):
    NOT_APPLICABLE = False
    reason: _PriceReason


def test_property_not_applicable_added_by_default():
    reason_enum = _PriceProperty.model_fields["reason"].annotation
    assert hasattr(reason_enum, "NOT_APPLICABLE")


def test_property_not_applicable_opt_out():
    reason_enum = _StrictPriceProperty.model_fields["reason"].annotation
    assert not hasattr(reason_enum, "NOT_APPLICABLE")


def test_property_value_defaults_to_none():
    p = _PriceProperty(reason=_PriceReason.CHEAP)
    assert p.value is None


def test_property_value_can_be_set():
    p = _PriceProperty(value=9.99, reason=_PriceReason.CHEAP)
    assert p.value == 9.99


def test_property_not_applicable_reason_accepted():
    """NOT_APPLICABLE reason member is a valid value."""
    reason_enum = _PriceProperty.model_fields["reason"].annotation
    p = _PriceProperty(reason=reason_enum.NOT_APPLICABLE)
    assert p.reason.name == "NOT_APPLICABLE"


# ── ClassificationProperty ──────────────────────────────────────────────────


class _Cat(StrEnum):
    POSITIVE = "Positive sentiment."
    NEGATIVE = "Negative sentiment."


class _Rsn(StrEnum):
    HAPPY = "Expresses happiness."
    SAD = "Expresses sadness."


class _Sentiment(ClassificationProperty):
    category: _Cat
    reason: _Rsn

    CATEGORY_REASONS = {
        _Cat.POSITIVE: [_Rsn.HAPPY],
        _Cat.NEGATIVE: [_Rsn.SAD],
    }


class _FreeSentiment(ClassificationProperty):
    category: _Cat
    reason: _Rsn


def test_classification_valid_reason_passes():
    s = _Sentiment(category=_Cat.POSITIVE, reason=_Rsn.HAPPY)
    assert s.category == _Cat.POSITIVE
    assert s.reason == _Rsn.HAPPY


def test_classification_invalid_reason_raises():
    with pytest.raises(PropertyConfigurationError):
        _Sentiment(category=_Cat.POSITIVE, reason=_Rsn.SAD)


def test_classification_not_applicable_category_allows_any_reason():
    """NOT_APPLICABLE category bypasses CATEGORY_REASONS validation."""
    cat_enum = _Sentiment.model_fields["category"].annotation
    rsn_enum = _Sentiment.model_fields["reason"].annotation
    s = _Sentiment(
        category=cat_enum.NOT_APPLICABLE,
        reason=rsn_enum.NOT_APPLICABLE,
    )
    assert s.category.name == "NOT_APPLICABLE"


def test_classification_not_applicable_reason_allowed_for_any_category():
    """NOT_APPLICABLE reason bypasses CATEGORY_REASONS validation."""
    rsn_enum = _Sentiment.model_fields["reason"].annotation
    s = _Sentiment(category=_Cat.POSITIVE, reason=rsn_enum.NOT_APPLICABLE)
    assert s.reason.name == "NOT_APPLICABLE"


def test_classification_empty_category_reasons_no_validation():
    """No CATEGORY_REASONS mapping means any reason is accepted."""
    s = _FreeSentiment(category=_Cat.POSITIVE, reason=_Rsn.SAD)
    assert s.category == _Cat.POSITIVE


def test_classification_confidence_none_by_default():
    s = _Sentiment(category=_Cat.POSITIVE, reason=_Rsn.HAPPY)
    assert s.confidence is None


def test_classification_confidence_valid_range():
    s = _Sentiment(category=_Cat.POSITIVE, reason=_Rsn.HAPPY, confidence=0.9)
    assert s.confidence == 0.9


def test_classification_confidence_out_of_range_raises():
    with pytest.raises(ValidationError):
        _Sentiment(category=_Cat.POSITIVE, reason=_Rsn.HAPPY, confidence=1.5)


def test_classification_confidence_zero_valid():
    s = _Sentiment(category=_Cat.POSITIVE, reason=_Rsn.HAPPY, confidence=0.0)
    assert s.confidence == 0.0
