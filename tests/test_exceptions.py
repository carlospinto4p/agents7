"""Unit tests for structured exception hierarchy."""

import pytest

from agents7.exceptions import (
    Agents7Exception,
    LLMException,
    ModelUnavailableError,
    ParseError,
    ProviderError,
    RateLimitError,
)


# --- Hierarchy ---


def test_parse_error_is_llm_exception():
    """ParseError inherits from LLMException."""
    assert issubclass(ParseError, LLMException)
    assert issubclass(ParseError, Agents7Exception)


def test_provider_error_is_llm_exception():
    """ProviderError inherits from LLMException."""
    assert issubclass(ProviderError, LLMException)


def test_model_unavailable_is_provider_error():
    """ModelUnavailableError inherits from ProviderError."""
    assert issubclass(ModelUnavailableError, ProviderError)
    assert issubclass(ModelUnavailableError, LLMException)


def test_rate_limit_is_provider_error():
    """RateLimitError inherits from ProviderError."""
    assert issubclass(RateLimitError, ProviderError)
    assert issubclass(RateLimitError, LLMException)


# --- Catchability ---


def test_catch_provider_errors_with_llm_exception():
    """All provider errors are catchable as LLMException."""
    with pytest.raises(LLMException):
        raise ProviderError("test")

    with pytest.raises(LLMException):
        raise ModelUnavailableError("test")

    with pytest.raises(LLMException):
        raise RateLimitError("test")


def test_catch_specific_provider_errors():
    """Specific errors are catchable by their own type."""
    with pytest.raises(ModelUnavailableError):
        raise ModelUnavailableError("model not found")

    with pytest.raises(RateLimitError):
        raise RateLimitError("429 too many requests")

    with pytest.raises(ParseError):
        raise ParseError("invalid json")
