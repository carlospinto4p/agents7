"""Unit tests for LLMEngine.check_available (no LLM required)."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agents7.llm import LLMEngine, RetryConfig

from helpers import DUMMY_METRICS


_REPLY = {"response": "p", "thinking": None, "metrics": DUMMY_METRICS}


def _engine(**kwargs) -> LLMEngine:
    return LLMEngine(provider="ollama", model_name="qwen3", **kwargs)


async def _never_answers(**_kwargs):
    await asyncio.sleep(10)


# ── check_available Tests ──────────────────────────────────


def test_check_available_when_the_provider_answers():
    engine = _engine()
    with patch.object(
        engine.llm_provider, "chat_async", new=AsyncMock(return_value=_REPLY)
    ) as chat:
        result = engine.check_available()

    assert (result.answered, result.error) == (True, None)
    assert chat.call_args.kwargs["max_tokens"] == 1


def test_check_available_gives_up_at_the_limit():
    """A provider that holds the request open is not answering."""
    engine = _engine()
    with patch.object(engine.llm_provider, "chat_async", new=_never_answers):
        result = engine.check_available(timeout=0.05)

    assert result.answered is False
    assert "no reply in 0.05 s" in result.error
    assert result.duration < 1


def test_check_available_reports_the_error():
    engine = _engine()
    failing = AsyncMock(side_effect=ConnectionError("refused"))
    with patch.object(engine.llm_provider, "chat_async", new=failing):
        result = engine.check_available()

    assert result.answered is False
    assert result.error == "ConnectionError: refused"


def test_check_available_makes_one_attempt_despite_retries():
    """Retrying would turn a 15-second check into a minute."""
    engine = _engine(retry=RetryConfig(max_retries=3))
    failing = AsyncMock(side_effect=ConnectionError("refused"))
    with patch.object(engine.llm_provider, "chat_async", new=failing):
        engine.check_available()

    assert failing.call_count == 1


def test_check_available_does_not_ask_the_fallback():
    """A fallback answering would say the primary is up when it is not."""
    fallback = _engine()
    engine = _engine(fallback_engine=fallback)
    failing = AsyncMock(side_effect=ConnectionError("refused"))
    backup = AsyncMock(return_value=_REPLY)
    with (
        patch.object(engine.llm_provider, "chat_async", new=failing),
        patch.object(fallback.llm_provider, "chat_async", new=backup),
    ):
        result = engine.check_available()

    assert result.answered is False
    backup.assert_not_called()


@pytest.mark.asyncio
async def test_check_available_async_when_the_provider_answers():
    engine = _engine()
    with patch.object(
        engine.llm_provider, "chat_async", new=AsyncMock(return_value=_REPLY)
    ):
        result = await engine.check_available_async()

    assert result.answered is True


def test_check_available_rejects_a_non_positive_limit():
    with pytest.raises(ValueError, match="> 0"):
        _engine().check_available(timeout=0)
