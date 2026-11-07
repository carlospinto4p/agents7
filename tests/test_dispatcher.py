"""Unit tests for EngineDispatcher (no LLM required)."""

from unittest.mock import AsyncMock, Mock

import ollama
import pytest

from agents7.exceptions import LLMException
from agents7.llm.cache import CacheConfig, ResponseCache
from agents7.llm.dispatcher import EngineDispatcher, _cache_store_callback
from agents7.llm.pricing import Pricing
from agents7.llm.retry import RetryConfig


DUMMY_RESULT = {
    "response": "hello",
    "role": "assistant",
    "thinking": None,
    "metrics": {"tokens": 10, "prompt_tokens": 5, "total_duration": 0.1},
}

COMMON = {
    "temperature": 0,
    "seed": None,
    "max_tokens": -1,
    "output_format": None,
    "think": False,
}


def make_dispatcher(
    primary=None,
    fallback=None,
    retry_config=None,
    cache_config=None,
    pricing=None,
    fallback_pricing=None,
    fallback_retry=None,
) -> EngineDispatcher:
    if primary is None:
        primary = Mock()
    return EngineDispatcher(
        primary=primary,
        fallback=fallback,
        retry_config=retry_config,
        cache=ResponseCache(cache_config)
        if cache_config is not None
        else None,
        pricing=pricing,
        model_id="ollama::test-model",
        fallback_pricing=fallback_pricing,
        fallback_retry=fallback_retry,
    )


# ── Cache ──────────────────────────────────────────────────────────────────


def test_cache_hit_skips_primary():
    """Cache hit returns cached value without calling primary."""
    primary = Mock()
    dispatcher = make_dispatcher(primary=primary, cache_config=CacheConfig())
    key = ResponseCache.make_key(
        model_id="ollama::test-model",
        method="chat",
        messages=[{"role": "user", "content": "hi"}],
        **COMMON,
    )
    cached = {"response": "cached", "role": "assistant", "metrics": {}}
    dispatcher._cache.put(key, cached)

    result = dispatcher.execute(
        "chat",
        primary.chat,
        "messages",
        [{"role": "user", "content": "hi"}],
        **COMMON,
    )
    assert result["response"] == "cached"
    primary.chat.assert_not_called()
    # A hit reports the lookup's own numbers, not the stored
    # call's — see `_as_cache_hit`.
    assert result["metrics"]["cached"] is True


def test_cache_miss_calls_primary_and_stores():
    """Cache miss calls primary and stores result."""
    primary = Mock()
    primary.chat.return_value = dict(DUMMY_RESULT)
    dispatcher = make_dispatcher(primary=primary, cache_config=CacheConfig())

    result = dispatcher.execute(
        "chat", primary.chat, "messages", [], **COMMON
    )
    assert result["response"] == "hello"
    primary.chat.assert_called_once()
    assert dispatcher._cache.size == 1


def test_nonzero_temp_bypasses_cache():
    """Non-zero temperature skips cache for both lookup and store."""
    primary = Mock()
    primary.chat.return_value = dict(DUMMY_RESULT)
    dispatcher = make_dispatcher(primary=primary, cache_config=CacheConfig())

    common = {**COMMON, "temperature": 0.7}
    dispatcher.execute("chat", primary.chat, "messages", [], **common)
    dispatcher.execute("chat", primary.chat, "messages", [], **common)

    assert primary.chat.call_count == 2
    assert dispatcher._cache.size == 0


# ── Fallback ───────────────────────────────────────────────────────────────


def test_no_fallback_primary_failure_raises():
    """Primary failure with no fallback re-raises the error."""
    primary = Mock()
    primary.chat.side_effect = LLMException("primary down")
    dispatcher = make_dispatcher(primary=primary)

    with pytest.raises(LLMException, match="primary down"):
        dispatcher.execute("chat", primary.chat, "messages", [], **COMMON)


def test_fallback_called_on_primary_failure():
    """Primary fails → fallback provider is called."""
    primary = Mock()
    primary.chat.side_effect = LLMException("primary down")
    fallback = Mock()
    fallback.chat.return_value = {"response": "fallback ok", "metrics": {}}
    dispatcher = make_dispatcher(primary=primary, fallback=fallback)

    result = dispatcher.execute(
        "chat", primary.chat, "messages", [], **COMMON
    )
    assert result["response"] == "fallback ok"
    fallback.chat.assert_called_once()


def test_fallback_failure_re_raises_primary_error():
    """Both primary and fallback fail → primary error is raised."""
    primary = Mock()
    primary.chat.side_effect = LLMException("primary down")
    fallback = Mock()
    fallback.chat.side_effect = LLMException("fallback down")
    dispatcher = make_dispatcher(primary=primary, fallback=fallback)

    with pytest.raises(LLMException, match="primary down"):
        dispatcher.execute("chat", primary.chat, "messages", [], **COMMON)


# ── Retries ────────────────────────────────────────────────────────────────


def test_retries_reach_the_response_metrics():
    """The whole pipeline carries what the retry loop counted.

    The count is taken inside `retry.py` and has to survive
    cost injection and cache storage to be worth taking.
    """
    primary = Mock()
    primary.chat.side_effect = [
        ollama.RequestError("connection refused"),
        dict(DUMMY_RESULT),
    ]
    dispatcher = make_dispatcher(
        primary=primary,
        retry_config=RetryConfig(max_retries=2, base_delay=0.001),
    )

    result = dispatcher.execute(
        "chat", primary.chat, "messages", [], **COMMON
    )

    assert result["metrics"]["retries"] == 1
    assert result["metrics"]["retry_duration"] > 0


def test_an_exhausted_call_carries_its_attempts_on_the_error():
    """A call that never succeeded still took every attempt.

    The count lived in a local `AttemptStats` and died with the
    raise, so a failed call's record could only say zero retries.
    Every failed attempt the loop counted is carried, the last one
    included.
    """
    primary = Mock()
    primary.chat.side_effect = ollama.RequestError("connection refused")
    dispatcher = make_dispatcher(
        primary=primary,
        retry_config=RetryConfig(max_retries=2, base_delay=0.001),
    )

    with pytest.raises(LLMException) as caught:
        dispatcher.execute("chat", primary.chat, "messages", [], **COMMON)

    assert caught.value.retries == 3
    assert caught.value.retry_duration > 0


def test_a_failed_fallback_carries_every_attempt_on_the_error():
    """Both models failing still reports what both tried."""
    primary = Mock()
    primary.chat.side_effect = ollama.RequestError("primary refused")
    fallback = Mock()
    fallback.chat.side_effect = ollama.RequestError("fallback refused")
    dispatcher = make_dispatcher(
        primary=primary,
        fallback=fallback,
        retry_config=RetryConfig(max_retries=1, base_delay=0.001),
    )

    with pytest.raises(LLMException) as caught:
        dispatcher.execute("chat", primary.chat, "messages", [], **COMMON)

    assert caught.value.retries == 4


def test_a_fallback_reports_every_attempt_that_led_to_it():
    """A rescued response still cost the primary's failures.

    The primary's attempts and the fallback's share one counter,
    because they were all made to produce this one response.
    """
    primary = Mock()
    primary.chat.side_effect = ollama.RequestError("connection refused")
    fallback = Mock()
    fallback.chat.return_value = {"response": "fallback ok", "metrics": {}}
    dispatcher = make_dispatcher(
        primary=primary,
        fallback=fallback,
        retry_config=RetryConfig(max_retries=2, base_delay=0.001),
    )

    result = dispatcher.execute(
        "chat", primary.chat, "messages", [], **COMMON
    )

    # Three primary attempts, all failed, then the fallback.
    assert result["metrics"]["retries"] == 3


# ── Cost injection ─────────────────────────────────────────────────────────


def test_cost_injected_into_metrics():
    """A flat rate adds cost_cents to result metrics.

    `Pricing` is dollars per million tokens, so a $1.00 rate on
    exactly one million tokens is 100 cents.
    """
    primary = Mock()
    primary.chat.return_value = {
        "response": "hi",
        "metrics": {"tokens": 1_000_000, "prompt_tokens": 0},
    }
    dispatcher = make_dispatcher(primary=primary, pricing=Pricing.flat(1.0))

    result = dispatcher.execute(
        "chat", primary.chat, "messages", [], **COMMON
    )
    assert result["metrics"]["cost_cents"] == pytest.approx(100.0)


def test_no_cost_config_metrics_unchanged():
    """Without pricing, cost_cents is not added."""
    primary = Mock()
    primary.chat.return_value = {
        "response": "hi",
        "metrics": {"tokens": 1000},
    }
    dispatcher = make_dispatcher(primary=primary, pricing=None)

    result = dispatcher.execute(
        "chat", primary.chat, "messages", [], **COMMON
    )
    assert "cost_cents" not in result["metrics"]


def test_cost_prices_input_and_output_separately():
    """Prompt and completion tokens bill at their own rates."""
    primary = Mock()
    primary.chat.return_value = {
        "response": "hi",
        "metrics": {"tokens": 1_000_000, "prompt_tokens": 1_000_000},
    }
    dispatcher = make_dispatcher(
        primary=primary, pricing=Pricing(input=0.44, output=1.32)
    )

    result = dispatcher.execute(
        "chat", primary.chat, "messages", [], **COMMON
    )
    # (0.44 + 1.32) USD = 176 cents.
    assert result["metrics"]["cost_cents"] == pytest.approx(176.0)


def test_cost_discounts_cached_prompt_tokens():
    """Cached prompt tokens bill at the cheaper cached rate."""
    primary = Mock()
    primary.chat.return_value = {
        "response": "hi",
        "metrics": {
            "tokens": 0,
            "prompt_tokens": 1_000_000,
            "cached_prompt_tokens": 750_000,
        },
    }
    dispatcher = make_dispatcher(
        primary=primary,
        pricing=Pricing(input=0.44, output=1.32, cached_input=0.014),
    )

    result = dispatcher.execute(
        "chat", primary.chat, "messages", [], **COMMON
    )
    # 250k fresh at 0.44 + 750k cached at 0.014, per 1M, in cents.
    expected = (0.25 * 0.44 + 0.75 * 0.014) * 100
    assert result["metrics"]["cost_cents"] == pytest.approx(expected)


# ── Async ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_cache_hit_skips_primary():
    """Async: cache hit returns cached result without LLM call."""
    primary_fn = AsyncMock(return_value=dict(DUMMY_RESULT))
    dispatcher = make_dispatcher(cache_config=CacheConfig())
    key = ResponseCache.make_key(
        model_id="ollama::test-model",
        method="chat_async",
        messages=[{"role": "user", "content": "hi"}],
        **COMMON,
    )
    cached = {"response": "cached", "role": "assistant", "metrics": {}}
    dispatcher._cache.put(key, cached)

    result = await dispatcher.execute_async(
        "chat_async",
        primary_fn,
        "messages",
        [{"role": "user", "content": "hi"}],
        **COMMON,
    )
    assert result["response"] == "cached"
    primary_fn.assert_not_called()
    assert result["metrics"]["cached"] is True


@pytest.mark.asyncio
async def test_async_fallback_on_primary_failure():
    """Async: primary fails → fallback is called."""
    primary_fn = AsyncMock(side_effect=LLMException("primary down"))
    fallback = Mock()
    fallback.chat_async = AsyncMock(
        return_value={"response": "fallback ok", "metrics": {}}
    )
    dispatcher = make_dispatcher(fallback=fallback)

    result = await dispatcher.execute_async(
        "chat_async", primary_fn, "messages", [], **COMMON
    )
    assert result["response"] == "fallback ok"
    fallback.chat_async.assert_called_once()


@pytest.mark.asyncio
async def test_timeout_raises_llm_exception():
    """execute_async with timeout raises LLMException on timeout."""
    import asyncio

    async def slow_fn(**_):
        await asyncio.sleep(10)
        return {}

    dispatcher = make_dispatcher()

    with pytest.raises(LLMException, match="timed out"):
        await dispatcher.execute_async(
            "chat_async", slow_fn, "messages", [], **COMMON, timeout=0.01
        )


# ── _cache_store_callback ──────────────────────────────────────────────────


def test_cache_store_callback_stores_value():
    """_cache_store_callback writes data to cache at the given key."""
    cache = ResponseCache(CacheConfig())
    data = {"response": "hello", "metrics": {}}
    _cache_store_callback(cache, "my-key", data)
    assert cache.get("my-key") == data


def test_cache_store_callback_overwrites_existing():
    """_cache_store_callback replaces an existing value at the same key."""
    cache = ResponseCache(CacheConfig())
    cache.put("k", {"response": "old"})
    _cache_store_callback(cache, "k", {"response": "new"})
    assert cache.get("k") == {"response": "new"}


def test_cache_store_callback_does_not_affect_other_keys():
    """_cache_store_callback only modifies the specified key."""
    cache = ResponseCache(CacheConfig())
    cache.put("other", {"response": "other"})
    _cache_store_callback(cache, "my-key", {"response": "hello"})
    assert cache.get("other") == {"response": "other"}


# ── A fallback is priced at its own rates ──────────────────────────────────
#
# Regression guard for the gap found on 4th September 2026: the
# dispatcher held one `Pricing` and applied it to every response,
# so a run rescued by a fallback was billed at the primary's
# rates. That was harmless while a fallback could only be another
# model on the same provider and nothing else; it is wrong the
# moment a paid primary falls back to a free local model, which
# is the case `fallback_engine` exists for.


def _failing_primary() -> Mock:
    """A primary whose every attempt raises."""
    primary = Mock()
    primary.chat.side_effect = LLMException("primary down")
    return primary


def _answering_fallback() -> Mock:
    """A fallback that answers with 1000 prompt + 1000 output."""
    fallback = Mock()
    fallback.chat.return_value = {
        "response": "rescued",
        "metrics": {
            "tokens": 1_000,
            "prompt_tokens": 1_000,
            "cached_prompt_tokens": 0,
        },
    }
    return fallback


def _execute(dispatcher, provider_fn) -> dict:
    """Run one chat call through the dispatcher."""
    return dispatcher.execute(
        "chat",
        provider_fn,
        "messages",
        [{"role": "user", "content": "hi"}],
        **COMMON,
    )


def test_fallback_priced_at_its_own_rates():
    """A free fallback under a paid primary costs nothing."""
    primary = _failing_primary()
    fallback = _answering_fallback()
    dispatcher = make_dispatcher(
        primary=primary,
        fallback=fallback,
        pricing=Pricing(input=3.0, output=15.0),
        fallback_pricing=Pricing(input=0.0, output=0.0),
    )

    result = _execute(dispatcher, primary.chat)

    assert result["response"] == "rescued"
    assert result["metrics"]["cost_cents"] == 0.0


def test_fallback_pricing_is_not_the_primarys():
    """The two rates give different money, and the right one wins."""
    primary = _failing_primary()
    fallback = _answering_fallback()
    dispatcher = make_dispatcher(
        primary=primary,
        fallback=fallback,
        pricing=Pricing(input=3.0, output=15.0),
        fallback_pricing=Pricing(input=0.3, output=1.5),
    )

    result = _execute(dispatcher, primary.chat)

    # A tenth of the primary's rates on the same token counts.
    primary_cost = Pricing(input=3.0, output=15.0).cost_cents(
        prompt_tokens=1_000, tokens=1_000, cached_prompt_tokens=0
    )
    assert result["metrics"]["cost_cents"] == pytest.approx(primary_cost / 10)


def test_fallback_pricing_defaults_to_the_primarys():
    """Omitting it keeps what a same-provider fallback did."""
    primary = _failing_primary()
    fallback = _answering_fallback()
    dispatcher = make_dispatcher(
        primary=primary,
        fallback=fallback,
        pricing=Pricing(input=3.0, output=15.0),
    )

    result = _execute(dispatcher, primary.chat)

    assert result["metrics"]["cost_cents"] == pytest.approx(
        Pricing(input=3.0, output=15.0).cost_cents(
            prompt_tokens=1_000, tokens=1_000, cached_prompt_tokens=0
        )
    )


def test_primary_still_priced_at_its_own_rates():
    """A call the primary answered is unaffected by the fallback."""
    primary = Mock()
    primary.chat.return_value = {
        "response": "ok",
        "metrics": {
            "tokens": 1_000,
            "prompt_tokens": 1_000,
            "cached_prompt_tokens": 0,
        },
    }
    dispatcher = make_dispatcher(
        primary=primary,
        fallback=_answering_fallback(),
        pricing=Pricing(input=3.0, output=15.0),
        fallback_pricing=Pricing(input=0.0, output=0.0),
    )

    result = _execute(dispatcher, primary.chat)

    assert result["metrics"]["cost_cents"] > 0


def test_unpriced_fallback_under_a_priced_primary_reports_none():
    """An unpriced fallback reads as unpriced, never as free."""
    primary = _failing_primary()
    fallback = _answering_fallback()
    dispatcher = make_dispatcher(
        primary=primary,
        fallback=fallback,
        pricing=Pricing(input=3.0, output=15.0),
        fallback_pricing=None,
    )
    # `None` means "inherit", so state it after construction to
    # reach the genuinely-unpriced case.
    dispatcher._fallback_pricing = None

    result = _execute(dispatcher, primary.chat)

    assert result["metrics"].get("cost_cents") is None


async def test_async_fallback_priced_at_its_own_rates():
    """The async path prices the fallback the same way."""
    primary = Mock()
    primary.chat_async = AsyncMock(side_effect=LLMException("primary down"))
    fallback = Mock()
    fallback.chat_async = AsyncMock(
        return_value={
            "response": "rescued",
            "metrics": {
                "tokens": 1_000,
                "prompt_tokens": 1_000,
                "cached_prompt_tokens": 0,
            },
        }
    )
    dispatcher = make_dispatcher(
        primary=primary,
        fallback=fallback,
        pricing=Pricing(input=3.0, output=15.0),
        fallback_pricing=Pricing(input=0.0, output=0.0),
    )

    result = await dispatcher.execute_async(
        "chat_async",
        primary.chat_async,
        "messages",
        [{"role": "user", "content": "hi"}],
        **COMMON,
    )

    assert result["response"] == "rescued"
    assert result["metrics"]["cost_cents"] == 0.0


def test_fallback_uses_its_own_retry_config():
    """A local fallback retries on its own budget, not the primary's."""
    primary = _failing_primary()
    fallback = Mock()
    fallback.chat.side_effect = [
        ollama.RequestError("connection refused"),
        {"response": "rescued", "metrics": {}},
    ]
    dispatcher = make_dispatcher(
        primary=primary,
        fallback=fallback,
        retry_config=RetryConfig(max_retries=0),
        fallback_retry=RetryConfig(max_retries=2, base_delay=0.001),
    )

    result = _execute(dispatcher, primary.chat)

    assert result["response"] == "rescued"
    # The primary got one attempt, the fallback two.
    assert primary.chat.call_count == 1
    assert fallback.chat.call_count == 2


def test_fallback_retry_defaults_to_the_primarys():
    """Omitting it keeps the single shared budget of before."""
    dispatcher = make_dispatcher(retry_config=RetryConfig(max_retries=2))
    assert dispatcher._fallback_retry == RetryConfig(max_retries=2)
