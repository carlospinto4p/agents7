"""Unit tests for EngineDispatcher (no LLM required)."""

from unittest.mock import AsyncMock, Mock

import pytest

from agents7.exceptions import LLMException
from agents7.llm.cache import CacheConfig, ResponseCache
from agents7.llm.dispatcher import EngineDispatcher, _cache_store_callback


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
    cost_per_1m_tokens=None,
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
        cost_per_1m_tokens=cost_per_1m_tokens,
        model_id="ollama::test-model",
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
    assert result == cached
    primary.chat.assert_not_called()


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


# ── Cost injection ─────────────────────────────────────────────────────────


def test_cost_injected_into_metrics():
    """cost_per_1m_tokens adds cost_cents to result metrics."""
    primary = Mock()
    primary.chat.return_value = {
        "response": "hi",
        "metrics": {"tokens": 1_000_000, "prompt_tokens": 0},
    }
    dispatcher = make_dispatcher(primary=primary, cost_per_1m_tokens=1.0)

    result = dispatcher.execute(
        "chat", primary.chat, "messages", [], **COMMON
    )
    assert result["metrics"]["cost_cents"] == pytest.approx(1.0)


def test_no_cost_config_metrics_unchanged():
    """Without cost_per_1m_tokens, cost_cents is not added."""
    primary = Mock()
    primary.chat.return_value = {
        "response": "hi",
        "metrics": {"tokens": 1000},
    }
    dispatcher = make_dispatcher(primary=primary, cost_per_1m_tokens=None)

    result = dispatcher.execute(
        "chat", primary.chat, "messages", [], **COMMON
    )
    assert "cost_cents" not in result["metrics"]


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
    assert result == cached
    primary_fn.assert_not_called()


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
