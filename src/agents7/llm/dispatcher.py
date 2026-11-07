"""Dispatch pipeline: cache → retry → fallback → cost → cache-store."""

import asyncio
import functools
import time

import ollama

from agents7.exceptions import LLMException
from agents7.llm.cache import ResponseCache
from agents7.llm.pricing import Pricing
from agents7.llm.retry import (
    AttemptStats,
    RetryConfig,
    _anthropic_errors,
    _openai_errors,
    attach_attempt_stats,
    retry_async,
    retry_sync,
    timeout_errors,
)
from agents7.llm.streaming import AsyncStreamResult, StreamResult
from agents7.logs import logger


def _dispatch_errors() -> tuple:
    """Provider and LLM error types that trigger fallback or re-raise."""
    return (
        LLMException,
        ollama.RequestError,
        ollama.ResponseError,
        *_openai_errors(),
        *_anthropic_errors(),
        *timeout_errors(),
    )


def _cache_store_callback(
    cache: ResponseCache,
    key: str,
    data: dict,
) -> None:
    cache.put(key, data)


class EngineDispatcher:
    """Cache, retry, fallback, cost injection and dispatch pipeline.

    Separates cross-cutting concerns from provider creation and the
    public API so that `LLMEngine` can be a thin orchestrator.

    :param primary: Primary LLM provider instance.
    :param fallback: Optional fallback provider; used when the
        primary fails after all retries.
    :param retry_config: Retry settings, or ``None`` for no retry.
    :param cache: Initialised `ResponseCache`, or ``None``.
    :param pricing: `Pricing` for this model, or ``None`` to leave
        ``cost_cents`` unset. When set, adds ``cost_cents`` to every
        result's metrics dict.
    :param model_id: Composite ``provider::model_name`` string used
        as part of cache keys.
    :param fallback_pricing: `Pricing` for the fallback, used in
        place of *pricing* on a response the fallback answered.
        ``None`` means the fallback is billed at the primary's
        rates, which is right only when the two are the same
        model on the same provider. A fallback on another
        provider always has its own — see
        `LLMEngine.fallback_engine`.
    :param fallback_retry: Retry settings for the fallback's own
        attempts, or ``None`` to reuse *retry_config*. A local
        fallback and a paid primary rarely want the same ones.
    """

    def __init__(
        self,
        primary,
        fallback,
        retry_config: RetryConfig | None,
        cache: ResponseCache | None,
        pricing: Pricing | None,
        model_id: str,
        fallback_pricing: Pricing | None = None,
        fallback_retry: RetryConfig | None = None,
    ):
        self._primary = primary
        self._fallback = fallback
        self._retry_config = retry_config
        self._cache = cache
        self._pricing = pricing
        self._model_id = model_id
        self._fallback_pricing = (
            pricing if fallback_pricing is None else fallback_pricing
        )
        self._fallback_retry = (
            retry_config if fallback_retry is None else fallback_retry
        )

    # --- Cache helpers ---

    def _cache_lookup(
        self,
        method: str,
        temperature: float,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> tuple[str | None, object | None]:
        if self._cache is None or temperature != 0:
            return None, None
        key = ResponseCache.make_key(
            model_id=self._model_id,
            method=method,
            temperature=temperature,
            tools=tools,
            **kwargs,
        )
        return key, self._cache.get(key)

    def _cache_store(self, key: str | None, value: object) -> None:
        if key is not None and self._cache is not None:
            self._cache.put(key, value)

    def _set_stream_cache_callback(
        self,
        result: StreamResult | AsyncStreamResult,
        key: str | None,
    ) -> None:
        # Fill the dispatcher's own fields before the stream is
        # cached or read. A streamed call used to skip cost
        # injection entirely, so `cost_cents` was `None` on every
        # streamed response however the engine was priced — and
        # streaming is the path a chat UI uses.
        result._on_metrics = self._finalize_metrics
        if key is not None:
            result._on_complete = functools.partial(
                _cache_store_callback,
                self._cache,
                key,
            )

    def _cache_check(
        self,
        method_name: str,
        input_kwarg: str,
        input_val,
        **common,
    ) -> tuple[str | None, dict | None]:
        """Look the call up, and re-measure a hit as its own call.

        :return: The cache key (None when caching is off or the
            temperature is non-zero) and the hit, already
            carrying the hit's own numbers, or None on a miss.
        """
        started = time.perf_counter()
        key, cached = self._cache_lookup(
            method_name,
            common["temperature"],
            **{input_kwarg: input_val},
            seed=common["seed"],
            max_tokens=common["max_tokens"],
            output_format=common["output_format"],
            think=common["think"],
            tools=common.get("tools"),
        )
        if cached is None:
            return key, None
        return key, self._as_cache_hit(cached, time.perf_counter() - started)

    def _as_cache_hit(self, cached: dict, elapsed: float) -> dict:
        """Return *cached* carrying the hit's own measurements.

        A hit is a real call with real numbers of its own: it
        took as long as the lookup took, and it cost nothing,
        because no request left the machine. Replaying the
        stored duration and cost made every total that included
        a cached call report time it did not take and money it
        did not spend.

        The token counts stay as the original call reported
        them — they describe the response being returned — and
        ``cached`` is what tells a consumer they were not
        billed a second time.

        :param cached: The stored result. Never mutated: the
            in-memory backend hands back the stored object
            itself, so writing to it would corrupt the entry
            for every later hit.
        :param elapsed: Seconds the cache lookup took.
        :return: A copy of *cached* with its own metrics.
        """
        metrics = dict(cached.get("metrics") or {})
        metrics["cached"] = True
        metrics["total_duration"] = elapsed
        # No request left the machine, so nothing could fail and
        # nothing was retried. Stated rather than left to the
        # defaults, so the stored call's retries are not replayed
        # as this one's.
        metrics["retries"] = 0
        metrics["retry_duration"] = 0.0
        # Zero only where the engine can price at all. An
        # unpriced engine has to keep reading as unpriced rather
        # than as free, or one cached call turns a whole run's
        # `None` total into `0.0`.
        metrics["cost_cents"] = 0.0 if self._pricing is not None else None
        return {**cached, "metrics": metrics}

    # --- Dispatcher-owned metrics ---

    def _finalize_metrics(
        self,
        metrics: dict,
        stats: AttemptStats | None = None,
        by_fallback: bool = False,
    ) -> None:
        """Fill the fields the dispatcher owns, in place.

        Shared by the plain and streaming paths so the two cannot
        drift — the streaming one was added after a streamed
        response was found reporting no cost at all.

        ``cached`` is stated as False rather than left to the
        model default, so "this call went to the provider" is a
        measurement in the dict and not an absence. The retry
        fields are stated for the same reason: a streamed call
        cannot be retried mid-stream, so it reports no retries
        because none were possible, not because nobody looked.

        :param metrics: The provider's metrics dict, mutated.
        :param stats: What the retry loop recorded, or ``None``
            on a path that runs no retry loop.
        :param by_fallback: Whether the fallback answered, in
            which case its own rates price the call. Billing a
            local fallback at a paid primary's rates would
            report money nobody spent.
        """
        metrics["cached"] = False
        metrics["retries"] = stats.retries if stats is not None else 0
        metrics["retry_duration"] = (
            stats.retry_duration if stats is not None else 0.0
        )
        pricing = self._fallback_pricing if by_fallback else self._pricing
        if pricing is None:
            return
        metrics["cost_cents"] = pricing.cost_cents(
            prompt_tokens=metrics.get("prompt_tokens", 0) or 0,
            tokens=metrics.get("tokens", 0) or 0,
            cached_prompt_tokens=(
                metrics.get("cached_prompt_tokens", 0) or 0
            ),
        )

    def _finalize_result(
        self,
        result: dict,
        stats: AttemptStats | None = None,
        by_fallback: bool = False,
    ) -> dict:
        """Fill the dispatcher-owned metrics on one result."""
        self._finalize_metrics(
            result.setdefault("metrics", {}), stats, by_fallback
        )
        return result

    # --- Fallback helpers ---

    def _fallback_sync(
        self,
        method_name: str,
        primary_error: Exception,
        input_kwarg: str,
        input_val,
        stats: AttemptStats,
        **common,
    ) -> dict:
        logger.warning(
            "Primary model failed: %s. Trying fallback model...",
            primary_error,
        )
        fallback_fn = getattr(self._fallback, method_name)
        try:
            return retry_sync(
                fallback_fn,
                self._fallback_retry,
                stats,
                **{input_kwarg: input_val},
                **common,
            )
        except _dispatch_errors() as e:
            logger.warning("Fallback model also failed: %s", e, exc_info=True)
            raise attach_attempt_stats(primary_error, stats)

    async def _fallback_async(
        self,
        method_name: str,
        primary_error: Exception,
        input_kwarg: str,
        input_val,
        stats: AttemptStats,
        **common,
    ) -> dict:
        logger.warning(
            "Primary model failed: %s. Trying fallback model...",
            primary_error,
        )
        fallback_fn = getattr(self._fallback, method_name)
        try:
            return await retry_async(
                fallback_fn,
                self._fallback_retry,
                stats,
                **{input_kwarg: input_val},
                **common,
            )
        except _dispatch_errors() as e:
            logger.warning("Fallback model also failed: %s", e, exc_info=True)
            raise attach_attempt_stats(primary_error, stats)

    # --- Dispatch methods ---

    def execute(
        self,
        method_name: str,
        provider_fn,
        input_kwarg: str,
        input_val,
        **common,
    ) -> dict:
        """Sync: cache-lookup → retry → fallback → cost → cache-store."""
        common.pop("timeout", None)
        key, cached = self._cache_check(
            method_name, input_kwarg, input_val, **common
        )
        if cached is not None:
            return cached
        # Shared with the fallback below: a response rescued by a
        # second model still cost every attempt the first one made.
        stats = AttemptStats()
        by_fallback = False
        try:
            result = retry_sync(
                provider_fn,
                self._retry_config,
                stats,
                **{input_kwarg: input_val},
                **common,
            )
        except _dispatch_errors() as primary_error:
            if self._fallback is None:
                attach_attempt_stats(primary_error, stats)
                raise
            result = self._fallback_sync(
                method_name,
                primary_error,
                input_kwarg,
                input_val,
                stats,
                **common,
            )
            by_fallback = True
        self._finalize_result(result, stats, by_fallback)
        self._cache_store(key, result)
        return result

    async def execute_async(
        self,
        method_name: str,
        provider_fn,
        input_kwarg: str,
        input_val,
        **common,
    ) -> dict:
        """Async: cache-lookup → retry → fallback → cost → cache-store.

        Applies timeout if ``timeout`` is present in *common*.
        """
        timeout = common.pop("timeout", None)
        key, cached = self._cache_check(
            method_name, input_kwarg, input_val, **common
        )
        if cached is not None:
            return cached

        stats = AttemptStats()
        # A list rather than a plain name so the closure can
        # report back which provider answered, which is what
        # decides the rates the call is priced at.
        answered_by_fallback: list[bool] = []

        async def _inner() -> dict:
            try:
                return await retry_async(
                    provider_fn,
                    self._retry_config,
                    stats,
                    **{input_kwarg: input_val},
                    **common,
                )
            except _dispatch_errors() as primary_error:
                if self._fallback is None:
                    attach_attempt_stats(primary_error, stats)
                    raise
                result = await self._fallback_async(
                    method_name,
                    primary_error,
                    input_kwarg,
                    input_val,
                    stats,
                    **common,
                )
                answered_by_fallback.append(True)
                return result

        try:
            if timeout is not None:
                result = await asyncio.wait_for(_inner(), timeout=timeout)
            else:
                result = await _inner()
        except asyncio.TimeoutError:
            raise attach_attempt_stats(
                LLMException(f"Request timed out after {timeout}s"), stats
            ) from None
        self._finalize_result(result, stats, bool(answered_by_fallback))
        self._cache_store(key, result)
        return result

    def stream(
        self,
        method_name: str,
        provider_fn,
        cached_cls,
        input_kwarg: str,
        input_val,
        **common,
    ):
        """Sync stream: cache-lookup → call → attach cache callback."""
        common.pop("timeout", None)
        key, cached = self._cache_check(
            method_name, input_kwarg, input_val, **common
        )
        if cached is not None:
            return cached_cls.from_cached(cached)
        result = provider_fn(**{input_kwarg: input_val}, **common)
        self._set_stream_cache_callback(result, key)
        return result

    async def stream_async(
        self,
        method_name: str,
        provider_fn,
        cached_cls,
        input_kwarg: str,
        input_val,
        **common,
    ):
        """Async stream: cache-lookup → await → attach cache callback."""
        common.pop("timeout", None)
        key, cached = self._cache_check(
            method_name, input_kwarg, input_val, **common
        )
        if cached is not None:
            return cached_cls.from_cached(cached)
        result = await provider_fn(**{input_kwarg: input_val}, **common)
        self._set_stream_cache_callback(result, key)
        return result
