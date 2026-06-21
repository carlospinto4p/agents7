"""Dispatch pipeline: cache → retry → fallback → cost → cache-store."""

import asyncio
import functools

import ollama

from agents7.exceptions import LLMException
from agents7.llm.cache import ResponseCache
from agents7.llm.retry import (
    RetryConfig,
    _openai_errors,
    retry_async,
    retry_sync,
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
    :param cost_per_1m_tokens: If set, adds ``cost_cents`` to every
        result's metrics dict.
    :param model_id: Composite ``provider::model_name`` string used
        as part of cache keys.
    """

    def __init__(
        self,
        primary,
        fallback,
        retry_config: RetryConfig | None,
        cache: ResponseCache | None,
        cost_per_1m_tokens: float | None,
        model_id: str,
    ):
        self._primary = primary
        self._fallback = fallback
        self._retry_config = retry_config
        self._cache = cache
        self._cost_per_1m_tokens = cost_per_1m_tokens
        self._model_id = model_id

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
    ) -> tuple[str | None, object | None]:
        return self._cache_lookup(
            method_name,
            common["temperature"],
            **{input_kwarg: input_val},
            seed=common["seed"],
            max_tokens=common["max_tokens"],
            output_format=common["output_format"],
            think=common["think"],
            tools=common.get("tools"),
        )

    # --- Cost injection ---

    def _inject_cost(self, result: dict) -> dict:
        if self._cost_per_1m_tokens is not None:
            metrics = result.get("metrics", {})
            total_tokens = metrics.get("tokens", 0) + metrics.get(
                "prompt_tokens", 0
            )
            metrics["cost_cents"] = (
                total_tokens * self._cost_per_1m_tokens / 1_000_000
            )
        return result

    # --- Fallback helpers ---

    def _fallback_sync(
        self,
        method_name: str,
        primary_error: Exception,
        input_kwarg: str,
        input_val,
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
                self._retry_config,
                **{input_kwarg: input_val},
                **common,
            )
        except _dispatch_errors() as e:
            logger.warning("Fallback model also failed: %s", e, exc_info=True)
            raise primary_error

    async def _fallback_async(
        self,
        method_name: str,
        primary_error: Exception,
        input_kwarg: str,
        input_val,
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
                self._retry_config,
                **{input_kwarg: input_val},
                **common,
            )
        except _dispatch_errors() as e:
            logger.warning("Fallback model also failed: %s", e, exc_info=True)
            raise primary_error

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
        try:
            result = retry_sync(
                provider_fn,
                self._retry_config,
                **{input_kwarg: input_val},
                **common,
            )
        except _dispatch_errors() as primary_error:
            if self._fallback is None:
                raise
            result = self._fallback_sync(
                method_name, primary_error, input_kwarg, input_val, **common
            )
        self._inject_cost(result)
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

        async def _inner() -> dict:
            try:
                return await retry_async(
                    provider_fn,
                    self._retry_config,
                    **{input_kwarg: input_val},
                    **common,
                )
            except _dispatch_errors() as primary_error:
                if self._fallback is None:
                    raise
                return await self._fallback_async(
                    method_name,
                    primary_error,
                    input_kwarg,
                    input_val,
                    **common,
                )

        try:
            if timeout is not None:
                result = await asyncio.wait_for(_inner(), timeout=timeout)
            else:
                result = await _inner()
        except asyncio.TimeoutError:
            raise LLMException(
                f"Request timed out after {timeout}s"
            ) from None
        self._inject_cost(result)
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
