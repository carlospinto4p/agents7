from __future__ import annotations

import functools
from abc import ABC, abstractmethod

from agents7.llm.cache import CacheConfig, ResponseCache
from agents7.llm.retry import RetryConfig, retry_async, retry_sync
from agents7.llm.streaming import AsyncStreamResult, StreamResult
from agents7.logs import logger


def _cache_store_callback(
    cache: ResponseCache,
    key: str,
    data: dict,
) -> None:
    """Module-level callback to store streaming result in cache."""
    cache.put(key, data)


class LLMEngine:
    """
    Base class for LLM engines.
    This class defines the interface for LLM engines, including methods
    for generating text and retrieving model information.

    Attributes:
        provider (str): The provider of the LLM engine (e.g., "ollama").
        model_name (str): The name of the model being used.
        retry_config (RetryConfig | None): Optional retry configuration.
        fallback_model (str | None): Optional fallback model name.
            When the primary model fails after retries, the engine
            retries the call with this model. If the fallback also
            fails, the primary error is raised.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        retry: RetryConfig | None = None,
        cache: CacheConfig | None = None,
        fallback_model: str | None = None,
    ):
        provider = provider.lower()
        self.provider = provider
        self.model_name = model_name
        self.model_id: str = f"{provider}::{model_name}"
        self.retry_config = retry
        self._cache: ResponseCache | None = (
            ResponseCache(cache) if cache is not None else None
        )
        self._fallback_provider = None

        if provider == "ollama":
            from agents7.llm.ollama import OllamaProvider

            self.llm_provider = OllamaProvider(model_name)
            if fallback_model:
                self._fallback_provider = OllamaProvider(fallback_model)

    def _cache_lookup(
        self,
        method: str,
        temperature: float,
        **kwargs,
    ) -> tuple[str | None, object | None]:
        """Check cache for a previous result.

        Returns:
            Tuple of (cache_key, cached_value). Both are
            `None` when caching is disabled or the temperature
            is non-zero.
        """
        if self._cache is None or temperature != 0:
            return None, None
        key = ResponseCache.make_key(
            model_id=self.model_id,
            method=method,
            temperature=temperature,
            **kwargs,
        )
        return key, self._cache.get(key)

    def _cache_store(
        self,
        key: str | None,
        value: object,
    ) -> None:
        """Store a result in cache if caching is active."""
        if key is not None and self._cache is not None:
            self._cache.put(key, value)

    def _set_stream_cache_callback(
        self,
        result: StreamResult | AsyncStreamResult,
        key: str | None,
    ) -> None:
        """Attach a cache-store callback to a stream result.

        When the stream completes, the callback stores the
        final data in the cache.
        """
        if key is not None:
            result._on_complete = functools.partial(
                _cache_store_callback,
                self._cache,
                key,
            )

    # --- Internal dispatch helpers ---

    def _execute(
        self,
        method_name: str,
        provider_fn,
        input_kwarg: str,
        input_val,
        **common,
    ) -> dict:
        """Sync cache-lookup → retry → fallback → cache-store."""
        key, cached = self._cache_lookup(
            method_name,
            common["temperature"],
            **{input_kwarg: input_val},
            seed=common["seed"],
            max_tokens=common["max_tokens"],
            output_format=common["output_format"],
            think=common["think"],
        )
        if cached is not None:
            return cached
        try:
            result = retry_sync(
                provider_fn,
                self.retry_config,
                **{input_kwarg: input_val},
                **common,
            )
        except Exception as primary_error:
            if self._fallback_provider is None:
                raise
            result = self._fallback_sync(
                method_name,
                primary_error,
                input_kwarg,
                input_val,
                **common,
            )
        self._cache_store(key, result)
        return result

    async def _execute_async(
        self,
        method_name: str,
        provider_fn,
        input_kwarg: str,
        input_val,
        **common,
    ) -> dict:
        """Async cache-lookup → retry → fallback → cache-store."""
        key, cached = self._cache_lookup(
            method_name,
            common["temperature"],
            **{input_kwarg: input_val},
            seed=common["seed"],
            max_tokens=common["max_tokens"],
            output_format=common["output_format"],
            think=common["think"],
        )
        if cached is not None:
            return cached
        try:
            result = await retry_async(
                provider_fn,
                self.retry_config,
                **{input_kwarg: input_val},
                **common,
            )
        except Exception as primary_error:
            if self._fallback_provider is None:
                raise
            result = await self._fallback_async(
                method_name,
                primary_error,
                input_kwarg,
                input_val,
                **common,
            )
        self._cache_store(key, result)
        return result

    def _stream(
        self,
        method_name: str,
        provider_fn,
        cached_cls,
        input_kwarg: str,
        input_val,
        **common,
    ):
        """Sync stream: cache-lookup → call → callback."""
        key, cached = self._cache_lookup(
            method_name,
            common["temperature"],
            **{input_kwarg: input_val},
            seed=common["seed"],
            max_tokens=common["max_tokens"],
            output_format=common["output_format"],
            think=common["think"],
        )
        if cached is not None:
            return cached_cls.from_cached(cached)
        result = provider_fn(
            **{input_kwarg: input_val},
            **common,
        )
        self._set_stream_cache_callback(result, key)
        return result

    async def _stream_async(
        self,
        method_name: str,
        provider_fn,
        cached_cls,
        input_kwarg: str,
        input_val,
        **common,
    ):
        """Async stream: cache-lookup → await → callback."""
        key, cached = self._cache_lookup(
            method_name,
            common["temperature"],
            **{input_kwarg: input_val},
            seed=common["seed"],
            max_tokens=common["max_tokens"],
            output_format=common["output_format"],
            think=common["think"],
        )
        if cached is not None:
            return cached_cls.from_cached(cached)
        result = await provider_fn(
            **{input_kwarg: input_val},
            **common,
        )
        self._set_stream_cache_callback(result, key)
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
        """Try the fallback provider after primary failure."""
        logger.warning(
            f"Primary model failed: {primary_error}. Trying fallback model..."
        )
        fallback_fn = getattr(self._fallback_provider, method_name)
        try:
            return retry_sync(
                fallback_fn,
                self.retry_config,
                **{input_kwarg: input_val},
                **common,
            )
        except Exception:
            raise primary_error

    async def _fallback_async(
        self,
        method_name: str,
        primary_error: Exception,
        input_kwarg: str,
        input_val,
        **common,
    ) -> dict:
        """Try the fallback provider after primary failure
        (async)."""
        logger.warning(
            f"Primary model failed: {primary_error}. Trying fallback model..."
        )
        fallback_fn = getattr(self._fallback_provider, method_name)
        try:
            return await retry_async(
                fallback_fn,
                self.retry_config,
                **{input_kwarg: input_val},
                **common,
            )
        except Exception:
            raise primary_error

    # --- Public API ---

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """Generate a chat response.

        See `LLMProvider.chat` for parameter and return
        value documentation.
        """
        return self._execute(
            "chat",
            self.llm_provider.chat,
            "messages",
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )

    async def chat_async(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """Async version of `chat`.

        See `LLMProvider.chat` for parameter and return
        value documentation.
        """
        return await self._execute_async(
            "chat",
            self.llm_provider.chat_async,
            "messages",
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """Generate text from a prompt.

        See `LLMProvider.generate` for parameter and
        return value documentation.
        """
        return self._execute(
            "generate",
            self.llm_provider.generate,
            "prompt",
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )

    async def generate_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """Async version of `generate`.

        See `LLMProvider.generate` for parameter and
        return value documentation.
        """
        return await self._execute_async(
            "generate",
            self.llm_provider.generate_async,
            "prompt",
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> StreamResult:
        """Stream a chat response, yielding text chunks.

        Bypasses retry logic — streaming calls cannot be
        retried mid-stream.
        """
        return self._stream(
            "chat_stream",
            self.llm_provider.chat_stream,
            StreamResult,
            "messages",
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )

    async def chat_stream_async(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> AsyncStreamResult:
        """Stream a chat response asynchronously.

        Bypasses retry logic — streaming calls cannot be
        retried mid-stream.
        """
        return await self._stream_async(
            "chat_stream_async",
            self.llm_provider.chat_stream_async,
            AsyncStreamResult,
            "messages",
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )

    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> StreamResult:
        """Stream a generate response, yielding text chunks.

        Bypasses retry logic — streaming calls cannot be
        retried mid-stream.
        """
        return self._stream(
            "generate_stream",
            self.llm_provider.generate_stream,
            StreamResult,
            "prompt",
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )

    async def generate_stream_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> AsyncStreamResult:
        """Stream a generate response asynchronously.

        Bypasses retry logic — streaming calls cannot be
        retried mid-stream.
        """
        return await self._stream_async(
            "generate_stream_async",
            self.llm_provider.generate_stream_async,
            AsyncStreamResult,
            "prompt",
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )

    def get_model_info(self) -> dict:
        """
        Get information about the model. Keys:
            - provider: The provider of the LLM engine (e.g., "ollama").
            - model_name: The name of the model being used.
            - model_id: A unique identifier for the model in the format
                "<provider>::<model_name>".
        """
        return self.llm_provider.get_model_info()


class LLMProvider(ABC):
    """
    Base class for LLM providers.
    This class defines the interface for LLM engines, including methods
    for generating text and retrieving model information.

    Attributes:
        provider (str): The provider of the LLM engine (e.g., "ollama").
        model_name (str): The name of the model being used.
    """

    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        self.model_id: str = f"{provider}::{model_name}"

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """
         Generate a chat response based on the provided
         messages.
         :param messages: A list of messages to send to
             the model. Each message should be a dictionary
             with keys "role" and "content". For example:
             [
                 {"role": "user", "content": "Hello, how are you?"},
                 {"role": "assistant", "content": "I'm fine, thank you!"}
             ]
             Available roles: "system", "user",
             "assistant" or "tool".
        :param temperature: Temperature for the model.
         :param max_tokens: Maximum number of tokens to
             generate. Default is -1, which means no limit.
         :param output_format: Format of the response.
             Can be "json" or the json schema to use,
             e.g.:
             {
                 "type": "object",
                 "properties": {
                     "age": {
                         "type": "integer"
                     },
                     "available": {
                         "type": "boolean"
                     }
                 },
                 "required": [
                   "age",
                   "available"
                 ]
             }
             Easiest way to use this is to pass a Pydantic
             model `model_json_schema()` method output.
             See: https://github.com/ollama/ollama-python
         :param think: If True, the model will think
             before generating the response.
         :param seed: Seed for random number generation.
             Default is None.
         :return: A dictionary containing the parsed
             response. Schema:
             - response: The generated text response.
             - role: The role of the message
                 (e.g., "user", "assistant").
             - thinking: The thinking part of the
                 response, if any.
             - metrics: A dictionary containing the
                 following metrics:
                 - tokens: The number of tokens in
                     the response.
                 - prompt_tokens: The number of tokens
                     in the prompt.
                 - total_duration: The total duration
                     of the response, in seconds.
        """
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )

    @abstractmethod
    async def chat_async(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """Async version of `chat`.

        See `chat` for parameter and return value
        documentation.
        """
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """
        Generate text based on the provided prompt using
        the Ollama API.
        :param prompt: The input prompt to send to the
            model.
        :param temperature: Temperature for the model.
        :param max_tokens: Maximum number of tokens to
            generate. Default is -1, which means no limit.
        :param output_format: Format of the response.
            Can be "json" or the json schema to use,
            e.g.:
            {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "integer"
                    },
                    "available": {
                        "type": "boolean"
                    }
                },
                "required": [
                  "age",
                  "available"
                ]
            }
            Easiest way to use this is to pass a Pydantic
            model `model_json_schema()` method output.
            See: https://github.com/ollama/ollama-python
        :param think: If True, the model will think
            before generating the response.
        :param seed: Seed for random number generation.
            Default is None.
        :return: A dictionary containing the parsed
            response. Schema:
            - response: The generated text response.
            - thinking: The thinking part of the
                response, if any.
            - metrics: A dictionary containing the
                following metrics:
                - tokens: The number of tokens in the
                    response.
                - prompt_tokens: The number of tokens
                    in the prompt.
                - total_duration: The total duration
                    of the response, in seconds.
        """
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """Async version of `generate`.

        See `generate` for parameter and return value
        documentation.
        """
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> StreamResult:
        """Stream a chat response, yielding text chunks."""
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )

    @abstractmethod
    async def chat_stream_async(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> AsyncStreamResult:
        """Stream a chat response asynchronously."""
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> StreamResult:
        """Stream a generate response, yielding text chunks."""
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )

    @abstractmethod
    async def generate_stream_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> AsyncStreamResult:
        """Stream a generate response asynchronously."""
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )

    def get_model_info(self) -> dict:
        """
        Get information about the model. Keys:
            - provider: The provider of the LLM engine (e.g., "ollama").
            - model_name: The name of the model being used.
            - model_id: A unique identifier for the model in the format
                "<provider>::<model_name>".
        """
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "model_id": f"{self.model_id}",
        }
