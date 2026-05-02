from abc import ABC, abstractmethod

from agents7.llm.cache import CacheConfig, ResponseCache, create_cache
from agents7.llm.dispatcher import EngineDispatcher
from agents7.llm.retry import RetryConfig
from agents7.llm.streaming import AsyncStreamResult, StreamResult


class LLMEngine:
    """
    Base class for LLM engines.
    This class defines the interface for LLM engines, including methods
    for generating text and retrieving model information.

    Attributes:
        provider (str): The LLM provider. Supported values:
            ``"ollama"`` (local Ollama server),
            ``"anthropic"`` (Claude via Anthropic API),
            and ``"openai_compat"`` (any OpenAI-compatible
            API).
        model_name (str): The name of the model being used.
        retry_config (RetryConfig | None): Optional retry
            configuration.
        fallback_model (str | None): Optional fallback model
            name. When the primary model fails after retries,
            the engine retries the call with this model. If
            the fallback also fails, the primary error is
            raised.

    Keyword Args (``openai_compat`` only):
        base_url (str | None): Custom API base URL.
        api_key (str | None): API key (falls back to
            ``OPENAI_API_KEY`` env var).
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        retry: RetryConfig | None = None,
        cache: CacheConfig | None = None,
        fallback_model: str | None = None,
        cost_per_1m_tokens: float | None = None,
        **kwargs,
    ):
        provider = provider.lower()
        self.provider = provider
        self.model_name = model_name
        self.model_id: str = f"{provider}::{model_name}"
        self.retry_config = retry
        self.cost_per_1m_tokens = cost_per_1m_tokens
        self.llm_provider = self._create_provider(
            provider, model_name, **kwargs
        )
        fallback = (
            self._create_provider(provider, fallback_model, **kwargs)
            if fallback_model
            else None
        )
        self._dispatcher = EngineDispatcher(
            primary=self.llm_provider,
            fallback=fallback,
            retry_config=retry,
            cache=create_cache(cache) if cache is not None else None,
            cost_per_1m_tokens=cost_per_1m_tokens,
            model_id=self.model_id,
        )

    @property
    def _cache(self) -> ResponseCache | None:
        return self._dispatcher._cache

    @property
    def _fallback_provider(self):
        return self._dispatcher._fallback

    @staticmethod
    def _create_provider(
        provider: str,
        model_name: str,
        **kwargs,
    ) -> "LLMProvider":
        """Instantiate an `LLMProvider` for the given
        provider name.

        :param provider: Provider identifier (``"ollama"``,
            ``"anthropic"``, or ``"openai_compat"``).
        :param model_name: Model name to pass to the
            provider.
        :param kwargs: Extra keyword arguments forwarded to
            the provider constructor (e.g. ``api_key``,
            ``base_url``).
        :return: An `LLMProvider` instance.
        :raises ValueError: If *provider* is not recognised.
        """
        if provider == "ollama":
            from agents7.llm.ollama import OllamaProvider

            return OllamaProvider(model_name)

        if provider == "anthropic":
            from agents7.llm.anthropic import AnthropicProvider

            return AnthropicProvider(
                model_name,
                api_key=kwargs.get("api_key"),
                base_url=kwargs.get("base_url"),
                default_max_tokens=kwargs.get("default_max_tokens", 4096),
            )

        if provider == "openai_compat":
            from agents7.llm.openai_compat import (
                OpenAICompatProvider,
            )

            return OpenAICompatProvider(
                model_name,
                base_url=kwargs.get("base_url"),
                api_key=kwargs.get("api_key"),
            )

        raise ValueError(f"Unknown provider: `{provider}`")

    # --- Public API ---

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
        timeout: float | None = None,
    ) -> dict:
        """Generate a chat response.

        See `LLMProvider.chat` for parameter and return
        value documentation.

        :param timeout: Per-call timeout in seconds. Only
            enforced for async calls. ``None`` means no
            timeout.
        """
        return self._dispatcher.execute(
            "chat",
            self.llm_provider.chat,
            "messages",
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
            timeout=timeout,
        )

    async def chat_async(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
        timeout: float | None = None,
    ) -> dict:
        """Async version of `chat`.

        See `LLMProvider.chat` for parameter and return
        value documentation.

        :param timeout: Per-call timeout in seconds.
            Raises ``LLMException`` on expiry.
        """
        return await self._dispatcher.execute_async(
            "chat",
            self.llm_provider.chat_async,
            "messages",
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
            timeout=timeout,
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
        timeout: float | None = None,
    ) -> dict:
        """Generate text from a prompt.

        See `LLMProvider.generate` for parameter and
        return value documentation.

        :param timeout: Per-call timeout in seconds.
        """
        return self._dispatcher.execute(
            "generate",
            self.llm_provider.generate,
            "prompt",
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
            timeout=timeout,
        )

    async def generate_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
        timeout: float | None = None,
    ) -> dict:
        """Async version of `generate`.

        See `LLMProvider.generate` for parameter and
        return value documentation.

        :param timeout: Per-call timeout in seconds.
        """
        return await self._dispatcher.execute_async(
            "generate",
            self.llm_provider.generate_async,
            "prompt",
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
            timeout=timeout,
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
        return self._dispatcher.stream(
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
        return await self._dispatcher.stream_async(
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
        return self._dispatcher.stream(
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
        return await self._dispatcher.stream_async(
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

        Default implementation delegates to `chat` with the
        prompt wrapped as a user message and strips the
        ``role`` key from the result. Providers with a
        native generate endpoint (e.g. Ollama) may override.

        See `chat` for parameter and return value
        documentation.
        """
        result = self.chat(
            self._prompt_to_messages(prompt),
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )
        result.pop("role", None)
        return result

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
        result = await self.chat_async(
            self._prompt_to_messages(prompt),
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )
        result.pop("role", None)
        return result

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

    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> StreamResult:
        """Stream a generate response.

        Default implementation delegates to `chat_stream`.
        Providers with a native generate endpoint may
        override.
        """
        return self.chat_stream(
            self._prompt_to_messages(prompt),
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

        Default implementation delegates to
        `chat_stream_async`. Providers with a native
        generate endpoint may override.
        """
        return await self.chat_stream_async(
            self._prompt_to_messages(prompt),
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )

    @staticmethod
    def _prompt_to_messages(
        prompt: str,
    ) -> list[dict[str, str]]:
        """Wrap a plain prompt string into a messages list."""
        return [{"role": "user", "content": prompt}]

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
