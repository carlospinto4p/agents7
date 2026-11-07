import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from agents7.llm.cache import CacheConfig, ResponseCache, create_cache
from agents7.llm.dispatcher import EngineDispatcher
from agents7.llm.pricing import Pricing
from agents7.llm.retry import DEFAULT_RETRY, RetryConfig
from agents7.llm.streaming import AsyncStreamResult, StreamResult
from agents7.settings import OpenAICompatSettings


@dataclass(frozen=True)
class Availability:
    """Whether a provider answered one small request in time.

    :param answered: True when a reply came back before the limit.
    :param duration: Seconds the check took, answered or not.
    :param error: Why it did not answer, as text. None when it did.
    """

    answered: bool
    duration: float
    error: str | None = None


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
        model_name (str | None): The name of the model being
            used. For the ``openai_compat`` provider this may
            be ``None`` to read the model from the
            ``OPENAI_COMPAT_MODEL_NAME`` environment variable.
        retry_config (RetryConfig | None): Retry configuration.
            ``None`` means `DEFAULT_RETRY` — two retries, which
            is the attempt count the `openai` and `anthropic`
            SDKs used to make on their own before the framework
            switched their retrying off. Pass
            ``RetryConfig(max_retries=0)`` to disable retrying.
        fallback_model (str | None): Optional fallback model
            name **on the primary's own provider**. When the
            primary model fails after retries, the engine
            retries the call with this model. If the fallback
            also fails, the primary error is raised.

            It carries a model name and nothing else, so the
            call is priced at the primary's rates — right when
            the two models bill alike, wrong when they do not.
            Use ``fallback_engine`` whenever they differ.
        fallback_engine (LLMEngine | None): Optional fallback on
            **any** provider, declared as an engine of its own —
            its own ``provider``, ``base_url``, ``api_key``,
            ``pricing``, ``timeout`` and ``retry``. This is what
            moves a call from a paid endpoint to a local one,
            which is the case worth having: an outage or a rate
            limit at a paid provider is exactly when a local
            model rescues a run, and a same-provider fallback
            cannot help with either.

            Its ``pricing`` prices any response it answers, so a
            run half-served by a free local model reports what
            it actually cost. Which model answered is on
            ``metrics.model`` and `CallRecord.model` either way.

            Mutually exclusive with ``fallback_model``. The
            fallback engine's own ``cache`` is rejected rather
            than ignored — one cache covers the call whichever
            model answers it, and it belongs to the primary.
        pricing (Pricing | None): Per-million-token rates for
            this model. When set, every response carries an
            estimated ``cost_cents``; when ``None``,
            ``cost_cents`` stays ``None``. Prefer this over
            ``cost_per_1m_tokens`` for any provider that bills
            prompt and completion differently, or that
            discounts a repeated prompt — most now do both.
        timeout (float | None): Seconds a single HTTP request to
            the provider may take before it is abandoned. Set on
            the provider's own client, so it covers sync, async
            and streaming calls alike, and each retry attempt
            gets its own fresh budget. ``None`` (the default)
            leaves each SDK's own behaviour in place. Size it
            from measured latency, not by guesswork: an agent
            that writes ten tokens and one that writes a
            thousand need very different numbers. Distinct from
            the per-call ``timeout`` argument of `chat_async`,
            which is one deadline for the whole request,
            retries included, and is enforced on async calls
            only.
        cost_per_1m_tokens (float | None): Older shorthand for
            a model billed at one flat rate for every token,
            prompt and completion alike. **In cents per million
            tokens**, unlike `Pricing`, which is in dollars as
            providers publish them — so ``0.5`` here means half
            a cent, and is the same as
            ``pricing=Pricing.flat(0.005)``. Ignored when
            ``pricing`` is given. Prefer `Pricing`.

    Keyword Args (``openai_compat`` only):
        base_url (str | None): Custom API base URL. Falls back
            to the ``OPENAI_COMPAT_BASE_URL`` env var.
        api_key (str | None): API key. Falls back to the
            ``OPENAI_COMPAT_API_KEY`` env var, then the
            ``openai`` SDK's own ``OPENAI_API_KEY``.
    """

    def __init__(
        self,
        provider: str,
        model_name: str | None = None,
        retry: RetryConfig | None = None,
        cache: CacheConfig | None = None,
        fallback_model: str | None = None,
        cost_per_1m_tokens: float | None = None,
        pricing: Pricing | None = None,
        timeout: float | None = None,
        fallback_engine: "LLMEngine | None" = None,
        **kwargs,
    ):
        provider = provider.lower()
        if not model_name and provider == "openai_compat":
            model_name = OpenAICompatSettings().model_name
        if not model_name:
            raise ValueError(
                "`model_name` is required. For the "
                "`openai_compat` provider you may instead set "
                "the `OPENAI_COMPAT_MODEL_NAME` environment "
                "variable."
            )
        self.provider = provider
        self.model_name = model_name
        self.model_id: str = f"{provider}::{model_name}"
        self.retry_config = DEFAULT_RETRY if retry is None else retry
        self.cost_per_1m_tokens = cost_per_1m_tokens
        if pricing is None and cost_per_1m_tokens is not None:
            # That parameter has always been cents per million
            # tokens; `Pricing` is dollars. Convert rather than
            # reinterpret, so an existing caller's numbers keep
            # meaning what they meant.
            pricing = Pricing.flat(cost_per_1m_tokens / 100)
        self.pricing = pricing
        self.timeout = timeout
        if timeout is not None:
            if timeout <= 0:
                raise ValueError(
                    f"`timeout` must be > 0 seconds, got {timeout}."
                )
            kwargs["timeout"] = timeout
        self.llm_provider = self._create_provider(
            provider, model_name, **kwargs
        )
        self.fallback_engine = fallback_engine
        fallback, fallback_pricing, fallback_retry = self._build_fallback(
            provider, fallback_model, fallback_engine, pricing, **kwargs
        )
        self._dispatcher = EngineDispatcher(
            primary=self.llm_provider,
            fallback=fallback,
            retry_config=self.retry_config,
            cache=create_cache(cache) if cache is not None else None,
            pricing=pricing,
            model_id=self.model_id,
            fallback_pricing=fallback_pricing,
            fallback_retry=fallback_retry,
        )

    def _build_fallback(
        self,
        provider: str,
        fallback_model: str | None,
        fallback_engine: "LLMEngine | None",
        pricing: Pricing | None,
        **kwargs,
    ) -> tuple[object | None, Pricing | None, RetryConfig | None]:
        """Resolve the fallback into a provider and its settings.

        Two shapes reach here. ``fallback_model`` is a model
        name on the primary's own provider, so it inherits the
        primary's rates and retry settings — it has nowhere to
        declare its own. ``fallback_engine`` is a whole engine,
        so it brings its own of both, and its provider may be a
        different one entirely.

        :param provider: The primary's provider identifier.
        :param fallback_model: A model name on that provider.
        :param fallback_engine: A fully declared fallback.
        :param pricing: The primary's rates, inherited by a
            bare ``fallback_model``.
        :param kwargs: Provider arguments, forwarded when
            building a same-provider fallback.
        :return: The fallback provider, its `Pricing`, and its
            `RetryConfig` — each ``None`` when there is no
            fallback or it declares none of its own.
        :raises ValueError: If both shapes are given, or if the
            fallback engine declares a cache of its own.
        """
        if fallback_engine is not None and fallback_model:
            raise ValueError(
                "Give `fallback_model` or `fallback_engine`, not "
                "both. `fallback_model` names another model on "
                "this provider; `fallback_engine` declares a "
                "whole engine and can name any provider."
            )
        if fallback_engine is not None:
            if fallback_engine._cache is not None:
                raise ValueError(
                    "A `fallback_engine` must not declare a "
                    "`cache`. One cache covers the call whichever "
                    "model answers it, and it belongs to the "
                    "primary engine — a second one would never be "
                    "read."
                )
            return (
                fallback_engine.llm_provider,
                fallback_engine.pricing,
                fallback_engine.retry_config,
            )
        if fallback_model:
            return (
                self._create_provider(provider, fallback_model, **kwargs),
                pricing,
                None,
            )
        return None, None, None

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

            return OllamaProvider(model_name, timeout=kwargs.get("timeout"))

        if provider == "anthropic":
            from agents7.llm.anthropic import AnthropicProvider

            return AnthropicProvider(
                model_name,
                api_key=kwargs.get("api_key"),
                base_url=kwargs.get("base_url"),
                default_max_tokens=kwargs.get("default_max_tokens", 4096),
                timeout=kwargs.get("timeout"),
            )

        if provider == "openai_compat":
            from agents7.llm.openai_compat import (
                OpenAICompatProvider,
            )

            return OpenAICompatProvider(
                model_name,
                base_url=kwargs.get("base_url"),
                api_key=kwargs.get("api_key"),
                timeout=kwargs.get("timeout"),
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
        tools: list[dict] | None = None,
    ) -> dict:
        """Generate a chat response.

        See `LLMProvider.chat` for parameter and return
        value documentation.

        :param timeout: Per-call timeout in seconds. Only
            enforced for async calls. ``None`` means no
            timeout.
        :param tools: Optional list of tool definitions in
            OpenAI/Anthropic function-calling format.
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
            tools=tools,
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
        tools: list[dict] | None = None,
    ) -> dict:
        """Async version of `chat`.

        See `LLMProvider.chat` for parameter and return
        value documentation.

        :param timeout: Per-call timeout in seconds.
            Raises ``LLMException`` on expiry.
        :param tools: Optional list of tool definitions in
            OpenAI/Anthropic function-calling format.
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
            tools=tools,
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

    def supports_warmup(self) -> bool:
        """Whether this engine supports warm-up (Ollama only)."""
        return self.provider == "ollama"

    def check_available(self, timeout: float = 15.0) -> Availability:
        """Send one small request and say whether it was answered.

        For deciding whether to start work that needs the provider,
        not for doing the work. It makes one attempt straight to the
        provider, so no retry, no cache and no fallback: a retry
        would stretch the check into the wait it exists to avoid, a
        cached reply says nothing about the provider now, and a
        fallback answering would report the primary as up.

        *timeout* is a wall clock over the whole request. A provider
        under load can accept a request and then hold the connection
        open without ever sending the reply; on 14th September 2026
        DeepSeek did that for hours, and this is what catches it.

        It asks for a single token and is not written to any call
        log, so on a paid provider it costs a few prompt tokens.

        :param timeout: Seconds to wait for the reply.
        :return: Whether it answered, how long the check took, and
            why not when it did not.
        :raises ValueError: If *timeout* is not positive.
        :raises RuntimeError: If called while an event loop is
            running — use `check_available_async` there.
        """

        async def check_then_close() -> Availability:
            # The loop below ends with this call, so any connection
            # the check opened is bound to a dead loop. Closing it
            # later, from another loop, raises "Event loop is
            # closed" — one agent's `--wait` printed exactly that
            # once its contract run collected the checked engine.
            try:
                return await self.check_available_async(timeout)
            finally:
                await self.llm_provider.aclose()

        return asyncio.run(check_then_close())

    async def check_available_async(
        self, timeout: float = 15.0
    ) -> Availability:
        """Async version of `check_available`."""
        if timeout <= 0:
            raise ValueError(f"`timeout` must be > 0 seconds, got {timeout}.")
        started = time.perf_counter()
        try:
            await asyncio.wait_for(
                self.llm_provider.chat_async(
                    messages=[{"role": "user", "content": "ping"}],
                    temperature=0.0,
                    max_tokens=1,
                ),
                timeout,
            )
        except TimeoutError:
            error = f"no reply in {timeout:g} s"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        else:
            return Availability(True, time.perf_counter() - started)
        return Availability(False, time.perf_counter() - started, error)

    def supports_tools(self) -> bool:
        """Whether the underlying provider supports tool calling."""
        return self.llm_provider.supports_tools()

    def build_tool_turn_messages(
        self,
        assistant_text: str,
        tool_calls: list[dict],
        results: list[str],
    ) -> list[dict]:
        """Delegate to the provider's tool-turn message builder.

        See `LLMProvider.build_tool_turn_messages`.
        """
        return self.llm_provider.build_tool_turn_messages(
            assistant_text, tool_calls, results
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

    async def aclose(self) -> None:
        """Close the async connections of every provider this uses.

        The primary, and the fallback in either shape. Call it
        before the event loop that made the async calls ends: an
        async HTTP client left for the garbage collector tries to
        close itself on a loop that has already shut, which prints
        *"Task exception was never retrieved ... Event loop is
        closed"* after an otherwise clean run.

        The engine stays usable: each provider swaps in a fresh
        async client that opens its connections on the next call,
        in whichever loop makes it.
        """
        await self.llm_provider.aclose()
        if self.fallback_engine is not None:
            await self.fallback_engine.aclose()
        elif self._fallback_provider is not None:
            await self._fallback_provider.aclose()


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

    async def aclose(self) -> None:
        """Close the provider's async client, if it holds one.

        A no-op here; each provider that opens an async client
        overrides it. See `LLMEngine.aclose` for when to call it.
        """

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
        tools: list[dict] | None = None,
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
         :param tools: Optional list of tool/function schemas
             the model may call. ``None`` disables tool
             calling. Only honored by providers where
             `supports_tools()` is ``True``.
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
        tools: list[dict] | None = None,
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

    def supports_tools(self) -> bool:
        """Whether this provider supports tool/function calling.

        :return: ``False`` by default; tool-capable providers
            override to return ``True``.
        """
        return False

    def build_tool_turn_messages(
        self,
        assistant_text: str,
        tool_calls: list[dict],
        results: list[str],
    ) -> list[dict]:
        """Build the messages appended after a tool round.

        Returns the assistant tool-call message plus one
        tool-result message per call, in this provider's wire
        format. Overridden by tool-capable providers.

        :param assistant_text: The assistant's text alongside
            the tool calls (often empty).
        :param tool_calls: The ``{"name", "arguments"}`` dicts
            the model requested.
        :param results: Stringified tool results, aligned by
            index with *tool_calls*.
        :return: A list of message dicts to extend the history.
        :raises NotImplementedError: If the provider does not
            support tools.
        """
        raise NotImplementedError(
            f"Provider `{self.provider}` does not support tools."
        )
