"""OpenAI-compatible LLM provider.

Works with any server exposing the OpenAI chat/completions API:
OpenAI, Azure OpenAI, vLLM, LM Studio, Ollama (OpenAI mode),
and other compatible backends.

Requires the optional ``openai`` package::

    uv pip install openai
"""

import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from agents7.llm.base import LLMProvider
from agents7.llm.streaming import AsyncStreamResult, StreamResult
from agents7.settings import OpenAICompatSettings

logger = logging.getLogger(__name__)

try:
    import openai as _openai
except ImportError:
    _openai = None


def _require_openai() -> None:
    if _openai is None:
        raise ImportError(
            "The `openai` package is required for the "
            "OpenAI-compatible provider. Install it with: "
            "uv pip install openai"
        )


class OpenAICompatProvider(LLMProvider):
    """LLM provider for OpenAI-compatible APIs.

    Any argument left as ``None`` falls back to the matching
    field of `OpenAICompatSettings` (i.e. the
    ``OPENAI_COMPAT_MODEL_NAME`` / ``OPENAI_COMPAT_BASE_URL`` /
    ``OPENAI_COMPAT_API_KEY`` environment variables), so an
    external provider like DeepSeek can be configured purely
    from the environment.

    Attributes:
        model_name (str | None): Model identifier (e.g.
            ``"gpt-4o"``, ``"deepseek-chat"``). ``None`` reads
            the ``OPENAI_COMPAT_MODEL_NAME`` env var.
        base_url (str | None): Custom API base URL. ``None``
            reads the ``OPENAI_COMPAT_BASE_URL`` env var, then
            the OpenAI default endpoint.
        api_key (str | None): API key. ``None`` reads the
            ``OPENAI_COMPAT_API_KEY`` env var, then the
            ``openai`` SDK's own ``OPENAI_API_KEY`` fallback.
        timeout (float | None): Seconds a single HTTP request
            may take before it is abandoned. ``None`` leaves the
            SDK's own default in place. Applies to sync, async
            and streaming calls alike. The SDK's own retrying is
            off in every case, so the limit means what it says
            and retry is the engine's `RetryConfig` alone.

    :raises ValueError: If no model name is given, either
        explicitly or via ``OPENAI_COMPAT_MODEL_NAME``.
    """

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ):
        _require_openai()
        settings = OpenAICompatSettings()
        model_name = model_name or settings.model_name
        if not model_name:
            raise ValueError(
                "No model configured for the OpenAI-compatible "
                "provider. Pass `model_name` or set the "
                "`OPENAI_COMPAT_MODEL_NAME` environment variable."
            )
        base_url = base_url if base_url is not None else settings.base_url
        api_key = api_key if api_key is not None else settings.api_key
        super().__init__("openai_compat", model_name)
        self._base_url = base_url
        self._api_key = api_key
        # The SDK's own default is NOT_GIVEN, not None, so the
        # keys are omitted rather than passed as None when no
        # timeout is configured, leaving stock SDK behaviour.
        #
        # `max_retries=0` is unconditional as of agents7 v5.3.0.
        # The SDK retries twice by itself, which made a declared
        # timeout mean three times what it said (measured: a 0.5s
        # timeout took 3.39s) and hid those attempts from
        # `LLMMetrics.retries` — three attempts could be made
        # while the response reported none. Retry belongs to the
        # engine's own `RetryConfig`, which counts and times what
        # it does; `DEFAULT_RETRY` takes this layer's place when
        # a caller names nothing.
        timeout_kwarg = {} if timeout is None else {"timeout": timeout}
        self._client = _openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            **timeout_kwarg,
        )
        self._async_client_kwargs = {
            "base_url": base_url,
            "api_key": api_key,
            "max_retries": 0,
            **timeout_kwarg,
        }
        self._async_client = _openai.AsyncOpenAI(**self._async_client_kwargs)
        # Set once an endpoint has told us it will not take a
        # json_schema response format, so the rejected request is
        # paid for at most once per provider instance.
        self._schema_unsupported = False

    async def aclose(self) -> None:
        """Close the async client's connections and start a fresh one.

        See `LLMProvider.aclose`. The sync client needs no loop to
        close, so it is left to the garbage collector.
        """
        closing = self._async_client
        self._async_client = _openai.AsyncOpenAI(**self._async_client_kwargs)
        await closing.close()

    # --- Helpers ---

    @staticmethod
    def _is_schema_rejection(exc: Exception) -> bool:
        """Is *exc* an endpoint refusing the json_schema format?

        "OpenAI-compatible" is a spectrum: DeepSeek, for one, answers
        ``400 — This response_format type is unavailable now`` to the
        `json_schema` block `_build_params` sends for any agent with
        an ``output_model``, while supporting plain ``json_object``
        perfectly well. Matched narrowly — a 400 that says nothing
        about the response format is somebody else's bug (a bad model
        name, a malformed message) and must keep propagating.

        Total by construction: it runs inside an ``except`` block, so
        anything it raised would replace the error the caller is
        actually trying to report.

        :param exc: The exception raised by the SDK call.
        :return: True when the request should be retried without the
            schema.
        """
        error_cls = getattr(_openai, "BadRequestError", None)
        if not isinstance(error_cls, type) or not isinstance(exc, error_cls):
            return False
        return "response_format" in str(exc).lower().replace(" ", "_")

    def _downgrade_to_json_object(self, params: dict) -> dict | None:
        """Rebuild *params* asking for plain JSON instead of a schema.

        The schema is not simply dropped: the model still has to
        answer in JSON, and the caller's parser still validates the
        result against the Pydantic model. What is lost is the
        provider enforcing the shape up front, which is a weaker
        guarantee, not an absent one — `Labeler` already retries on a
        parse failure.

        :param params: The rejected request kwargs.
        :return: The retry kwargs, or None when the request did not
            carry a schema and so has nothing to downgrade.
        """
        fmt = params.get("response_format") or {}
        if fmt.get("type") != "json_schema":
            return None
        if not self._schema_unsupported:
            self._schema_unsupported = True
            logger.warning(
                "%s at %s rejected a json_schema response format; "
                "falling back to json_object for this provider. "
                "Output is still validated against the output model.",
                self.model_name,
                self._base_url or "the default endpoint",
            )
        return {**params, "response_format": {"type": "json_object"}}

    def _request_params(self, *args: Any, **kwargs: Any) -> dict:
        """Build request kwargs, honouring a known schema rejection.

        Same arguments as `_build_params`. Once this provider has
        seen the endpoint refuse a schema, later requests ask for
        plain JSON directly rather than being rejected again.
        """
        params = self._build_params(*args, **kwargs)
        if not self._schema_unsupported:
            return params
        return self._downgrade_to_json_object(params) or params

    def _create(self, params: dict):
        """Call the endpoint, retrying once without the schema.

        :param params: Request kwargs from `_request_params`.
        :return: The SDK's response (or stream) object.
        """
        try:
            return self._client.chat.completions.create(**params)
        except Exception as exc:
            retry = (
                self._downgrade_to_json_object(params)
                if self._is_schema_rejection(exc)
                else None
            )
            if retry is None:
                raise
            return self._client.chat.completions.create(**retry)

    async def _create_async(self, params: dict):
        """Async version of `_create`.

        :param params: Request kwargs from `_request_params`.
        :return: The SDK's response (or stream) object.
        """
        try:
            return await self._async_client.chat.completions.create(**params)
        except Exception as exc:
            retry = (
                self._downgrade_to_json_object(params)
                if self._is_schema_rejection(exc)
                else None
            )
            if retry is None:
                raise
            return await self._async_client.chat.completions.create(**retry)

    @staticmethod
    def _build_params(
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        output_format: str | dict | None,
        seed: int | None,
        stream: bool = False,
        tools: list[dict] | None = None,
    ) -> dict:
        """Build kwargs for the OpenAI chat completions call.

        :param tools: Optional tool schemas. The registry already
            emits OpenAI's native ``{"type": "function",
            "function": {...}}`` shape, so they pass through
            unchanged.
        """
        params: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens > 0:
            params["max_tokens"] = max_tokens
        if seed is not None:
            params["seed"] = seed
        if tools:
            params["tools"] = tools
        if isinstance(output_format, dict):
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": output_format,
                    "strict": False,
                },
            }
        elif output_format == "json":
            params["response_format"] = {"type": "json_object"}
        if stream:
            params["stream_options"] = {"include_usage": True}
        return params

    @classmethod
    def _parse_response(cls, response, elapsed: float = 0.0) -> dict:
        """Parse a non-streaming chat completion response.

        :param response: The provider's completion object.
        :param elapsed: Wall-clock seconds the request took,
            measured by the caller. The OpenAI-compatible API
            reports no server-side duration of its own, so the
            caller times it — unlike Ollama, which returns one.
        """
        choice = response.choices[0]
        content = choice.message.content or ""
        thinking = None
        if hasattr(choice.message, "reasoning_content"):
            thinking = choice.message.reasoning_content
        usage = response.usage
        return {
            "response": content.strip(),
            "role": choice.message.role,
            "thinking": thinking.strip() if thinking else None,
            "tool_calls": cls._parse_tool_calls(choice.message),
            "metrics": {
                "tokens": usage.completion_tokens or 0,
                "prompt_tokens": usage.prompt_tokens or 0,
                "cached_prompt_tokens": cls._cached_prompt_tokens(usage),
                "total_duration": elapsed,
                "model": cls._model_id(response),
            },
        }

    @staticmethod
    def _model_id(response) -> str | None:
        """Name the model that answered, as ``openai_compat::x``.

        Taken from the reply, which is the only place the
        *resolved* name appears: an endpoint is free to serve an
        alias with something else, and DeepSeek does — ask for
        `deepseek-chat` and the reply says `deepseek-v4-flash`.
        It also differs from the configured name whenever a
        `fallback_model` answered.

        :param response: The provider's completion object.
        :return: ``openai_compat::<model>``, or ``None``.
        """
        model = getattr(response, "model", None)
        return f"openai_compat::{model}" if model else None

    @staticmethod
    def _cached_prompt_tokens(usage) -> int:
        """How many prompt tokens came from the prompt cache.

        Two shapes are in the wild and neither is guaranteed
        present, so both are probed and anything missing counts
        as zero — an uncached call and a provider that does not
        report caching are then indistinguishable, which is the
        right answer for a cost estimate either way.

        - OpenAI's own: ``usage.prompt_tokens_details.cached_tokens``.
        - DeepSeek's: ``usage.prompt_cache_hit_tokens``.

        :param usage: The completion's usage object.
        :return: Cached prompt tokens, or 0.
        """
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None)
        if cached is None:
            cached = getattr(usage, "prompt_cache_hit_tokens", None)
        return int(cached or 0)

    @staticmethod
    def _parse_tool_calls(message) -> list[dict]:
        """Extract tool calls from a chat completion message.

        OpenAI returns ``message.tool_calls`` as a list of
        objects whose ``function.arguments`` is a JSON *string*;
        it is decoded into a dict to match the provider-neutral
        ``{"name", "arguments", "id"}`` shape consumed by
        `ToolAgent`. A malformed or empty argument string yields
        an empty dict rather than raising.

        :param message: The ``choices[0].message`` object.
        :return: A list of ``{"name", "arguments", "id"}`` dicts
            (empty when the model requested no tools).
        """
        raw = getattr(message, "tool_calls", None) or []
        calls: list[dict] = []
        for call in raw:
            fn = call.function
            arguments = fn.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments else {}
                except json.JSONDecodeError:
                    arguments = {}
            calls.append(
                {
                    "name": fn.name,
                    "arguments": arguments,
                    "id": call.id,
                }
            )
        return calls

    # --- Chat (sync / async) ---

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
        """See `LLMProvider.chat` for documentation."""
        params = self._request_params(
            self.model_name,
            messages,
            temperature,
            max_tokens,
            output_format,
            seed,
            tools=tools,
        )
        start = time.monotonic()
        response = self._create(params)
        return self._parse_response(response, time.monotonic() - start)

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
        """See `LLMProvider.chat` for documentation."""
        params = self._request_params(
            self.model_name,
            messages,
            temperature,
            max_tokens,
            output_format,
            seed,
            tools=tools,
        )
        start = time.monotonic()
        response = await self._create_async(params)
        return self._parse_response(response, time.monotonic() - start)

    # --- Chat streaming (sync / async) ---

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
        params = self._request_params(
            self.model_name,
            messages,
            temperature,
            max_tokens,
            output_format,
            seed,
            stream=True,
        )
        raw_stream = self._create(params)
        adapted = _adapt_stream_sync(raw_stream)
        return StreamResult(adapted, "chat", provider="openai_compat")

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
        params = self._request_params(
            self.model_name,
            messages,
            temperature,
            max_tokens,
            output_format,
            seed,
            stream=True,
        )
        raw_stream = await self._create_async(params)
        adapted = _adapt_stream_async(raw_stream)
        return AsyncStreamResult(adapted, "chat", provider="openai_compat")

    # --- Tool calling ---

    def supports_tools(self) -> bool:
        """OpenAI-compatible APIs support tool calling."""
        return True

    def build_tool_turn_messages(
        self,
        assistant_text: str,
        tool_calls: list[dict],
        results: list[str],
    ) -> list[dict]:
        """Build OpenAI assistant + tool-result messages.

        Emits one assistant message carrying the optional text and
        an OpenAI-format ``tool_calls`` array, then one
        ``role: "tool"`` message per call linked back by
        ``tool_call_id``. Each call's ``arguments`` dict is
        re-serialised to the JSON string OpenAI expects on the
        wire. ``content`` is ``None`` when the assistant produced
        no text alongside the calls.

        See `LLMProvider.build_tool_turn_messages`.
        """
        assistant = {
            "role": "assistant",
            "content": assistant_text or None,
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"]),
                    },
                }
                for call in tool_calls
            ],
        }
        tool_msgs = [
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result,
            }
            for call, result in zip(tool_calls, results)
        ]
        return [assistant, *tool_msgs]


# --- Stream adapters ---
# Convert OpenAI streaming chunks into the namespace format
# that StreamResult / AsyncStreamResult expect (matching the
# Ollama chunk interface).


class _OAIChunk:
    """Adapter that makes an OpenAI stream chunk look like an
    Ollama chunk for the StreamResult extractor."""

    __slots__ = (
        "message",
        "done",
        "eval_count",
        "prompt_eval_count",
        "cached_prompt_tokens",
        "total_duration",
        "model",
    )

    def __init__(self, chunk: Any) -> None:
        # Every chunk carries it, so a stream cut short still
        # names what was answering. `StreamResult` reads it off
        # the final chunk and prefixes the provider.
        self.model = getattr(chunk, "model", None)
        delta = chunk.choices[0].delta if chunk.choices else None
        content = ""
        role = None
        reasoning = None
        if delta is not None:
            content = delta.content or ""
            role = delta.role
            if hasattr(delta, "reasoning_content"):
                reasoning = delta.reasoning_content or ""

        self.message = _OAIMessage(content, role, reasoning)

        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self.done = True
            self.eval_count = usage.completion_tokens or 0
            self.prompt_eval_count = usage.prompt_tokens or 0
            self.cached_prompt_tokens = (
                OpenAICompatProvider._cached_prompt_tokens(usage)
            )
            # None, not 0: this API reports no server-side duration,
            # and `StreamResult` fills it from the wall clock it
            # measured. A 0 here was indistinguishable from a call
            # that genuinely took no time, and shipped as one.
            self.total_duration = None
        else:
            self.done = False
            self.eval_count = None
            self.prompt_eval_count = None
            self.cached_prompt_tokens = None
            self.total_duration = None


class _OAIMessage:
    __slots__ = ("content", "role", "thinking")

    def __init__(
        self,
        content: str,
        role: str | None,
        thinking: str | None,
    ) -> None:
        self.content = content
        self.role = role
        self.thinking = thinking


def _adapt_stream_sync(raw_stream) -> Iterator:
    """Yield adapted chunks from a sync OpenAI stream."""
    for chunk in raw_stream:
        if not chunk.choices and not getattr(chunk, "usage", None):
            continue
        yield _OAIChunk(chunk)


async def _adapt_stream_async(raw_stream) -> AsyncIterator:
    """Yield adapted chunks from an async OpenAI stream."""
    async for chunk in raw_stream:
        if not chunk.choices and not getattr(chunk, "usage", None):
            continue
        yield _OAIChunk(chunk)
