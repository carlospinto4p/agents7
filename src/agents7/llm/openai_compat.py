"""OpenAI-compatible LLM provider.

Works with any server exposing the OpenAI chat/completions API:
OpenAI, Azure OpenAI, vLLM, LM Studio, Ollama (OpenAI mode),
and other compatible backends.

Requires the optional ``openai`` package::

    uv pip install openai
"""

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from agents7.llm.base import LLMProvider
from agents7.llm.streaming import AsyncStreamResult, StreamResult
from agents7.settings import OpenAICompatSettings

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

    :raises ValueError: If no model name is given, either
        explicitly or via ``OPENAI_COMPAT_MODEL_NAME``.
    """

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
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
        self._client = _openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self._async_client = _openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    # --- Helpers ---

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
    def _parse_response(cls, response) -> dict:
        """Parse a non-streaming chat completion response."""
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
                "total_duration": 0.0,
            },
        }

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
        params = self._build_params(
            self.model_name,
            messages,
            temperature,
            max_tokens,
            output_format,
            seed,
            tools=tools,
        )
        response = self._client.chat.completions.create(**params)
        return self._parse_response(response)

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
        params = self._build_params(
            self.model_name,
            messages,
            temperature,
            max_tokens,
            output_format,
            seed,
            tools=tools,
        )
        response = await self._async_client.chat.completions.create(**params)
        return self._parse_response(response)

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
        params = self._build_params(
            self.model_name,
            messages,
            temperature,
            max_tokens,
            output_format,
            seed,
            stream=True,
        )
        raw_stream = self._client.chat.completions.create(**params)
        adapted = _adapt_stream_sync(raw_stream)
        return StreamResult(adapted, "chat")

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
        params = self._build_params(
            self.model_name,
            messages,
            temperature,
            max_tokens,
            output_format,
            seed,
            stream=True,
        )
        raw_stream = await self._async_client.chat.completions.create(
            **params
        )
        adapted = _adapt_stream_async(raw_stream)
        return AsyncStreamResult(adapted, "chat")

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
        "total_duration",
    )

    def __init__(self, chunk: Any) -> None:
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
            self.total_duration = 0
        else:
            self.done = False
            self.eval_count = None
            self.prompt_eval_count = None
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
