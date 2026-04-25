"""OpenAI-compatible LLM provider.

Works with any server exposing the OpenAI chat/completions API:
OpenAI, Azure OpenAI, vLLM, LM Studio, Ollama (OpenAI mode),
and other compatible backends.

Requires the optional ``openai`` package::

    uv pip install openai
"""

from collections.abc import AsyncIterator, Iterator
from typing import Any

from agents7.llm.base import LLMProvider
from agents7.llm.streaming import AsyncStreamResult, StreamResult

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

    Attributes:
        model_name (str): Model identifier (e.g.
            ``"gpt-4o"``, ``"deepseek-r1"``).
        base_url (str | None): Custom API base URL.
            ``None`` uses the default OpenAI endpoint.
        api_key (str | None): API key. Falls back to the
            ``OPENAI_API_KEY`` environment variable when
            ``None``.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        _require_openai()
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
    ) -> dict:
        """Build kwargs for the OpenAI chat completions call."""
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

    @staticmethod
    def _parse_response(response) -> dict:
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
            "metrics": {
                "tokens": usage.completion_tokens or 0,
                "prompt_tokens": usage.prompt_tokens or 0,
                "total_duration": 0.0,
            },
        }

    # --- Chat (sync / async) ---

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """See `LLMProvider.chat` for documentation."""
        params = self._build_params(
            self.model_name,
            messages,
            temperature,
            max_tokens,
            output_format,
            seed,
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
    ) -> dict:
        """See `LLMProvider.chat` for documentation."""
        params = self._build_params(
            self.model_name,
            messages,
            temperature,
            max_tokens,
            output_format,
            seed,
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
