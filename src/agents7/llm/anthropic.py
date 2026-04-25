"""Native Anthropic/Claude LLM provider.

Uses the official ``anthropic`` package for direct access
to Claude models (Haiku, Sonnet, Opus).

Requires the optional ``anthropic`` package::

    uv pip install anthropic
"""

import time
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any

from agents7.llm.base import LLMProvider
from agents7.llm.streaming import AsyncStreamResult, StreamResult

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None


def _require_anthropic() -> None:
    if _anthropic is None:
        raise ImportError(
            "The `anthropic` package is required for the "
            "Anthropic provider. Install it with: "
            "uv pip install anthropic"
        )


_DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(LLMProvider):
    """LLM provider for Claude models via the Anthropic API.

    Attributes:
        model_name (str): Model identifier (e.g.
            ``"claude-sonnet-4-20250514"``).
        api_key (str | None): API key. Falls back to the
            ``ANTHROPIC_API_KEY`` environment variable.
        base_url (str | None): Custom API base URL.
        default_max_tokens (int): Default max_tokens when
            not specified per call. Defaults to 4096.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        default_max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        _require_anthropic()
        super().__init__("anthropic", model_name)
        self._default_max_tokens = default_max_tokens
        client_kwargs: dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self._client = _anthropic.Anthropic(**client_kwargs)
        self._async_client = _anthropic.AsyncAnthropic(**client_kwargs)

    # --- Helpers ---

    def _build_params(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        think: bool,
        stream: bool = False,
    ) -> dict:
        """Build kwargs for the Anthropic messages call."""
        system_text = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                chat_messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                    }
                )

        effective_max = (
            max_tokens if max_tokens > 0 else self._default_max_tokens
        )

        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": chat_messages,
            "max_tokens": effective_max,
            "stream": stream,
        }
        if system_text:
            params["system"] = system_text
        if temperature > 0:
            params["temperature"] = temperature
        if think:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": max(1024, effective_max // 2),
            }
        return params

    @staticmethod
    def _parse_response(
        response: Any,
        start_time: float,
    ) -> dict:
        """Parse a non-streaming Anthropic response."""
        content_text = ""
        thinking_text = ""
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "thinking":
                thinking_text += block.thinking

        usage = response.usage
        elapsed = time.monotonic() - start_time

        return {
            "response": content_text.strip(),
            "role": response.role,
            "thinking": (thinking_text.strip() if thinking_text else None),
            "metrics": {
                "tokens": usage.output_tokens,
                "prompt_tokens": usage.input_tokens,
                "total_duration": elapsed,
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
            messages,
            temperature,
            max_tokens,
            think,
        )
        start = time.monotonic()
        response = self._client.messages.create(**params)
        return self._parse_response(response, start)

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
            messages,
            temperature,
            max_tokens,
            think,
        )
        start = time.monotonic()
        response = await self._async_client.messages.create(**params)
        return self._parse_response(response, start)

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
            messages,
            temperature,
            max_tokens,
            think,
            stream=True,
        )
        raw_stream = self._client.messages.create(**params)
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
            messages,
            temperature,
            max_tokens,
            think,
            stream=True,
        )
        raw_stream = await self._async_client.messages.create(**params)
        adapted = _adapt_stream_async(raw_stream)
        return AsyncStreamResult(adapted, "chat")

    def get_model_info(self) -> dict:
        """Return provider and model metadata."""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "model_id": self.model_id,
        }


# --- Stream adapters ---
# Convert Anthropic streaming events into the namespace
# format that StreamResult / AsyncStreamResult expect
# (matching the Ollama chunk interface).


def _make_chunk(
    content: str = "",
    thinking: str = "",
    role: str = "assistant",
    done: bool = False,
    output_tokens: int | None = None,
    input_tokens: int | None = None,
) -> SimpleNamespace:
    """Build an Ollama-compatible chunk namespace."""
    msg = SimpleNamespace(
        content=content,
        thinking=thinking,
        role=role,
    )
    return SimpleNamespace(
        message=msg,
        done=done,
        eval_count=output_tokens,
        prompt_eval_count=input_tokens,
        total_duration=0 if done else None,
    )


def _adapt_stream_sync(raw_stream) -> Iterator:
    """Yield adapted chunks from an Anthropic stream."""
    input_tokens = 0
    output_tokens = 0

    for event in raw_stream:
        event_type = event.type

        if event_type == "message_start":
            usage = getattr(event.message, "usage", None)
            if usage:
                input_tokens = usage.input_tokens

        elif event_type == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                yield _make_chunk(content=delta.text)
            elif delta.type == "thinking_delta":
                yield _make_chunk(
                    thinking=delta.thinking,
                )

        elif event_type == "message_delta":
            usage = getattr(event, "usage", None)
            if usage:
                output_tokens = usage.output_tokens
            yield _make_chunk(
                done=True,
                output_tokens=output_tokens,
                input_tokens=input_tokens,
            )


async def _adapt_stream_async(raw_stream) -> AsyncIterator:
    """Yield adapted chunks from an async Anthropic stream."""
    input_tokens = 0
    output_tokens = 0

    async for event in raw_stream:
        event_type = event.type

        if event_type == "message_start":
            usage = getattr(event.message, "usage", None)
            if usage:
                input_tokens = usage.input_tokens

        elif event_type == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                yield _make_chunk(content=delta.text)
            elif delta.type == "thinking_delta":
                yield _make_chunk(
                    thinking=delta.thinking,
                )

        elif event_type == "message_delta":
            usage = getattr(event, "usage", None)
            if usage:
                output_tokens = usage.output_tokens
            yield _make_chunk(
                done=True,
                output_tokens=output_tokens,
                input_tokens=input_tokens,
            )
