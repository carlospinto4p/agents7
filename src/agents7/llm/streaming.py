"""Streaming wrappers for LLM responses.

StreamResult and AsyncStreamResult yield content chunks as strings
when iterated. After iteration completes, the full response, thinking,
role, and metrics are available as attributes.
"""

from __future__ import annotations

from typing import AsyncIterator, Iterator


class StreamResult:
    """Wraps a sync iterator of ollama streaming chunks.

    Usage::

        stream = engine.chat_stream(messages=[...])
        for chunk in stream:
            print(chunk, end="", flush=True)
        print(stream.response)
        print(stream.metrics)
    """

    def __init__(
        self,
        iterator: Iterator,
        extractor: str,
    ):
        self._iterator = iterator
        self._extractor = extractor
        self._response_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._role: str | None = None
        self._metrics: dict = {}
        self._done = False

    @property
    def response(self) -> str:
        """Full accumulated response text."""
        return "".join(self._response_parts)

    @property
    def thinking(self) -> str | None:
        """Accumulated thinking text, or None if empty."""
        text = "".join(self._thinking_parts)
        return text if text else None

    @property
    def role(self) -> str | None:
        """Message role (chat only)."""
        return self._role

    @property
    def metrics(self) -> dict:
        """Token counts and duration from the final chunk."""
        return self._metrics

    def __iter__(self) -> Iterator[str]:
        return self._iterate()

    def _iterate(self) -> Iterator[str]:
        for chunk in self._iterator:
            content, thinking = self._extract(chunk)

            if thinking:
                self._thinking_parts.append(thinking)

            if content:
                self._response_parts.append(content)
                yield content

            self._read_final(chunk)

        self._done = True

    def _extract(self, chunk) -> tuple[str, str]:
        """Extract content and thinking from a chunk."""
        content = ""
        thinking = ""

        if self._extractor == "chat":
            msg = getattr(chunk, "message", None)
            if msg is not None:
                content = getattr(msg, "content", "") or ""
                thinking = getattr(msg, "thinking", "") or ""
                if self._role is None:
                    role = getattr(msg, "role", None)
                    if role:
                        self._role = role
        else:
            content = getattr(chunk, "response", "") or ""
            thinking = getattr(chunk, "thinking", "") or ""

        return content, thinking

    def _read_final(self, chunk) -> None:
        """Read metrics from the final chunk."""
        if getattr(chunk, "done", False):
            eval_count = getattr(chunk, "eval_count", None)
            prompt_eval_count = getattr(chunk, "prompt_eval_count", None)
            total_duration = getattr(chunk, "total_duration", None)
            self._metrics = {
                "tokens": eval_count or 0,
                "prompt_tokens": prompt_eval_count or 0,
                "total_duration": (
                    total_duration / 10**9 if total_duration else 0
                ),
            }


class AsyncStreamResult:
    """Wraps an async iterator of ollama streaming chunks.

    Usage::

        stream = engine.chat_stream_async(messages=[...])
        async for chunk in stream:
            print(chunk, end="", flush=True)
        print(stream.response)
        print(stream.metrics)
    """

    def __init__(
        self,
        iterator: AsyncIterator,
        extractor: str,
    ):
        self._iterator = iterator
        self._extractor = extractor
        self._response_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._role: str | None = None
        self._metrics: dict = {}
        self._done = False

    @property
    def response(self) -> str:
        """Full accumulated response text."""
        return "".join(self._response_parts)

    @property
    def thinking(self) -> str | None:
        """Accumulated thinking text, or None if empty."""
        text = "".join(self._thinking_parts)
        return text if text else None

    @property
    def role(self) -> str | None:
        """Message role (chat only)."""
        return self._role

    @property
    def metrics(self) -> dict:
        """Token counts and duration from the final chunk."""
        return self._metrics

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        async for chunk in self._iterator:
            content, thinking = self._extract(chunk)

            if thinking:
                self._thinking_parts.append(thinking)

            if content:
                self._response_parts.append(content)
                yield content

            self._read_final(chunk)

        self._done = True

    def _extract(self, chunk) -> tuple[str, str]:
        """Extract content and thinking from a chunk."""
        content = ""
        thinking = ""

        if self._extractor == "chat":
            msg = getattr(chunk, "message", None)
            if msg is not None:
                content = getattr(msg, "content", "") or ""
                thinking = getattr(msg, "thinking", "") or ""
                if self._role is None:
                    role = getattr(msg, "role", None)
                    if role:
                        self._role = role
        else:
            content = getattr(chunk, "response", "") or ""
            thinking = getattr(chunk, "thinking", "") or ""

        return content, thinking

    def _read_final(self, chunk) -> None:
        """Read metrics from the final chunk."""
        if getattr(chunk, "done", False):
            eval_count = getattr(chunk, "eval_count", None)
            prompt_eval_count = getattr(chunk, "prompt_eval_count", None)
            total_duration = getattr(chunk, "total_duration", None)
            self._metrics = {
                "tokens": eval_count or 0,
                "prompt_tokens": prompt_eval_count or 0,
                "total_duration": (
                    total_duration / 10**9 if total_duration else 0
                ),
            }
