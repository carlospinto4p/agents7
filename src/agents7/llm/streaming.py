"""Streaming wrappers for LLM responses.

StreamResult and AsyncStreamResult yield content chunks as strings
when iterated. After iteration completes, the full response, thinking,
role, and metrics are available as attributes.
"""

from __future__ import annotations

from typing import AsyncIterator, Callable, Iterator


class _StreamBase:
    """Shared state and helpers for stream result classes."""

    def _init_state(
        self,
        iterator,
        extractor: str,
        _on_complete: Callable[[dict], None] | None,
    ) -> None:
        self._iterator = iterator
        self._extractor = extractor
        self._response_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._role: str | None = None
        self._metrics: dict = {}
        self._done = False
        self._on_complete = _on_complete

    @property
    def response(self) -> str:
        """Full accumulated response text."""
        return "".join(self._response_parts)

    @property
    def thinking(self) -> str | None:
        """Accumulated thinking text, or `None` if empty."""
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

    def _as_dict(self) -> dict:
        """Serialize accumulated state for caching."""
        return {
            "response": self.response,
            "thinking": self.thinking,
            "role": self._role,
            "metrics": dict(self._metrics),
        }

    def _process_chunk(self, chunk) -> str | None:
        """Process a single chunk, returning content or
        `None`."""
        content, thinking = self._extract(chunk)

        if thinking:
            self._thinking_parts.append(thinking)

        if content:
            self._response_parts.append(content)

        self._read_final(chunk)
        return content or None

    def _finish(self) -> None:
        """Mark stream as done and fire completion
        callback."""
        self._done = True
        if self._on_complete is not None:
            self._on_complete(self._as_dict())

    @classmethod
    def _restore_from_cached(cls, cached: dict):
        """Populate a new instance from cached data."""
        obj = cls.__new__(cls)
        obj._iterator = None
        obj._extractor = ""
        obj._response_parts = [cached["response"]]
        thinking = cached.get("thinking")
        obj._thinking_parts = [thinking] if thinking else []
        obj._role = cached.get("role")
        obj._metrics = cached.get("metrics", {})
        obj._done = True
        obj._on_complete = None
        return obj


class StreamResult(_StreamBase):
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
        _on_complete: (Callable[[dict], None] | None) = None,
    ):
        self._init_state(iterator, extractor, _on_complete)

    def __iter__(self) -> Iterator[str]:
        if self._done:
            return iter(self._response_parts)
        return self._iterate()

    def _iterate(self) -> Iterator[str]:
        for chunk in self._iterator:
            content = self._process_chunk(chunk)
            if content:
                yield content
        self._finish()

    @classmethod
    def from_cached(cls, cached: dict) -> StreamResult:
        """Create a pre-filled instance from cached data."""
        obj = cls._restore_from_cached(cached)
        obj._iterator = iter(())
        return obj


class AsyncStreamResult(_StreamBase):
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
        _on_complete: (Callable[[dict], None] | None) = None,
    ):
        self._init_state(iterator, extractor, _on_complete)

    def __aiter__(self) -> AsyncIterator[str]:
        if self._done:
            return self._replay()
        return self._iterate()

    async def _replay(self) -> AsyncIterator[str]:
        """Yield cached response parts."""
        for part in self._response_parts:
            yield part

    async def _iterate(self) -> AsyncIterator[str]:
        async for chunk in self._iterator:
            content = self._process_chunk(chunk)
            if content:
                yield content
        self._finish()

    @classmethod
    def from_cached(cls, cached: dict) -> AsyncStreamResult:
        """Create a pre-filled instance from cached data."""
        return cls._restore_from_cached(cached)
