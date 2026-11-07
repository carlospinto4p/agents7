"""Streaming wrappers for LLM responses.

StreamResult and AsyncStreamResult yield content chunks as strings
when iterated. After iteration completes, the full response, thinking,
role, and metrics are available as attributes.
"""

import time
from typing import Any, AsyncIterator, Callable, Iterator


class _StreamBase:
    """Shared state and helpers for stream result classes."""

    def _init_state(
        self,
        iterator,
        extractor: str,
        _on_complete: Callable[[dict], None] | None,
        provider: str | None = None,
    ) -> None:
        self._iterator = iterator
        self._extractor = extractor
        # Prefixes the model name the final chunk reports, so a
        # streamed call names its model the same way a plain one
        # does. `None` where the caller did not say, and the name
        # is then reported unqualified rather than mislabelled.
        self._provider = provider
        self._response_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._role: str | None = None
        self._metrics: dict = {}
        self._done = False
        self._on_complete = _on_complete
        # Runs on the finished metrics dict before anything reads or
        # caches it, so a streamed call can be priced the way a plain
        # one is. Set by the dispatcher; `None` leaves metrics as the
        # provider reported them.
        self._on_metrics: Callable[[dict], None] | None = None
        self._response_cache: str = ""
        self._thinking_cache: str | None = None
        # A streamed call is timed here because only Ollama reports a
        # duration of its own; the OpenAI-compatible providers report
        # none and used to leave a hard 0 in the metrics.
        self._started = time.monotonic()

    @property
    def response(self) -> str:
        """Full accumulated response text."""
        if self._done:
            return self._response_cache
        return "".join(self._response_parts)

    @property
    def thinking(self) -> str | None:
        """Accumulated thinking text, or `None` if empty."""
        if self._done:
            return self._thinking_cache
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

    def _extract(self, chunk: Any) -> tuple[str, str]:
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

    def _read_final(self, chunk: Any) -> None:
        """Read metrics from the final chunk.

        Carries the same fields a non-streamed call reports, so a
        caller does not have to know which path produced them.
        `cost_cents` is added afterwards by the dispatcher, once
        these token counts exist to price.
        """
        if getattr(chunk, "done", False):
            eval_count = getattr(chunk, "eval_count", None)
            prompt_eval_count = getattr(chunk, "prompt_eval_count", None)
            cached = getattr(chunk, "cached_prompt_tokens", None)
            total_duration = getattr(chunk, "total_duration", None)
            self._metrics = {
                "tokens": eval_count or 0,
                "prompt_tokens": prompt_eval_count or 0,
                "cached_prompt_tokens": cached or 0,
                # Ollama reports server-side nanoseconds; everything
                # else reports nothing, so fall back to the wall clock
                # measured around the stream rather than to a zero.
                "total_duration": (
                    total_duration / 10**9
                    if total_duration
                    else time.monotonic() - self._started
                ),
                # Every SDK echoes the model on its chunks, so a
                # streamed call can name what answered it just as
                # a plain one does. Left unqualified by a provider
                # prefix only when the chunk names nothing.
                "model": self._model_from(chunk),
            }

    def _model_from(self, chunk: Any) -> str | None:
        """Name the model a stream's final chunk reports.

        :param chunk: The chunk that carried ``done``.
        :return: ``<provider>::<model>``, or ``None``.
        """
        model = getattr(chunk, "model", None)
        if not model:
            return None
        return f"{self._provider}::{model}" if self._provider else model

    def _as_dict(self) -> dict:
        """Serialize accumulated state for caching."""
        return {
            "response": self.response,
            "thinking": self.thinking,
            "role": self._role,
            "metrics": dict(self._metrics),
        }

    def _process_chunk(self, chunk: Any) -> str | None:
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
        self._response_cache = "".join(self._response_parts)
        thinking = "".join(self._thinking_parts)
        self._thinking_cache = thinking if thinking else None
        self._done = True
        # Before `_as_dict`, so a cached stream stores the same
        # metrics a live one reports.
        if self._on_metrics is not None and self._metrics:
            self._on_metrics(self._metrics)
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
        obj._response_cache = cached["response"]
        obj._thinking_cache = thinking
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
        provider: str | None = None,
    ):
        self._init_state(iterator, extractor, _on_complete, provider)

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
    def from_cached(cls, cached: dict) -> "StreamResult":
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
        provider: str | None = None,
    ):
        self._init_state(iterator, extractor, _on_complete, provider)

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
    def from_cached(cls, cached: dict) -> "AsyncStreamResult":
        """Create a pre-filled instance from cached data."""
        return cls._restore_from_cached(cached)
