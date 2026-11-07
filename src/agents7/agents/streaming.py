"""Agent-level streaming wrappers.

`AgentStreamResult` and `AgentAsyncStreamResult` wrap the
engine-level stream results, yielding text chunks during
iteration. After iteration completes, the full
`AgentResponse` is available via the `agent_response`
property.
"""

from typing import AsyncIterator, Callable, Iterator

from agents7.agents.parsing import ResponseParser
from agents7.exceptions import ParseError
from agents7.llm.streaming import AsyncStreamResult, StreamResult
from agents7.models.base import AgentResponse, LLMMetrics


class _AgentStreamBase:
    """Shared state and logic for agent stream wrappers."""

    __slots__ = (
        "_stream",
        "_parser",
        "_agent_response",
        "_on_complete",
    )

    def __init__(
        self,
        stream: StreamResult | AsyncStreamResult | None,
        parser: ResponseParser | None,
        on_complete: Callable[[AgentResponse], None] | None = None,
    ) -> None:
        self._stream = stream
        self._parser = parser
        self._agent_response: AgentResponse | None = None
        self._on_complete = on_complete

    @property
    def agent_response(self) -> AgentResponse | None:
        """Parsed `AgentResponse`, available after iteration."""
        return self._agent_response

    def _finalize(self) -> None:
        """Parse the accumulated response into an
        `AgentResponse`, then fire `on_complete`.

        `on_complete` is how a streamed call reaches the call
        log. It has to fire from here rather than from
        `act_stream`, because whether the call succeeded is only
        known once the caller has finished iterating and the
        accumulated text has been parsed. A caller that never
        iterates therefore records nothing — there is no
        outcome yet to record.
        """
        try:
            parsed = self._parser.parse(self._stream.response)
        except ParseError:
            parsed = None
        self._agent_response = AgentResponse(
            response=self._stream.response,
            model=self._parser.run_validators(parsed),
            thinking=self._stream.thinking,
            metrics=LLMMetrics(**self._stream.metrics),
        )
        if self._on_complete is not None:
            self._on_complete(self._agent_response)

    @classmethod
    def from_failed(
        cls, response: AgentResponse | None
    ) -> "_AgentStreamBase":
        """Create a pre-filled failed result.

        Used by `Pipeline.act_stream` / `act_stream_async`
        when an intermediate step fails.

        :param response: The failed step's response, or
            None.
        :return: An instance that yields nothing.
        """
        obj = cls.__new__(cls)
        obj._stream = None
        obj._parser = None
        # The step that failed logged its own call through
        # `act()`; emitting again here would double-count it.
        obj._on_complete = None
        failed = response or AgentResponse(
            response="",
            model=None,
            thinking=None,
            metrics=LLMMetrics(
                tokens=0,
                prompt_tokens=0,
                total_duration=0.0,
            ),
        )
        obj._agent_response = failed
        return obj


class AgentStreamResult(_AgentStreamBase):
    """Sync streaming wrapper that yields text chunks.

    After iteration, access the parsed result via
    `agent_response`.

    Usage::

        stream = agent.act_stream({"text": "..."})
        for chunk in stream:
            print(chunk, end="", flush=True)
        print(stream.agent_response.model)
    """

    __slots__ = ()

    def __init__(
        self,
        stream: StreamResult,
        parser: ResponseParser,
        on_complete: Callable[[AgentResponse], None] | None = None,
    ) -> None:
        super().__init__(stream, parser, on_complete)

    def __iter__(self) -> Iterator[str]:
        if self._stream is None:
            return iter(())
        for chunk in self._stream:
            yield chunk
        self._finalize()


class AgentAsyncStreamResult(_AgentStreamBase):
    """Async streaming wrapper that yields text chunks.

    After iteration, access the parsed result via
    `agent_response`.

    Usage::

        stream = await agent.act_stream_async({"text": "..."})
        async for chunk in stream:
            print(chunk, end="", flush=True)
        print(stream.agent_response.model)
    """

    __slots__ = ()

    def __init__(
        self,
        stream: AsyncStreamResult,
        parser: ResponseParser,
        on_complete: Callable[[AgentResponse], None] | None = None,
    ) -> None:
        super().__init__(stream, parser, on_complete)

    def __aiter__(self) -> AsyncIterator[str]:
        if self._stream is None:
            return self._empty()
        return self._iterate()

    @staticmethod
    async def _empty() -> AsyncIterator[str]:
        """Yield nothing (used for failed results)."""
        return
        yield  # noqa: RET504

    async def _iterate(self) -> AsyncIterator[str]:
        async for chunk in self._stream:
            yield chunk
        self._finalize()
