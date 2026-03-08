"""Agent-level streaming wrappers.

`AgentStreamResult` and `AgentAsyncStreamResult` wrap the
engine-level stream results, yielding text chunks during
iteration. After iteration completes, the full
`AgentResponse` is available via the `agent_response`
property.
"""

from __future__ import annotations

from typing import AsyncIterator, Iterator

from agents7.agents.parsing import ResponseParser
from agents7.exceptions import ParseError
from agents7.llm.streaming import AsyncStreamResult, StreamResult
from agents7.models.base import AgentResponse, LLMMetrics


class AgentStreamResult:
    """Sync streaming wrapper that yields text chunks.

    After iteration, access the parsed result via
    `agent_response`.

    Usage::

        stream = agent.act_stream({"text": "..."})
        for chunk in stream:
            print(chunk, end="", flush=True)
        print(stream.agent_response.model)
    """

    __slots__ = ("_stream", "_parser", "_agent_response")

    def __init__(
        self,
        stream: StreamResult,
        parser: ResponseParser,
    ) -> None:
        self._stream = stream
        self._parser = parser
        self._agent_response: AgentResponse | None = None

    def __iter__(self) -> Iterator[str]:
        for chunk in self._stream:
            yield chunk
        self._finalize()

    @property
    def agent_response(self) -> AgentResponse | None:
        """Parsed `AgentResponse`, available after iteration."""
        return self._agent_response

    def _finalize(self) -> None:
        """Parse the accumulated response into an
        `AgentResponse`."""
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


class AgentAsyncStreamResult:
    """Async streaming wrapper that yields text chunks.

    After iteration, access the parsed result via
    `agent_response`.

    Usage::

        stream = await agent.act_stream_async({"text": "..."})
        async for chunk in stream:
            print(chunk, end="", flush=True)
        print(stream.agent_response.model)
    """

    __slots__ = ("_stream", "_parser", "_agent_response")

    def __init__(
        self,
        stream: AsyncStreamResult,
        parser: ResponseParser,
    ) -> None:
        self._stream = stream
        self._parser = parser
        self._agent_response: AgentResponse | None = None

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        async for chunk in self._stream:
            yield chunk
        self._finalize()

    @property
    def agent_response(self) -> AgentResponse | None:
        """Parsed `AgentResponse`, available after iteration."""
        return self._agent_response

    def _finalize(self) -> None:
        """Parse the accumulated response into an
        `AgentResponse`."""
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
