"""Unit tests for agent-level streaming (act_stream / act_stream_async)."""

from enum import StrEnum
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from agents7.agents.detector import Detector
from agents7.agents.streaming import AgentAsyncStreamResult, AgentStreamResult
from agents7.llm.streaming import AsyncStreamResult, StreamResult
from agents7.models.base import AgentResponse, LLMMetrics, Property

from helpers import ENGINE, make_labeler


# --- Fixtures ---


class DummyOutput(BaseModel):
    value: str


def _make_stream_labeler():
    """Create a labeler with DummyOutput model."""
    return make_labeler(output_model=DummyOutput)


def _make_chunk(content: str, *, done: bool = True) -> Mock:
    chunk = Mock()
    chunk.message = Mock(content=content, thinking="", role="assistant")
    chunk.done = done
    chunk.eval_count = 10
    chunk.prompt_eval_count = 5
    # Ollama chunks carry no cached count, and a bare `Mock` would
    # invent one rather than be absent — which is not what the
    # provider does.
    chunk.cached_prompt_tokens = None
    chunk.total_duration = 1_000_000_000
    # Same hazard as the line above: a real Ollama chunk names the
    # model as a string, and a bare `Mock` hands back a `Mock`.
    chunk.model = "qwen3"
    return chunk


def _make_failed_response() -> AgentResponse:
    return AgentResponse(
        response="error",
        model=None,
        thinking=None,
        metrics=LLMMetrics(tokens=0, prompt_tokens=0, total_duration=0.0),
    )


# --- Sync streaming ---


def test_act_stream_yields_chunks():
    """act_stream yields text chunks from engine stream."""
    agent = _make_stream_labeler()
    stream = StreamResult(iter([_make_chunk('{"value": "hello"}')]), "chat")

    with patch.object(agent.engine, "chat_stream", return_value=stream):
        result = agent.act_stream({"text": "test"})
        chunks = list(result)

    assert len(chunks) == 1
    assert chunks[0] == '{"value": "hello"}'
    assert result.agent_response is not None
    assert result.agent_response.model.value == "hello"
    assert result.agent_response.metrics.tokens == 10


def test_act_stream_invalid_json():
    """act_stream sets model=None when response is invalid."""
    agent = _make_stream_labeler()
    stream = StreamResult(iter([_make_chunk("not json")]), "chat")

    with patch.object(agent.engine, "chat_stream", return_value=stream):
        result = agent.act_stream({"text": "test"})
        list(result)

    assert result.agent_response is not None
    assert result.agent_response.model is None
    assert result.agent_response.response == "not json"


def test_agent_response_none_before_iteration():
    """agent_response is None before iterating."""
    agent = _make_stream_labeler()
    stream = StreamResult(iter([]), "chat")

    with patch.object(agent.engine, "chat_stream", return_value=stream):
        result = agent.act_stream({"text": "test"})

    assert result.agent_response is None


# --- AgentStreamResult.from_failed() ---


def test_sync_from_failed_with_response_yields_nothing():
    """from_failed() with a response yields no chunks."""
    result = AgentStreamResult.from_failed(_make_failed_response())
    assert list(result) == []


def test_sync_from_failed_agent_response_is_the_given_response():
    """from_failed() exposes the supplied response via agent_response."""
    failed = _make_failed_response()
    result = AgentStreamResult.from_failed(failed)
    list(result)
    assert result.agent_response is failed


def test_sync_from_failed_none_creates_empty_response():
    """from_failed(None) synthesises an empty zero-metric response."""
    result = AgentStreamResult.from_failed(None)
    assert list(result) == []
    assert result.agent_response is not None
    assert result.agent_response.response == ""
    assert result.agent_response.model is None
    assert result.agent_response.metrics.tokens == 0


# --- Async streaming ---


@pytest.mark.asyncio
async def test_act_stream_async_yields_chunks():
    """act_stream_async yields text chunks."""
    agent = _make_stream_labeler()
    chunk = _make_chunk('{"value": "world"}')

    async def _fake_iter():
        yield chunk

    stream = AsyncStreamResult(_fake_iter(), "chat")

    with patch.object(
        agent.engine,
        "chat_stream_async",
        new_callable=AsyncMock,
        return_value=stream,
    ):
        result = await agent.act_stream_async({"text": "test"})
        chunks = [c async for c in result]

    assert len(chunks) == 1
    assert chunks[0] == '{"value": "world"}'
    assert result.agent_response is not None
    assert result.agent_response.model.value == "world"


# --- AgentAsyncStreamResult.from_failed() ---


@pytest.mark.asyncio
async def test_async_from_failed_with_response_yields_nothing():
    """Async from_failed() with a response yields no chunks."""
    result = AgentAsyncStreamResult.from_failed(_make_failed_response())
    chunks = [c async for c in result]
    assert chunks == []


@pytest.mark.asyncio
async def test_async_from_failed_agent_response_is_given_response():
    """Async from_failed() exposes the supplied response."""
    failed = _make_failed_response()
    result = AgentAsyncStreamResult.from_failed(failed)
    [c async for c in result]
    assert result.agent_response is failed


@pytest.mark.asyncio
async def test_async_from_failed_none_creates_empty_response():
    """Async from_failed(None) synthesises an empty zero-metric response."""
    result = AgentAsyncStreamResult.from_failed(None)
    chunks = [c async for c in result]
    assert chunks == []
    assert result.agent_response is not None
    assert result.agent_response.response == ""
    assert result.agent_response.metrics.tokens == 0


# --- Detector streaming ---


class _TempReason(StrEnum):
    HOT = "It is hot."
    COLD = "It is cold."


class _Temperature(Property[float]):
    reason: _TempReason


def _make_detector() -> Detector:
    return Detector(
        name="temp-detector",
        version=1,
        engine=ENGINE,
        system_prompt="Detect temperature.",
        output_model=_Temperature,
    )


def test_detector_act_stream_yields_chunks():
    """Detector.act_stream yields chunks and parses a Property response."""
    agent = _make_detector()
    response_json = '{"value": 25.0, "reason": "It is hot."}'
    stream = StreamResult(iter([_make_chunk(response_json)]), "chat")

    with patch.object(agent.engine, "chat_stream", return_value=stream):
        result = agent.act_stream({"text": "warm day"})
        chunks = list(result)

    assert len(chunks) == 1
    assert result.agent_response is not None
    assert result.agent_response.model.value == 25.0


@pytest.mark.asyncio
async def test_detector_act_stream_async_yields_chunks():
    """Detector.act_stream_async yields chunks and parses a Property."""
    agent = _make_detector()
    response_json = '{"value": -5.0, "reason": "It is cold."}'
    chunk = _make_chunk(response_json)

    async def _fake_iter():
        yield chunk

    stream = AsyncStreamResult(_fake_iter(), "chat")

    with patch.object(
        agent.engine,
        "chat_stream_async",
        new_callable=AsyncMock,
        return_value=stream,
    ):
        result = await agent.act_stream_async({"text": "cold day"})
        chunks = [c async for c in result]

    assert len(chunks) == 1
    assert result.agent_response is not None
    assert result.agent_response.model.value == -5.0
