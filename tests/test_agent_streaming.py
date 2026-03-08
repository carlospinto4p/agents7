"""Unit tests for agent-level streaming (act_stream / act_stream_async)."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from agents7.llm.streaming import AsyncStreamResult, StreamResult

from helpers import make_labeler


# --- Fixtures ---


class DummyOutput(BaseModel):
    value: str


def _make_stream_labeler():
    """Create a labeler with DummyOutput model."""
    return make_labeler(output_model=DummyOutput)


# --- Sync streaming ---


def test_act_stream_yields_chunks():
    """act_stream yields text chunks from engine stream."""
    agent = _make_stream_labeler()
    mock_chunk = Mock()
    mock_chunk.message = Mock(
        content='{"value": "hello"}',
        thinking="",
        role="assistant",
    )
    mock_chunk.done = True
    mock_chunk.eval_count = 10
    mock_chunk.prompt_eval_count = 5
    mock_chunk.total_duration = 1_000_000_000

    stream = StreamResult(iter([mock_chunk]), "chat")

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
    mock_chunk = Mock()
    mock_chunk.message = Mock(
        content="not json",
        thinking="",
        role="assistant",
    )
    mock_chunk.done = True
    mock_chunk.eval_count = 5
    mock_chunk.prompt_eval_count = 3
    mock_chunk.total_duration = 500_000_000

    stream = StreamResult(iter([mock_chunk]), "chat")

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


# --- Async streaming ---


@pytest.mark.asyncio
async def test_act_stream_async_yields_chunks():
    """act_stream_async yields text chunks."""
    agent = _make_stream_labeler()
    mock_chunk = Mock()
    mock_chunk.message = Mock(
        content='{"value": "world"}',
        thinking="",
        role="assistant",
    )
    mock_chunk.done = True
    mock_chunk.eval_count = 8
    mock_chunk.prompt_eval_count = 4
    mock_chunk.total_duration = 2_000_000_000

    async def _fake_iter():
        yield mock_chunk

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
