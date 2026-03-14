"""Unit tests for Pipeline.act_stream and act_stream_async."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from agents7.agents.labeler import Labeler
from agents7.agents.pipeline import Pipeline, PipelineStep
from agents7.agents.streaming import (
    AgentAsyncStreamResult,
    AgentStreamResult,
)
from agents7.llm import LLMEngine

from helpers import DUMMY_METRICS, ENGINE, async_iter, chat_chunk


# --- Helper models ---


class StepOneOutput(BaseModel):
    name: str


class StepTwoOutput(BaseModel):
    greeting: str


def _make_labeler(output_model=StepOneOutput):
    return Labeler(
        name="test",
        version=1,
        engine=ENGINE,
        system_prompt="Test.",
        output_model=output_model,
    )


def _step_one_mapper(response):
    return {"text": f"Hello, {response.model.name}!"}


def _chat_response(data):
    return {
        "role": "assistant",
        "response": json.dumps(data),
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }


def _stream_chunks(text):
    """Build mock stream chunks for a complete response."""
    return [
        chat_chunk(content=text, role="assistant"),
        chat_chunk(
            done=True,
            eval_count=10,
            prompt_eval_count=5,
            total_duration=100_000_000,
        ),
    ]


# --- act_stream: success ---


@patch.object(LLMEngine, "chat_stream")
@patch.object(LLMEngine, "chat")
def test_pipeline_stream_two_steps(mock_chat, mock_stream):
    """Streams the final step after running intermediate."""
    mock_chat.return_value = _chat_response({"name": "Alice"})

    from agents7.llm.streaming import StreamResult

    chunks = _stream_chunks('{"greeting": "Hi Alice"}')
    mock_stream.return_value = StreamResult(iter(chunks), "chat")

    pipeline = Pipeline(
        steps=[
            PipelineStep(
                agent=_make_labeler(StepOneOutput),
                mapper=_step_one_mapper,
            ),
            PipelineStep(agent=_make_labeler(StepTwoOutput)),
        ]
    )

    stream = pipeline.act_stream({"text": "test"})
    collected = list(stream)
    assert len(collected) > 0
    assert stream.agent_response is not None
    assert stream.agent_response.model is not None


# --- act_stream: single step ---


@patch.object(LLMEngine, "chat_stream")
def test_pipeline_stream_single_step(mock_stream):
    """Single-step pipeline streams directly."""
    from agents7.llm.streaming import StreamResult

    chunks = _stream_chunks('{"name": "Bob"}')
    mock_stream.return_value = StreamResult(iter(chunks), "chat")

    pipeline = Pipeline(
        steps=[PipelineStep(agent=_make_labeler(StepOneOutput))]
    )

    stream = pipeline.act_stream({"text": "test"})
    collected = list(stream)
    assert len(collected) > 0
    assert stream.agent_response.model is not None


# --- act_stream: intermediate failure ---


@patch.object(LLMEngine, "chat")
def test_pipeline_stream_intermediate_failure(mock_chat):
    """Returns failed stream when intermediate step fails."""
    mock_chat.return_value = {
        "role": "assistant",
        "response": "bad json",
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }

    pipeline = Pipeline(
        steps=[
            PipelineStep(
                agent=_make_labeler(StepOneOutput),
                mapper=_step_one_mapper,
            ),
            PipelineStep(agent=_make_labeler(StepTwoOutput)),
        ]
    )

    stream = pipeline.act_stream({"text": "test"})
    collected = list(stream)
    assert collected == []
    assert stream.agent_response is not None
    assert stream.agent_response.model is None


# --- act_stream: mapper failure ---


@patch.object(LLMEngine, "chat")
def test_pipeline_stream_mapper_failure(mock_chat):
    """Returns failed stream when mapper raises."""
    mock_chat.return_value = _chat_response({"name": "Alice"})

    def bad_mapper(resp):
        raise RuntimeError("mapper broke")

    pipeline = Pipeline(
        steps=[
            PipelineStep(
                agent=_make_labeler(StepOneOutput),
                mapper=bad_mapper,
            ),
            PipelineStep(agent=_make_labeler(StepTwoOutput)),
        ]
    )

    stream = pipeline.act_stream({"text": "test"})
    collected = list(stream)
    assert collected == []
    assert stream.agent_response.model is not None


# --- act_stream: condition skips last step ---


@patch.object(LLMEngine, "chat")
def test_pipeline_stream_condition_skips_last(mock_chat):
    """Returns failed stream when last step condition fails."""
    mock_chat.return_value = _chat_response({"name": "Alice"})

    pipeline = Pipeline(
        steps=[
            PipelineStep(
                agent=_make_labeler(StepOneOutput),
                mapper=_step_one_mapper,
            ),
            PipelineStep(
                agent=_make_labeler(StepTwoOutput),
                condition=lambda d: False,
            ),
        ]
    )

    stream = pipeline.act_stream({"text": "test"})
    collected = list(stream)
    assert collected == []
    assert stream.agent_response is not None
    assert stream.agent_response.model is None


# --- act_stream_async: success ---


@patch.object(LLMEngine, "chat_stream_async", new_callable=AsyncMock)
@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_pipeline_stream_async_two_steps(mock_chat, mock_stream):
    """Async streams the final step after intermediate."""
    mock_chat.return_value = _chat_response({"name": "Alice"})

    from agents7.llm.streaming import AsyncStreamResult

    chunks = _stream_chunks('{"greeting": "Hi Alice"}')
    mock_stream.return_value = AsyncStreamResult(async_iter(chunks), "chat")

    pipeline = Pipeline(
        steps=[
            PipelineStep(
                agent=_make_labeler(StepOneOutput),
                mapper=_step_one_mapper,
            ),
            PipelineStep(agent=_make_labeler(StepTwoOutput)),
        ]
    )

    stream = await pipeline.act_stream_async({"text": "test"})
    collected = [chunk async for chunk in stream]
    assert len(collected) > 0
    assert stream.agent_response is not None


# --- act_stream_async: intermediate failure ---


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_pipeline_stream_async_failure(mock_chat):
    """Async returns failed stream on intermediate failure."""
    mock_chat.return_value = {
        "role": "assistant",
        "response": "bad",
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }

    pipeline = Pipeline(
        steps=[
            PipelineStep(
                agent=_make_labeler(StepOneOutput),
                mapper=_step_one_mapper,
            ),
            PipelineStep(agent=_make_labeler(StepTwoOutput)),
        ]
    )

    stream = await pipeline.act_stream_async({"text": "test"})
    collected = [chunk async for chunk in stream]
    assert collected == []
    assert stream.agent_response.model is None


# --- from_failed ---


def test_agent_stream_result_from_failed():
    """AgentStreamResult.from_failed yields nothing."""
    stream = AgentStreamResult.from_failed(None)
    chunks = list(stream)
    assert chunks == []
    assert stream.agent_response is not None
    assert stream.agent_response.model is None


@pytest.mark.asyncio
async def test_agent_async_stream_result_from_failed():
    """AgentAsyncStreamResult.from_failed yields nothing."""
    stream = AgentAsyncStreamResult.from_failed(None)
    chunks = [c async for c in stream]
    assert chunks == []
    assert stream.agent_response is not None
    assert stream.agent_response.model is None
