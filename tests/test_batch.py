"""Unit tests for batch flattening logic (no LLM required)."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agents7.llm import LLMEngine

from helpers import batch_chat_response, make_labeler


# --- Sync: act_in_batch ---


@patch.object(LLMEngine, "chat")
def test_single_chunk_flattening(mock_chat):
    """3 items within one chunk → 1 LLM call, 3 responses."""
    mock_chat.return_value = batch_chat_response(
        [{"value": "a"}, {"value": "b"}, {"value": "c"}]
    )
    agent = make_labeler()
    results = agent.act_in_batch(
        [{"x": 1}, {"x": 2}, {"x": 3}],
        requests_per_prompt=10,
        iter_func=None,
    )

    assert mock_chat.call_count == 1
    assert len(results) == 3
    assert [r.idx for r in results] == [0, 1, 2]
    assert [r.model.value for r in results] == ["a", "b", "c"]


@patch.object(LLMEngine, "chat")
def test_multi_chunk_flattening(mock_chat):
    """5 items, requests_per_prompt=2 → 3 LLM calls, idx 0..4."""
    mock_chat.side_effect = [
        batch_chat_response([{"value": "a"}, {"value": "b"}]),
        batch_chat_response([{"value": "c"}, {"value": "d"}]),
        batch_chat_response([{"value": "e"}]),
    ]
    agent = make_labeler()
    results = agent.act_in_batch(
        [{"x": i} for i in range(5)],
        requests_per_prompt=2,
        iter_func=None,
    )

    assert mock_chat.call_count == 3
    assert len(results) == 5
    assert [r.idx for r in results] == [0, 1, 2, 3, 4]
    assert [r.model.value for r in results] == [
        "a",
        "b",
        "c",
        "d",
        "e",
    ]


@patch.object(LLMEngine, "chat")
def test_shared_metadata_across_chunk(mock_chat):
    """All items in a chunk share response, thinking, metrics."""
    mock_chat.return_value = batch_chat_response(
        [{"value": "a"}, {"value": "b"}],
        thinking="some thinking",
    )
    agent = make_labeler()
    results = agent.act_in_batch(
        [{"x": 1}, {"x": 2}],
        requests_per_prompt=10,
        iter_func=None,
    )

    assert results[0].response == results[1].response
    assert results[0].thinking == results[1].thinking
    assert results[0].metrics == results[1].metrics


@patch.object(LLMEngine, "chat")
def test_empty_input_list(mock_chat):
    """Empty input list → empty output, no LLM calls."""
    agent = make_labeler()
    results = agent.act_in_batch(
        [],
        requests_per_prompt=10,
        iter_func=None,
    )

    assert results == []
    assert mock_chat.call_count == 0


# --- Async: act_in_batch_async ---


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_async_single_chunk_flattening(mock_chat):
    """3 items within one chunk → 1 LLM call, 3 responses."""
    mock_chat.return_value = batch_chat_response(
        [{"value": "a"}, {"value": "b"}, {"value": "c"}]
    )
    agent = make_labeler()
    results = await agent.act_in_batch_async(
        [{"x": 1}, {"x": 2}, {"x": 3}],
        requests_per_prompt=10,
        with_tqdm=False,
    )

    assert mock_chat.call_count == 1
    assert len(results) == 3
    assert [r.idx for r in results] == [0, 1, 2]
    assert [r.model.value for r in results] == ["a", "b", "c"]


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_async_multi_chunk_preserves_order(mock_chat):
    """5 items, requests_per_prompt=2 → correct order."""
    # Map chunk content to its response (order-independent)
    chunk_map = {
        "{'x': 0}": ([{"value": "a"}, {"value": "b"}], 0.06),
        "{'x': 2}": ([{"value": "c"}, {"value": "d"}], 0.04),
        "{'x': 4}": ([{"value": "e"}], 0.02),
    }

    async def _side_effect(**kwargs):
        content = kwargs["messages"][1]["content"]
        for marker, (items, delay) in chunk_map.items():
            if marker in content:
                await asyncio.sleep(delay)
                return batch_chat_response(items)

    mock_chat.side_effect = _side_effect

    agent = make_labeler()
    results = await agent.act_in_batch_async(
        [{"x": i} for i in range(5)],
        requests_per_prompt=2,
        with_tqdm=False,
    )

    assert len(results) == 5
    assert [r.idx for r in results] == [0, 1, 2, 3, 4]
    assert [r.model.value for r in results] == [
        "a",
        "b",
        "c",
        "d",
        "e",
    ]


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_async_shared_metadata_across_chunk(mock_chat):
    """All items in a chunk share response, thinking, metrics."""
    mock_chat.return_value = batch_chat_response(
        [{"value": "a"}, {"value": "b"}],
        thinking="some thinking",
    )
    agent = make_labeler()
    results = await agent.act_in_batch_async(
        [{"x": 1}, {"x": 2}],
        requests_per_prompt=10,
        with_tqdm=False,
    )

    assert results[0].response == results[1].response
    assert results[0].thinking == results[1].thinking
    assert results[0].metrics == results[1].metrics


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_async_empty_input_list(mock_chat):
    """Empty input list → empty output, no LLM calls."""
    agent = make_labeler()
    results = await agent.act_in_batch_async(
        [],
        requests_per_prompt=10,
        with_tqdm=False,
    )

    assert results == []
    assert mock_chat.call_count == 0
