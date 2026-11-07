"""Unit tests for LLM streaming support (no LLM required)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agents7.llm.streaming import AsyncStreamResult, StreamResult

from helpers import async_iter, chat_chunk, gen_chunk, make_engine

_BARE_CHUNK = SimpleNamespace()  # chunk with no attributes at all


# --- StreamResult (sync) ---


def test_yields_chat_content_chunks():
    """Chat chunks yield content strings."""
    chunks = [
        chat_chunk(content="Hello"),
        chat_chunk(content=" world"),
        chat_chunk(
            done=True,
            eval_count=10,
            prompt_eval_count=5,
            total_duration=2_000_000_000,
        ),
    ]
    stream = StreamResult(iter(chunks), "chat")
    collected = list(stream)

    assert collected == ["Hello", " world"]
    assert stream.response == "Hello world"


def test_yields_generate_content_chunks():
    """Generate chunks yield response strings."""
    chunks = [
        gen_chunk(response="foo"),
        gen_chunk(response="bar"),
        gen_chunk(
            done=True,
            eval_count=8,
            prompt_eval_count=3,
            total_duration=1_000_000_000,
        ),
    ]
    stream = StreamResult(iter(chunks), "generate")
    collected = list(stream)

    assert collected == ["foo", "bar"]
    assert stream.response == "foobar"


def test_metrics_from_final_chunk():
    """Metrics are populated from the done=True chunk."""
    chunks = [
        chat_chunk(content="x"),
        chat_chunk(
            done=True,
            eval_count=42,
            prompt_eval_count=10,
            total_duration=3_500_000_000,
        ),
    ]
    stream = StreamResult(iter(chunks), "chat")
    list(stream)

    assert stream.metrics["tokens"] == 42
    assert stream.metrics["prompt_tokens"] == 10
    assert stream.metrics["total_duration"] == 3.5


def test_thinking_accumulated_not_yielded():
    """Thinking text is accumulated but not yielded."""
    chunks = [
        chat_chunk(thinking="Let me think..."),
        chat_chunk(content="Answer"),
        chat_chunk(
            done=True,
            eval_count=5,
            prompt_eval_count=2,
            total_duration=1_000_000_000,
        ),
    ]
    stream = StreamResult(iter(chunks), "chat")
    collected = list(stream)

    assert collected == ["Answer"]
    assert stream.thinking == "Let me think..."


def test_empty_stream():
    """Empty iterator produces empty response."""
    stream = StreamResult(iter([]), "chat")
    collected = list(stream)

    assert collected == []
    assert stream.response == ""
    assert stream.thinking is None
    assert stream.metrics == {}


def test_role_captured_for_chat():
    """Role is captured from the first chat chunk."""
    chunks = [
        chat_chunk(content="Hi", role="assistant"),
        chat_chunk(
            done=True,
            eval_count=1,
            prompt_eval_count=1,
            total_duration=1_000_000_000,
        ),
    ]
    stream = StreamResult(iter(chunks), "chat")
    list(stream)

    assert stream.role == "assistant"


def test_role_is_none_for_generate():
    """Generate mode has no role."""
    chunks = [
        gen_chunk(response="out"),
        gen_chunk(
            done=True,
            eval_count=1,
            prompt_eval_count=1,
            total_duration=1_000_000_000,
        ),
    ]
    stream = StreamResult(iter(chunks), "generate")
    list(stream)

    assert stream.role is None


# --- AsyncStreamResult ---


@pytest.mark.asyncio
async def test_yieldschat_chunks_async():
    """Async chat chunks yield content strings."""
    chunks = [
        chat_chunk(content="Hello"),
        chat_chunk(content=" async"),
        chat_chunk(
            done=True,
            eval_count=6,
            prompt_eval_count=4,
            total_duration=2_000_000_000,
        ),
    ]
    stream = AsyncStreamResult(async_iter(chunks), "chat")
    collected = []
    async for text in stream:
        collected.append(text)

    assert collected == ["Hello", " async"]
    assert stream.response == "Hello async"


@pytest.mark.asyncio
async def test_metrics_from_final_chunk_async():
    """Async metrics are populated from done chunk."""
    chunks = [
        chat_chunk(content="x"),
        chat_chunk(
            done=True,
            eval_count=20,
            prompt_eval_count=8,
            total_duration=4_000_000_000,
        ),
    ]
    stream = AsyncStreamResult(async_iter(chunks), "chat")
    async for _ in stream:
        pass

    assert stream.metrics["tokens"] == 20
    assert stream.metrics["prompt_tokens"] == 8
    assert stream.metrics["total_duration"] == 4.0


@pytest.mark.asyncio
async def test_thinking_accumulated_async():
    """Async thinking is accumulated but not yielded."""
    chunks = [
        chat_chunk(thinking="Hmm..."),
        chat_chunk(content="Done"),
        chat_chunk(
            done=True,
            eval_count=3,
            prompt_eval_count=1,
            total_duration=1_000_000_000,
        ),
    ]
    stream = AsyncStreamResult(async_iter(chunks), "chat")
    collected = []
    async for text in stream:
        collected.append(text)

    assert collected == ["Done"]
    assert stream.thinking == "Hmm..."


# --- LLMEngine delegation ---


def test_chat_stream_delegates():
    """chat_stream delegates to provider.chat_stream."""
    engine = make_engine()
    sentinel = StreamResult(iter([]), "chat")
    engine.llm_provider.chat_stream.return_value = sentinel

    result = engine.chat_stream(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
    )

    assert result is sentinel
    engine.llm_provider.chat_stream.assert_called_once_with(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
        max_tokens=-1,
        output_format=None,
        think=False,
        seed=None,
    )


@pytest.mark.asyncio
async def test_chat_stream_async_delegates():
    """chat_stream_async delegates to provider."""
    engine = make_engine()
    sentinel = AsyncStreamResult(async_iter([]), "chat")
    engine.llm_provider.chat_stream_async = AsyncMock(return_value=sentinel)

    result = await engine.chat_stream_async(
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result is sentinel
    engine.llm_provider.chat_stream_async.assert_called_once()


def test_generate_stream_delegates():
    """generate_stream delegates to provider."""
    engine = make_engine()
    sentinel = StreamResult(iter([]), "generate")
    engine.llm_provider.generate_stream.return_value = sentinel

    result = engine.generate_stream(prompt="test")

    assert result is sentinel
    engine.llm_provider.generate_stream.assert_called_once_with(
        prompt="test",
        temperature=0.7,
        max_tokens=-1,
        output_format=None,
        think=False,
        seed=None,
    )


@pytest.mark.asyncio
async def test_generate_stream_async_delegates():
    """generate_stream_async delegates to provider."""
    engine = make_engine()
    sentinel = AsyncStreamResult(async_iter([]), "generate")
    engine.llm_provider.generate_stream_async = AsyncMock(
        return_value=sentinel
    )

    result = await engine.generate_stream_async(prompt="test")

    assert result is sentinel
    engine.llm_provider.generate_stream_async.assert_called_once()


@patch("agents7.llm.retry.retry_sync")
def test_streaming_bypasses_retry(mock_retry):
    """Streaming methods do NOT use retry wrappers."""
    engine = make_engine()
    sentinel = StreamResult(iter([]), "chat")
    engine.llm_provider.chat_stream.return_value = sentinel

    engine.chat_stream(
        messages=[{"role": "user", "content": "hi"}],
    )

    mock_retry.assert_not_called()


# --- _extract with missing chunk attributes ---


def test_extract_chat_no_message_attr():
    """Chat chunk with no message attribute yields empty strings."""
    stream = StreamResult(iter([_BARE_CHUNK]), "chat")
    collected = list(stream)
    assert collected == []
    assert stream.response == ""
    assert stream.thinking is None


def test_extract_chat_message_is_none():
    """Chat chunk with message=None yields empty strings."""
    chunk = SimpleNamespace(message=None, done=False)
    stream = StreamResult(iter([chunk]), "chat")
    collected = list(stream)
    assert collected == []
    assert stream.response == ""


def test_extract_generate_no_response_attr():
    """Generate chunk with no response attribute yields empty string."""
    stream = StreamResult(iter([_BARE_CHUNK]), "generate")
    collected = list(stream)
    assert collected == []
    assert stream.response == ""


def test_extract_chat_message_missing_content():
    """Chat chunk whose message has no content attribute yields empty."""
    chunk = SimpleNamespace(
        message=SimpleNamespace(role="assistant"), done=False
    )
    stream = StreamResult(iter([chunk]), "chat")
    collected = list(stream)
    assert collected == []
    assert stream.response == ""


# --- Metrics parity with a non-streamed call ---


def test_metrics_carry_cached_prompt_tokens():
    """A streamed call reports the cached count like a plain one.

    It reported nothing, so a streamed turn priced every prompt
    token at the fresh rate — about 20x the real figure for an
    agent that resends a long system prompt every turn.
    """
    chunks = [
        chat_chunk(content="x"),
        chat_chunk(
            done=True,
            eval_count=9,
            prompt_eval_count=6239,
            cached_prompt_tokens=6144,
            total_duration=1_000_000_000,
        ),
    ]
    stream = StreamResult(iter(chunks), "chat")
    list(stream)

    assert stream.metrics["cached_prompt_tokens"] == 6144


def test_metrics_default_cached_prompt_tokens_to_zero():
    """A provider that does not cache prompts reports none."""
    chunks = [
        chat_chunk(content="x"),
        chat_chunk(
            done=True, eval_count=9, prompt_eval_count=5, total_duration=1
        ),
    ]
    stream = StreamResult(iter(chunks), "chat")
    list(stream)

    assert stream.metrics["cached_prompt_tokens"] == 0


def test_duration_falls_back_to_the_wall_clock():
    """A provider reporting no duration is timed here instead.

    Only Ollama sends a server-side duration. The
    OpenAI-compatible path sent a hard 0, which is
    indistinguishable from a call that took no time — and that is
    what shipped.
    """
    chunks = [
        chat_chunk(content="x"),
        chat_chunk(
            done=True,
            eval_count=9,
            prompt_eval_count=5,
            total_duration=None,
        ),
    ]
    stream = StreamResult(iter(chunks), "chat")
    list(stream)

    assert stream.metrics["total_duration"] > 0


def test_server_reported_duration_wins():
    """Ollama's own nanoseconds are still preferred."""
    chunks = [
        chat_chunk(content="x"),
        chat_chunk(
            done=True,
            eval_count=9,
            prompt_eval_count=5,
            total_duration=3_500_000_000,
        ),
    ]
    stream = StreamResult(iter(chunks), "chat")
    list(stream)

    assert stream.metrics["total_duration"] == 3.5


def test_on_metrics_runs_before_the_completion_callback():
    """Cost is injected before the stream is cached or read.

    A streamed response skipped cost injection entirely, so
    `cost_cents` was `None` however the engine was priced —
    and streaming is the path a chat UI uses.
    """
    seen = {}

    def price(metrics):
        metrics["cost_cents"] = 1.25

    stream = StreamResult(iter([chat_chunk(done=True, eval_count=1)]), "chat")
    stream._on_metrics = price
    stream._on_complete = lambda payload: seen.update(payload)
    list(stream)

    assert stream.metrics["cost_cents"] == 1.25
    # The cached copy carries it too, or a cache hit would lose it.
    assert seen["metrics"]["cost_cents"] == 1.25
