"""Unit tests for LLM streaming support (no LLM required)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agents7.llm.base import LLMEngine
from agents7.llm.streaming import AsyncStreamResult, StreamResult


# ===================================================================
# Helpers — mock ollama chunk objects
# ===================================================================


def _chat_chunk(
    content: str = "",
    thinking: str = "",
    role: str = "assistant",
    done: bool = False,
    eval_count: int | None = None,
    prompt_eval_count: int | None = None,
    total_duration: int | None = None,
) -> SimpleNamespace:
    """Build a mock ollama chat streaming chunk."""
    msg = SimpleNamespace(
        content=content,
        thinking=thinking,
        role=role,
    )
    return SimpleNamespace(
        message=msg,
        done=done,
        eval_count=eval_count,
        prompt_eval_count=prompt_eval_count,
        total_duration=total_duration,
    )


def _gen_chunk(
    response: str = "",
    thinking: str = "",
    done: bool = False,
    eval_count: int | None = None,
    prompt_eval_count: int | None = None,
    total_duration: int | None = None,
) -> SimpleNamespace:
    """Build a mock ollama generate streaming chunk."""
    return SimpleNamespace(
        response=response,
        thinking=thinking,
        done=done,
        eval_count=eval_count,
        prompt_eval_count=prompt_eval_count,
        total_duration=total_duration,
    )


async def _async_iter(items):
    """Turn a list into an async iterator."""
    for item in items:
        yield item


# ===================================================================
# 1. StreamResult (sync)
# ===================================================================


class TestStreamResult:
    def test_yields_chat_content_chunks(self):
        """Chat chunks yield content strings."""
        chunks = [
            _chat_chunk(content="Hello"),
            _chat_chunk(content=" world"),
            _chat_chunk(
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

    def test_yields_generate_content_chunks(self):
        """Generate chunks yield response strings."""
        chunks = [
            _gen_chunk(response="foo"),
            _gen_chunk(response="bar"),
            _gen_chunk(
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

    def test_metrics_from_final_chunk(self):
        """Metrics are populated from the done=True chunk."""
        chunks = [
            _chat_chunk(content="x"),
            _chat_chunk(
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

    def test_thinking_accumulated_not_yielded(self):
        """Thinking text is accumulated but not yielded."""
        chunks = [
            _chat_chunk(thinking="Let me think..."),
            _chat_chunk(content="Answer"),
            _chat_chunk(
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

    def test_empty_stream(self):
        """Empty iterator produces empty response."""
        stream = StreamResult(iter([]), "chat")
        collected = list(stream)

        assert collected == []
        assert stream.response == ""
        assert stream.thinking is None
        assert stream.metrics == {}

    def test_role_captured_for_chat(self):
        """Role is captured from the first chat chunk."""
        chunks = [
            _chat_chunk(content="Hi", role="assistant"),
            _chat_chunk(
                done=True,
                eval_count=1,
                prompt_eval_count=1,
                total_duration=1_000_000_000,
            ),
        ]
        stream = StreamResult(iter(chunks), "chat")
        list(stream)

        assert stream.role == "assistant"

    def test_role_is_none_for_generate(self):
        """Generate mode has no role."""
        chunks = [
            _gen_chunk(response="out"),
            _gen_chunk(
                done=True,
                eval_count=1,
                prompt_eval_count=1,
                total_duration=1_000_000_000,
            ),
        ]
        stream = StreamResult(iter(chunks), "generate")
        list(stream)

        assert stream.role is None


# ===================================================================
# 2. AsyncStreamResult
# ===================================================================


class TestAsyncStreamResult:
    @pytest.mark.asyncio
    async def test_yields_chat_chunks_async(self):
        """Async chat chunks yield content strings."""
        chunks = [
            _chat_chunk(content="Hello"),
            _chat_chunk(content=" async"),
            _chat_chunk(
                done=True,
                eval_count=6,
                prompt_eval_count=4,
                total_duration=2_000_000_000,
            ),
        ]
        stream = AsyncStreamResult(_async_iter(chunks), "chat")
        collected = []
        async for text in stream:
            collected.append(text)

        assert collected == ["Hello", " async"]
        assert stream.response == "Hello async"

    @pytest.mark.asyncio
    async def test_metrics_from_final_chunk_async(self):
        """Async metrics are populated from done chunk."""
        chunks = [
            _chat_chunk(content="x"),
            _chat_chunk(
                done=True,
                eval_count=20,
                prompt_eval_count=8,
                total_duration=4_000_000_000,
            ),
        ]
        stream = AsyncStreamResult(_async_iter(chunks), "chat")
        async for _ in stream:
            pass

        assert stream.metrics["tokens"] == 20
        assert stream.metrics["prompt_tokens"] == 8
        assert stream.metrics["total_duration"] == 4.0

    @pytest.mark.asyncio
    async def test_thinking_accumulated_async(self):
        """Async thinking is accumulated but not yielded."""
        chunks = [
            _chat_chunk(thinking="Hmm..."),
            _chat_chunk(content="Done"),
            _chat_chunk(
                done=True,
                eval_count=3,
                prompt_eval_count=1,
                total_duration=1_000_000_000,
            ),
        ]
        stream = AsyncStreamResult(_async_iter(chunks), "chat")
        collected = []
        async for text in stream:
            collected.append(text)

        assert collected == ["Done"]
        assert stream.thinking == "Hmm..."


# ===================================================================
# 3. LLMEngine delegation
# ===================================================================


class TestEngineStreamDelegation:
    def _make_engine(self):
        """Create an LLMEngine with a mocked provider."""
        engine = object.__new__(LLMEngine)
        engine.provider = "ollama"
        engine.model_name = "test-model"
        engine.model_id = "ollama::test-model"
        engine.retry_config = None
        engine.llm_provider = Mock()
        return engine

    def test_chat_stream_delegates(self):
        """chat_stream delegates to provider.chat_stream."""
        engine = self._make_engine()
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
    async def test_chat_stream_async_delegates(self):
        """chat_stream_async delegates to provider."""
        engine = self._make_engine()
        sentinel = AsyncStreamResult(_async_iter([]), "chat")
        engine.llm_provider.chat_stream_async = AsyncMock(
            return_value=sentinel
        )

        result = await engine.chat_stream_async(
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result is sentinel
        engine.llm_provider.chat_stream_async.assert_called_once()

    def test_generate_stream_delegates(self):
        """generate_stream delegates to provider."""
        engine = self._make_engine()
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
    async def test_generate_stream_async_delegates(self):
        """generate_stream_async delegates to provider."""
        engine = self._make_engine()
        sentinel = AsyncStreamResult(_async_iter([]), "generate")
        engine.llm_provider.generate_stream_async = AsyncMock(
            return_value=sentinel
        )

        result = await engine.generate_stream_async(prompt="test")

        assert result is sentinel
        engine.llm_provider.generate_stream_async.assert_called_once()

    @patch("agents7.llm.retry.retry_sync")
    def test_streaming_bypasses_retry(self, mock_retry):
        """Streaming methods do NOT use retry wrappers."""
        engine = self._make_engine()
        sentinel = StreamResult(iter([]), "chat")
        engine.llm_provider.chat_stream.return_value = sentinel

        engine.chat_stream(
            messages=[{"role": "user", "content": "hi"}],
        )

        mock_retry.assert_not_called()
