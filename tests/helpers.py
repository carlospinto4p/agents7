"""Shared test helpers used across test files."""

import asyncio
import json
from enum import StrEnum
from types import SimpleNamespace
from unittest.mock import Mock

from pydantic import BaseModel

from agents7.agents.labeler import Labeler
from agents7.llm import LLMEngine
from agents7.llm.cache import CacheConfig, ResponseCache
from agents7.llm.dispatcher import EngineDispatcher
from agents7.models import AgentResponse


# --- Shared constants ---

ENGINE = LLMEngine(provider="ollama", model_name="qwen3")

PERSONA_SUMMARY = "David Garcia, Spanish journalist, curious"

DUMMY_METRICS = {
    "tokens": 10,
    "prompt_tokens": 5,
    "total_duration": 0.1,
}


# --- Shared models ---


class DummyOutput(BaseModel):
    value: str


class DummyReason(StrEnum):
    GOOD = "Good result."
    BAD = "Bad result."


class DummyOutputWithEnum(BaseModel):
    label: str
    reason: DummyReason


# --- Response factory helpers ---


def batch_chat_response(
    items: list[dict],
    thinking: str | None = None,
) -> dict:
    """Build a mock chat return value for batch mode."""
    return {
        "role": "assistant",
        "response": json.dumps({"items": items}),
        "thinking": thinking,
        "metrics": DUMMY_METRICS,
    }


def make_agent_response(
    model: BaseModel | None,
    idx: int | None = None,
) -> AgentResponse:
    """Build a mock AgentResponse with arbitrary model."""
    return AgentResponse(
        response="raw",
        model=model,
        thinking=None,
        metrics=DUMMY_METRICS,
        idx=idx,
    )


# --- Agent factory helpers ---


def make_labeler(**kwargs) -> Labeler:
    """Create a Labeler with sensible test defaults."""
    defaults = {
        "name": "test",
        "version": 1,
        "engine": ENGINE,
        "system_prompt": "Test prompt.",
        "output_model": DummyOutput,
    }
    defaults.update(kwargs)
    return Labeler(**defaults)


def make_agent(
    agent_cls: type = Labeler,
    **kwargs,
):
    """Create any agent type with sensible test defaults.

    Works for Labeler, Detector, Classifier, Regressor.
    Pass ``output_model`` to override the default.
    """
    defaults = {
        "name": "test",
        "version": 1,
        "engine": ENGINE,
        "system_prompt": "Test prompt.",
        "output_model": DummyOutput,
    }
    defaults.update(kwargs)
    return agent_cls(**defaults)


# --- Engine factory helpers ---


def make_engine(cache: CacheConfig | None = None):
    """Create an LLMEngine with a mocked provider."""
    engine = object.__new__(LLMEngine)
    engine.provider = "ollama"
    engine.model_name = "test-model"
    engine.model_id = "ollama::test-model"
    engine.retry_config = None
    engine.cost_per_1m_tokens = None
    provider = Mock()
    engine.llm_provider = provider
    engine._dispatcher = EngineDispatcher(
        primary=provider,
        fallback=None,
        retry_config=None,
        cache=ResponseCache(cache) if cache is not None else None,
        cost_per_1m_tokens=None,
        model_id="ollama::test-model",
    )
    return engine


# --- Streaming chunk helpers ---


def chat_chunk(
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


def gen_chunk(
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


async def async_iter(items):
    """Turn a list into an async iterator."""
    for item in items:
        yield item


class ConcurrencyTracker:
    """Track peak concurrent async calls.

    Usage in tests::

        tracker = ConcurrencyTracker(return_value)

        mock_chat.side_effect = tracker.side_effect
        # ... run async code ...
        assert tracker.peak == expected_peak
    """

    def __init__(self, return_value: dict, delay: float = 0.05):
        self.peak = 0
        self._current = 0
        self._lock = asyncio.Lock()
        self._return_value = return_value
        self._delay = delay

    async def side_effect(self, **kwargs):
        """Use as ``mock.side_effect``."""
        async with self._lock:
            self._current += 1
            self.peak = max(self.peak, self._current)
        await asyncio.sleep(self._delay)
        async with self._lock:
            self._current -= 1
        return dict(self._return_value)


# --- Chat response factory ---


def chat_response_dict(
    text: str = "Hello!",
    thinking: str | None = None,
) -> dict:
    """Build a mock ``engine.chat()`` return value."""
    return {
        "role": "assistant",
        "response": text,
        "thinking": thinking,
        "metrics": DUMMY_METRICS,
    }


# --- Chatter / Guard factory helpers ---


def make_chatter(**kwargs):
    """Create a `Chatter` with sensible test defaults."""
    from agents7.agents.chatter import Chatter

    defaults = {
        "name": "test-chatter",
        "version": 1,
        "engine": ENGINE,
        "system_prompt": "You are a helpful assistant.",
    }
    defaults.update(kwargs)
    return Chatter(**defaults)


def make_guard(**kwargs):
    """Create a `ContextGuard` with test defaults."""
    from agents7.agents.context_guard import ContextGuard

    defaults = {
        "name": "test-guard",
        "version": 1,
        "engine": ENGINE,
        "system_prompt": "(auto-generated)",
        "persona_summary": PERSONA_SUMMARY,
    }
    defaults.update(kwargs)
    return ContextGuard(**defaults)


# --- UI test helpers (moved from conftest.py) ---


class MockModel(BaseModel):
    """Shared mock model for UI evaluation/comparison tests."""

    category: str = "POSITIVE"
    reason: str = "HAPPY"


def mock_metrics():
    """Create standard LLMMetrics for UI tests."""
    from agents7.models.base import LLMMetrics

    return LLMMetrics(tokens=10, prompt_tokens=5, total_duration=0.1)


def make_agent_mock(
    responses: list[AgentResponse],
):
    """Return a mock agent that returns responses in order.

    Mocks both ``act`` (sequential fallback) and
    ``act_in_batch`` (used by ``run_evaluation``).
    """
    from unittest.mock import MagicMock

    agent = MagicMock()
    agent.act = MagicMock(side_effect=responses)
    agent.act_in_batch = MagicMock(
        return_value=list(responses),
    )
    return agent


def make_response(
    category: str = "POSITIVE",
    reason: str = "HAPPY",
) -> AgentResponse:
    """Build a mock AgentResponse for UI tests."""
    model = MockModel(category=category, reason=reason)
    return AgentResponse(
        response="{}",
        model=model,
        metrics=mock_metrics(),
    )
