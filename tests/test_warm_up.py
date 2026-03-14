"""Unit tests for Labeler.warm_up and warm_up_async."""

from unittest.mock import patch

import pytest

from agents7.agents.labeler import Labeler
from agents7.llm import LLMEngine
from agents7.llm.ollama import OllamaProvider

from helpers import DummyOutput

_GENERATE_RETURN = {
    "response": "",
    "thinking": None,
    "metrics": {
        "tokens": 1,
        "prompt_tokens": 1,
        "total_duration": 10_000_000,
    },
}


def _make_agent():
    engine = LLMEngine(provider="ollama", model_name="qwen3")
    return Labeler(
        name="test",
        version=1,
        engine=engine,
        system_prompt="Test.",
        output_model=DummyOutput,
    )


# --- warm_up (sync) ---


@patch.object(OllamaProvider, "generate")
def test_warm_up_calls_generate(mock_generate):
    """warm_up sends a trivial prompt to the engine."""
    mock_generate.return_value = _GENERATE_RETURN
    agent = _make_agent()
    agent.warm_up()

    mock_generate.assert_called_once()
    kwargs = mock_generate.call_args[1]
    assert kwargs["prompt"] == "ping"
    assert kwargs["max_tokens"] == 1


def test_warm_up_noop_for_openai_compat():
    """warm_up is a no-op for non-Ollama providers."""
    agent = _make_agent()
    agent.engine.provider = "openai_compat"

    agent.warm_up()


# --- warm_up_async ---


@pytest.mark.asyncio
async def test_warm_up_async_calls_generate():
    """warm_up_async sends a trivial prompt."""
    agent = _make_agent()
    called = {}

    async def fake_generate_async(**kwargs):
        called.update(kwargs)
        return _GENERATE_RETURN

    agent.engine.generate_async = fake_generate_async
    await agent.warm_up_async()

    assert called["prompt"] == "ping"
    assert called["max_tokens"] == 1


@pytest.mark.asyncio
async def test_warm_up_async_noop_for_openai_compat():
    """warm_up_async is a no-op for non-Ollama providers."""
    agent = _make_agent()
    agent.engine.provider = "openai_compat"
    called = False

    async def fake_generate_async(**kwargs):
        nonlocal called
        called = True

    agent.engine.generate_async = fake_generate_async
    await agent.warm_up_async()

    assert not called
