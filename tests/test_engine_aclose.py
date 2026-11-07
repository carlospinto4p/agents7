"""Unit tests for closing async clients (`aclose`)."""

from unittest.mock import AsyncMock

from agents7.agents.chatter import Chatter
from agents7.agents.facts import FactExtractor
from agents7.llm import LLMEngine

from helpers import make_guard, make_guarded_chatter


def _is_closed(client, provider: str) -> bool:
    """Whether a provider's async client is closed."""
    if provider == "ollama":
        return client._client.is_closed
    return client.is_closed()


async def _closes(engine: LLMEngine) -> bool:
    """Close *engine* and say whether its old async client closed."""
    client = engine.llm_provider._async_client
    await engine.aclose()
    return _is_closed(client, engine.provider)


# ── Provider / Engine Tests ────────────────────────────────


async def test_openai_compat_aclose_closes_async_client():
    assert await _closes(LLMEngine("openai_compat", "m", api_key="k"))


async def test_anthropic_aclose_closes_async_client():
    assert await _closes(LLMEngine("anthropic", "m", api_key="k"))


async def test_ollama_aclose_closes_async_client():
    assert await _closes(LLMEngine("ollama", "m"))


async def test_aclose_leaves_a_fresh_async_client():
    """Closing releases connections; it does not end the engine."""
    engine = LLMEngine("openai_compat", "m", api_key="k", timeout=20)
    old = engine.llm_provider._async_client

    await engine.aclose()

    new = engine.llm_provider._async_client
    assert new is not old
    assert not new.is_closed()
    assert new.timeout == 20
    assert new.max_retries == 0


async def test_aclose_leaves_sync_client_open():
    engine = LLMEngine("openai_compat", "m", api_key="k")

    await engine.aclose()

    assert not engine.llm_provider._client.is_closed()


async def test_aclose_closes_same_provider_fallback():
    engine = LLMEngine("openai_compat", "m", api_key="k", fallback_model="f")
    client = engine._fallback_provider._async_client

    await engine.aclose()

    assert client.is_closed()


async def test_aclose_closes_fallback_engine():
    fallback = LLMEngine("ollama", "local")
    engine = LLMEngine(
        "openai_compat", "m", api_key="k", fallback_engine=fallback
    )
    client = fallback.llm_provider._async_client

    await engine.aclose()

    assert client._client.is_closed


def test_check_available_closes_what_it_opened():
    """Its loop ends with the call, so it must close in that loop."""
    engine = LLMEngine("ollama", "m")
    engine.llm_provider.chat_async = AsyncMock(return_value={})
    engine.llm_provider.aclose = AsyncMock()

    check = engine.check_available()

    assert check.answered
    engine.llm_provider.aclose.assert_awaited_once()


def test_check_available_closes_when_unanswered():
    engine = LLMEngine("ollama", "m")
    engine.llm_provider.chat_async = AsyncMock(side_effect=OSError("down"))
    engine.llm_provider.aclose = AsyncMock()

    check = engine.check_available()

    assert not check.answered
    engine.llm_provider.aclose.assert_awaited_once()


# ── Agent Tests ────────────────────────────────────────────


async def test_agent_aclose_closes_guard_engine():
    guard_engine = LLMEngine("ollama", "guard")
    chatter_engine = LLMEngine("openai_compat", "m", api_key="k")
    guard_client = guard_engine.llm_provider._async_client
    chatter_client = chatter_engine.llm_provider._async_client
    agent = make_guarded_chatter(
        guard=make_guard(engine=guard_engine), engine=chatter_engine
    )

    await agent.aclose()

    assert chatter_client.is_closed()
    assert guard_client._client.is_closed


async def test_agent_aclose_closes_fact_extractor_engine():
    extractor_engine = LLMEngine("ollama", "facts")
    client = extractor_engine.llm_provider._async_client
    agent = Chatter(
        name="c",
        version=1,
        engine=LLMEngine("ollama", "m"),
        system_prompt="p",
        fact_extractor=FactExtractor(
            name="f", version=1, engine=extractor_engine
        ),
    )

    await agent.aclose()

    assert client._client.is_closed


def test_agent_engines_are_distinct():
    """A guard sharing the chatter's engine is closed once."""
    shared = LLMEngine("ollama", "m")
    agent = make_guarded_chatter(
        guard=make_guard(engine=shared), engine=shared
    )

    assert agent._engines() == [shared]
