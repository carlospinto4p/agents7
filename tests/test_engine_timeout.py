"""Unit tests for the engine's per-request timeout.

The timeout is set on each provider's own HTTP client rather
than wrapped around the call, so it covers sync, async and
streaming alike and each retry attempt gets its own budget.
"""

import socket
import threading
import time
from unittest.mock import patch

import httpx
import openai
import pytest

from agents7.agents.config_loader import build_engine
from agents7.exceptions import AgentConfigurationError, ProviderError
from agents7.llm.base import LLMEngine
from agents7.llm.ollama import OllamaProvider
from agents7.llm.openai_compat import OpenAICompatProvider
from agents7.llm.retry import (
    RetryConfig,
    _is_retryable,
    timeout_errors,
)


# ── Engine wiring ──────────────────────────────────────────


def test_engine_passes_timeout_to_ollama_provider():
    """The engine hands its timeout to OllamaProvider."""
    with patch("agents7.llm.ollama.OllamaProvider") as mock_cls:
        LLMEngine._create_provider("ollama", "qwen3", timeout=20)
    mock_cls.assert_called_once_with("qwen3", timeout=20)


def test_engine_passes_timeout_to_openai_compat_provider():
    """The engine hands its timeout to OpenAICompatProvider."""
    target = "agents7.llm.openai_compat.OpenAICompatProvider"
    with patch(target) as mock_cls:
        LLMEngine._create_provider(
            "openai_compat", "deepseek-chat", timeout=20
        )
    assert mock_cls.call_args.kwargs["timeout"] == 20


def test_engine_passes_timeout_to_anthropic_provider():
    """The engine hands its timeout to AnthropicProvider."""
    target = "agents7.llm.anthropic.AnthropicProvider"
    with patch(target) as mock_cls:
        LLMEngine._create_provider("anthropic", "claude-3", timeout=20)
    assert mock_cls.call_args.kwargs["timeout"] == 20


def test_engine_records_timeout_on_itself():
    """`LLMEngine.timeout` reports what was configured."""
    engine = LLMEngine("openai_compat", "m", api_key="x", timeout=20)
    assert engine.timeout == 20.0


def test_engine_timeout_defaults_to_none():
    """Without a timeout the engine reports none."""
    engine = LLMEngine("openai_compat", "m", api_key="x")
    assert engine.timeout is None


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_engine_rejects_non_positive_timeout(bad):
    """A timeout of zero or less is a configuration error."""
    with pytest.raises(ValueError, match="must be > 0"):
        LLMEngine("openai_compat", "m", api_key="x", timeout=bad)


def test_fallback_provider_gets_the_same_timeout():
    """A fallback model is built with the timeout too.

    It answers the same calls as the primary, so a stall there
    would hang exactly as long.
    """
    engine = LLMEngine(
        "openai_compat",
        "m",
        api_key="x",
        fallback_model="m2",
        timeout=20,
    )
    assert engine._fallback_provider._client.timeout == 20


# ── What reaches the SDK clients ───────────────────────────


def test_openai_compat_sets_timeout_on_both_clients():
    """Sync and async clients both carry the timeout."""
    provider = OpenAICompatProvider("m", api_key="x", timeout=20)
    assert provider._client.timeout == 20
    assert provider._async_client.timeout == 20


def test_openai_compat_disables_sdk_retries_with_timeout():
    """Declaring a timeout switches off the SDK's own retrying.

    The SDK retries twice by default, which would make a
    declared 20-second limit take a minute and hide those
    attempts from `LLMMetrics.retries`.
    """
    provider = OpenAICompatProvider("m", api_key="x", timeout=20)
    assert provider._client.max_retries == 0
    assert provider._async_client.max_retries == 0


def test_openai_compat_disables_sdk_retries_without_a_timeout():
    """The SDK's own retrying is off even with no timeout set.

    It used to ride along with the timeout, so a config that
    declared neither had three attempts made for it while
    `LLMMetrics.retries` reported zero. Unconditional since
    v5.3.0 — `DEFAULT_RETRY` takes that layer's place.
    """
    provider = OpenAICompatProvider("m", api_key="x")
    assert provider._client.max_retries == 0
    assert provider._async_client.max_retries == 0
    # The timeout genuinely stays at the SDK's own default.
    assert provider._client.timeout is not None


def test_anthropic_sets_timeout_and_disables_sdk_retries():
    """The Anthropic client is configured the same way."""
    from agents7.llm.anthropic import AnthropicProvider

    provider = AnthropicProvider("claude-3", api_key="x", timeout=20)
    assert provider._client.timeout == 20
    assert provider._client.max_retries == 0
    assert provider._async_client.timeout == 20


def test_anthropic_disables_sdk_retries_without_a_timeout():
    """The Anthropic SDK's own retrying is off with no timeout.

    Same unconditional switch-off as the OpenAI client above.
    """
    from agents7.llm.anthropic import AnthropicProvider

    provider = AnthropicProvider("claude-3", api_key="x")
    assert provider._client.max_retries == 0
    assert provider._async_client.max_retries == 0


def test_ollama_builds_clients_with_the_timeout():
    """Both Ollama clients are built with the timeout.

    Ollama's module-level `chat` builds a fresh client per call
    and so cannot carry one, which is why the sync path uses an
    instance client.
    """
    with patch("agents7.llm.ollama.Client") as sync_cls:
        with patch("agents7.llm.ollama.AsyncClient") as async_cls:
            OllamaProvider("qwen3", timeout=20)
    sync_cls.assert_called_once_with(timeout=20)
    async_cls.assert_called_once_with(timeout=20)


def test_ollama_builds_clients_without_a_timeout():
    """No timeout means the key is not passed at all."""
    with patch("agents7.llm.ollama.Client") as sync_cls:
        with patch("agents7.llm.ollama.AsyncClient"):
            OllamaProvider("qwen3")
    sync_cls.assert_called_once_with()


# ── Reading it from YAML ───────────────────────────────────


def _engine_section(**extra) -> dict:
    return {
        "provider": "openai_compat",
        "model": "deepseek-chat",
        "api_key": "x",
        **extra,
    }


def test_build_engine_reads_the_timeout():
    """`engine.timeout` in YAML reaches the engine."""
    engine = build_engine(_engine_section(timeout=20))
    assert engine.timeout == 20.0


def test_build_engine_without_a_timeout():
    """The section is optional and defaults to no timeout."""
    engine = build_engine(_engine_section())
    assert engine.timeout is None


@pytest.mark.parametrize("bad", ["20s", [20], True, {}])
def test_build_engine_rejects_a_non_numeric_timeout(bad):
    """A timeout that is not a number is a config error."""
    with pytest.raises(AgentConfigurationError, match="number of seconds"):
        build_engine(_engine_section(timeout=bad))


@pytest.mark.parametrize("bad", [0, -5])
def test_build_engine_rejects_a_non_positive_timeout(bad):
    """A timeout of zero or less is a config error."""
    with pytest.raises(AgentConfigurationError, match="greater than 0"):
        build_engine(_engine_section(timeout=bad))


# ── A stall is worth retrying ──────────────────────────────


def test_every_installed_provider_reports_a_timeout_class():
    """Each provider's timeout type is known to the retry loop.

    They are unrelated classes, and Ollama's is httpx's own —
    `ollama.RequestError` does not wrap it — so a request that
    ran out of time would otherwise escape the retry loop.
    """
    classes = timeout_errors()
    assert httpx.TimeoutException in classes
    assert openai.APITimeoutError in classes


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("stalled"),
        httpx.ConnectTimeout("stalled"),
        openai.APITimeoutError(request=httpx.Request("POST", "http://x")),
    ],
)
def test_a_timed_out_call_is_retryable(error):
    """A stall is exactly what a second attempt is for."""
    assert _is_retryable(error) is True


# ── End to end, against a server that never answers ────────


@pytest.fixture
def stalled_endpoint():
    """A URL whose server accepts connections and never replies.

    Reproduces the failure this timeout exists for: the call
    that took 603.6 seconds on 30th August 2026 while every
    other call that day was under 6.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    accepted = []

    def accept_forever() -> None:
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            accepted.append(conn)

    threading.Thread(target=accept_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.getsockname()[1]}/v1"
    srv.close()
    for conn in accepted:
        conn.close()


def _stalling_engine(base_url: str, **extra) -> LLMEngine:
    return LLMEngine(
        "openai_compat",
        "stub-model",
        base_url=base_url,
        api_key="x",
        timeout=0.5,
        **extra,
    )


def test_a_stalled_call_is_abandoned_at_the_timeout(stalled_endpoint):
    """The call gives up at the limit instead of hanging.

    Sync, which had no timeout at all before: the dispatcher
    discarded the per-call argument on that path.

    Retrying is switched off explicitly, so this measures one
    attempt against the limit. Since v5.3.0 an engine given no
    `retry` gets `DEFAULT_RETRY` rather than none, and the test
    below covers what a retried stall costs.
    """
    engine = _stalling_engine(
        stalled_endpoint, retry=RetryConfig(max_retries=0)
    )
    started = time.perf_counter()
    with pytest.raises(openai.APITimeoutError):
        engine.chat(messages=[{"role": "user", "content": "hi"}])
    assert time.perf_counter() - started < 3.0


def test_a_stalled_call_is_retried_then_gives_up(stalled_endpoint):
    """With retries configured, a stall is retried and reported."""
    engine = _stalling_engine(
        stalled_endpoint,
        retry=RetryConfig(max_retries=2, base_delay=0.1, backoff_factor=1.0),
    )
    started = time.perf_counter()
    with pytest.raises(ProviderError, match="retries exhausted"):
        engine.chat(messages=[{"role": "user", "content": "hi"}])
    # Three attempts of 0.5s plus two 0.1s waits, and nowhere
    # near the nine seconds the SDK's own hidden retries cost
    # before this shipped.
    assert 1.5 < time.perf_counter() - started < 5.0


@pytest.mark.asyncio
async def test_a_stalled_async_call_is_abandoned(stalled_endpoint):
    """The async path is bounded by the same limit."""
    engine = _stalling_engine(
        stalled_endpoint, retry=RetryConfig(max_retries=0)
    )
    started = time.perf_counter()
    with pytest.raises(openai.APITimeoutError):
        await engine.chat_async(messages=[{"role": "user", "content": "hi"}])
    assert time.perf_counter() - started < 3.0


@pytest.mark.asyncio
async def test_a_stalled_stream_is_abandoned(stalled_endpoint):
    """Streaming is bounded too — it discarded the timeout before."""
    engine = _stalling_engine(stalled_endpoint)
    started = time.perf_counter()
    with pytest.raises(openai.APITimeoutError):
        await engine.chat_stream_async(
            messages=[{"role": "user", "content": "hi"}]
        )
    assert time.perf_counter() - started < 3.0
