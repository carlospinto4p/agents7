"""Unit tests for retry with exponential backoff (no LLM required)."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import ollama
import pytest

from agents7.agents.labeler import Labeler
from agents7.exceptions import LLMException
from agents7.llm import LLMEngine
from agents7.llm.dispatcher import _dispatch_errors
from agents7.llm.retry import (
    DEFAULT_RETRY,
    AttemptStats,
    RetryConfig,
    _all_provider_errors,
    _is_retryable,
    retry_async,
    retry_sync,
)


# --- _is_retryable ---


def test_is_retryable_request_error():
    """ollama.RequestError is always retryable."""
    assert _is_retryable(ollama.RequestError("conn refused")) is True


def test_is_retryable_response_error_500():
    """ollama.ResponseError with status >= 500 is retryable."""
    err = ollama.ResponseError("server error")
    err.status_code = 500
    assert _is_retryable(err) is True


def test_is_not_retryable_response_error_400():
    """ollama.ResponseError with status 400 is not retryable."""
    err = ollama.ResponseError("bad request")
    err.status_code = 400
    assert _is_retryable(err) is False


def test_is_not_retryable_response_error_401():
    """ollama.ResponseError with status 401 is not retryable."""
    err = ollama.ResponseError("unauthorized")
    err.status_code = 401
    assert _is_retryable(err) is False


def test_is_not_retryable_response_error_403():
    """ollama.ResponseError with status 403 is not retryable."""
    err = ollama.ResponseError("forbidden")
    err.status_code = 403
    assert _is_retryable(err) is False


def test_is_retryable_generic_429():
    """Generic error with status_code=429 is retryable."""

    class FakeAPIError(Exception):
        status_code = 429

    assert _is_retryable(FakeAPIError()) is True


def test_is_not_retryable_generic_400():
    """Generic error with status_code=400 is not retryable."""

    class FakeAPIError(Exception):
        status_code = 400

    assert _is_retryable(FakeAPIError()) is False


def test_is_not_retryable_no_status():
    """Plain exception without status_code is not retryable."""
    assert _is_retryable(ValueError("generic")) is False


# --- RetryConfig validation ---


def test_defaults():
    """Default values are 3 retries, 1s base, 30s cap, 2x."""
    cfg = RetryConfig()
    assert cfg.max_retries == 3
    assert cfg.base_delay == 1.0
    assert cfg.max_delay == 30.0
    assert cfg.backoff_factor == 2.0


def test_custom_values():
    """Custom values are stored correctly."""
    cfg = RetryConfig(
        max_retries=5,
        base_delay=0.5,
        max_delay=60.0,
        backoff_factor=3.0,
    )
    assert cfg.max_retries == 5
    assert cfg.base_delay == 0.5
    assert cfg.max_delay == 60.0
    assert cfg.backoff_factor == 3.0


def test_negative_max_retries():
    """Negative max_retries raises LLMException."""
    with pytest.raises(LLMException, match="max_retries"):
        RetryConfig(max_retries=-1)


def test_invalid_base_delay():
    """Zero base_delay raises LLMException."""
    with pytest.raises(LLMException, match="base_delay"):
        RetryConfig(base_delay=0)


def test_invalid_backoff_factor():
    """backoff_factor < 1 raises LLMException."""
    with pytest.raises(LLMException, match="backoff_factor"):
        RetryConfig(backoff_factor=0.5)


# --- retry_sync logic ---


def test_succeeds_first_attempt():
    """No retry needed, returns result directly."""
    fn = Mock(return_value={"response": "ok"})
    cfg = RetryConfig(max_retries=3, base_delay=0.01)

    result = retry_sync(fn, cfg, x=1)

    assert result == {"response": "ok"}
    assert fn.call_count == 1


@patch("agents7.llm.retry.time.sleep")
def test_succeeds_after_retry(mock_sleep):
    """RequestError on first call, success on second."""
    fn = Mock(
        side_effect=[
            ollama.RequestError("connection refused"),
            {"response": "ok"},
        ]
    )
    cfg = RetryConfig(max_retries=3, base_delay=0.01)

    result = retry_sync(fn, cfg, x=1)

    assert result == {"response": "ok"}
    assert fn.call_count == 2
    mock_sleep.assert_called_once()


@patch("agents7.llm.retry.time.sleep")
def test_retries_exhausted(mock_sleep):
    """All retries fail, raises LLMException."""
    fn = Mock(side_effect=ollama.RequestError("connection refused"))
    cfg = RetryConfig(max_retries=2, base_delay=0.01)

    with pytest.raises(LLMException, match="retries exhausted"):
        retry_sync(fn, cfg)

    # 1 initial + 2 retries = 3 calls
    assert fn.call_count == 3
    assert mock_sleep.call_count == 2


def test_non_retryable_error():
    """ResponseError(404) raises immediately without retry."""
    error = ollama.ResponseError("not found")
    error.status_code = 404
    fn = Mock(side_effect=error)
    cfg = RetryConfig(max_retries=3, base_delay=0.01)

    with pytest.raises(ollama.ResponseError):
        retry_sync(fn, cfg)

    assert fn.call_count == 1


@patch("agents7.llm.retry.time.sleep")
def test_server_error_retried(mock_sleep):
    """ResponseError(500) triggers retry, then succeeds."""
    error = ollama.ResponseError("internal server error")
    error.status_code = 500
    fn = Mock(side_effect=[error, {"response": "recovered"}])
    cfg = RetryConfig(max_retries=3, base_delay=0.01)

    result = retry_sync(fn, cfg)

    assert result == {"response": "recovered"}
    assert fn.call_count == 2
    mock_sleep.assert_called_once()


def test_no_retry_when_none():
    """config=None means error propagates as-is."""
    fn = Mock(side_effect=ollama.RequestError("connection refused"))

    with pytest.raises(ollama.RequestError):
        retry_sync(fn, None)

    assert fn.call_count == 1


# --- retry_async logic ---


@pytest.mark.asyncio
@patch("agents7.llm.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_async_succeeds_after_retry(mock_sleep):
    """Async RequestError on first call, success on second."""
    fn = AsyncMock(
        side_effect=[
            ollama.RequestError("connection refused"),
            {"response": "ok"},
        ]
    )
    cfg = RetryConfig(max_retries=3, base_delay=0.01)

    result = await retry_async(fn, cfg, x=1)

    assert result == {"response": "ok"}
    assert fn.call_count == 2
    mock_sleep.assert_called_once()


@pytest.mark.asyncio
@patch("agents7.llm.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_async_retries_exhausted(mock_sleep):
    """All async retries fail, raises LLMException."""
    fn = AsyncMock(side_effect=ollama.RequestError("connection refused"))
    cfg = RetryConfig(max_retries=2, base_delay=0.01)

    with pytest.raises(LLMException, match="retries exhausted"):
        await retry_async(fn, cfg)

    assert fn.call_count == 3
    assert mock_sleep.call_count == 2


# --- YAML config parsing ---


def test_yaml_with_retry(tmp_path):
    """Parses retry section into RetryConfig."""
    yaml_content = (
        "name: RetryLabeler\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "    retry:\n"
        "        max_retries: 5\n"
        "        base_delay: 2.0\n"
        "system_prompt: Test prompt.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
    )
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content)

    agent = Labeler.from_yaml(str(yaml_file))
    cfg = agent.engine.retry_config

    assert cfg is not None
    assert cfg.max_retries == 5
    assert cfg.base_delay == 2.0
    # Defaults for unspecified fields
    assert cfg.max_delay == 30.0
    assert cfg.backoff_factor == 2.0


def test_yaml_without_retry(tmp_path):
    """No retry section means the default, not no retrying.

    Before v5.3.0 this left `retry_config` as ``None`` and the
    provider SDK quietly retried twice instead, uncounted. The
    default replaces that layer with one that reports itself.
    """
    yaml_content = (
        "name: NoRetryLabeler\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Test prompt.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
    )
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content)

    agent = Labeler.from_yaml(str(yaml_file))

    assert agent.engine.retry_config == DEFAULT_RETRY


# --- A 500 is caught and retried on every provider ---
#
# Regression guard. Each provider raises its own unrelated error
# hierarchy, and the retry loops build their `except` clause from
# `_all_provider_errors()`. Anthropic was missing from that tuple
# for as long as it had existed, so `_is_retryable` said to retry
# an `anthropic.InternalServerError` and the call never reached it.
# Testing only the providers that happened to work is what let a
# whole provider go unretried, so all three are covered here.


def _ollama_500() -> Exception:
    """Build the 500 an Ollama call raises."""
    err = ollama.ResponseError("internal server error")
    err.status_code = 500
    return err


def _openai_500() -> Exception:
    """Build the 500 an OpenAI-compatible call raises."""
    import openai

    return openai.InternalServerError(
        "internal server error",
        response=_http_500(),
        body=None,
    )


def _anthropic_500() -> Exception:
    """Build the 500 an Anthropic call raises."""
    import anthropic

    return anthropic.InternalServerError(
        "internal server error",
        response=_http_500(),
        body=None,
    )


def _http_500() -> httpx.Response:
    """Build the HTTP response both SDK errors are built from."""
    return httpx.Response(
        500,
        request=httpx.Request("POST", "https://example.invalid/v1"),
    )


_PROVIDER_500 = [
    pytest.param(_ollama_500, id="ollama"),
    pytest.param(_openai_500, id="openai_compat"),
    pytest.param(_anthropic_500, id="anthropic"),
]


@pytest.mark.parametrize("build_error", _PROVIDER_500)
def test_provider_500_is_retryable(build_error):
    """Every provider's 500 is classified as retryable."""
    assert _is_retryable(build_error()) is True


@pytest.mark.parametrize("build_error", _PROVIDER_500)
def test_provider_500_is_caught(build_error):
    """Every provider's 500 is in the retry loop's except clause.

    `_is_retryable` returning True is worth nothing if the error
    never reaches it, which is exactly how Anthropic failed.
    """
    assert isinstance(build_error(), _all_provider_errors())


@pytest.mark.parametrize("build_error", _PROVIDER_500)
def test_provider_500_reaches_the_dispatcher(build_error):
    """Every provider's 500 triggers fallback rather than escaping."""
    assert isinstance(build_error(), _dispatch_errors())


@pytest.mark.parametrize("build_error", _PROVIDER_500)
def test_provider_500_retried_sync(build_error):
    """A 500 from any provider is retried, then succeeds."""
    fn = Mock(side_effect=[build_error(), {"response": "recovered"}])

    with patch("agents7.llm.retry.time.sleep"):
        result = retry_sync(fn, RetryConfig(max_retries=3, base_delay=0.01))

    assert result == {"response": "recovered"}
    assert fn.call_count == 2


@pytest.mark.parametrize("build_error", _PROVIDER_500)
async def test_provider_500_retried_async(build_error):
    """The async loop retries a 500 from any provider too."""
    fn = AsyncMock(side_effect=[build_error(), {"response": "recovered"}])

    with patch("agents7.llm.retry.asyncio.sleep", new=AsyncMock()):
        result = await retry_async(
            fn, RetryConfig(max_retries=3, base_delay=0.01)
        )

    assert result == {"response": "recovered"}
    assert fn.call_count == 2


@pytest.mark.parametrize("build_error", _PROVIDER_500)
def test_provider_500_counted_as_a_retry(build_error):
    """The attempt a 500 cost is recorded, not silently absorbed."""
    stats = AttemptStats()
    fn = Mock(side_effect=[build_error(), {"response": "recovered"}])

    with patch("agents7.llm.retry.time.sleep"):
        retry_sync(fn, RetryConfig(max_retries=3, base_delay=0.01), stats)

    assert stats.retries == 1
    assert stats.retry_duration > 0


# --- The default retry layer ---
#
# Item 2 of the 4th September backlog. Both SDKs retried twice by
# themselves and the framework only switched that off when a
# `timeout` was declared, so a config declaring neither had three
# attempts made for it while `LLMMetrics.retries` reported zero —
# a measurement produced by a library and recorded by nobody. The
# SDKs are now off in every case and `DEFAULT_RETRY` takes their
# place, which is only worth doing if what it does is counted.


def test_default_retry_replaces_the_sdk_attempt_count():
    """The default makes the same three attempts the SDKs made."""
    assert DEFAULT_RETRY.max_retries == 2


def test_engine_without_retry_gets_the_default():
    """An engine given no `retry` retries, rather than not."""
    engine = LLMEngine("ollama", "test-model")

    assert engine.retry_config == DEFAULT_RETRY
    assert engine._dispatcher._retry_config == DEFAULT_RETRY


def test_engine_retry_can_still_be_switched_off():
    """`max_retries=0` is how a caller now asks for no retrying.

    `retry=None` no longer means it, so there has to be a way to
    say it, and the dispatcher has to receive what was asked for.
    """
    engine = LLMEngine(
        "ollama", "test-model", retry=RetryConfig(max_retries=0)
    )

    assert engine.retry_config.max_retries == 0
    assert engine._dispatcher._retry_config.max_retries == 0


def test_a_default_retry_is_counted():
    """An undeclared retry still reports itself.

    The whole point of replacing the SDK layer: an agent whose
    config says nothing about retrying now has its attempts on
    the response instead of hidden inside a client.
    """
    provider = Mock()
    provider.chat.side_effect = [
        ollama.RequestError("connection refused"),
        {"response": "recovered", "metrics": {}},
    ]
    engine = LLMEngine("ollama", "test-model")
    engine._dispatcher._primary = provider

    with patch("agents7.llm.retry.time.sleep"):
        result = engine._dispatcher.execute(
            "chat",
            provider.chat,
            "messages",
            [],
            temperature=0,
            seed=None,
            max_tokens=-1,
            output_format=None,
            think=False,
        )

    assert provider.chat.call_count == 2
    assert result["metrics"]["retries"] == 1
    assert result["metrics"]["retry_duration"] > 0


# --- A connection failure is retried where the config says ---
#
# Regression guard for the gap found on 4th September 2026: a
# connection failure carries no status code, because nothing
# answered to give it one, so `_is_retryable`'s fall-through to
# `status_code` classified `openai.APIConnectionError` and
# `anthropic.APIConnectionError` as non-retryable while the
# identical failure against Ollama was retried. Both are now
# named, and `RetryConfig.retry_connection_errors` decides.


def _ollama_connection_error() -> Exception:
    """Build the connection failure an Ollama call raises."""
    return ollama.RequestError("connection refused")


def _openai_connection_error() -> Exception:
    """Build the connection failure an OpenAI-compat call raises."""
    import openai

    return openai.APIConnectionError(
        request=httpx.Request("POST", "https://example.invalid/v1"),
    )


def _anthropic_connection_error() -> Exception:
    """Build the connection failure an Anthropic call raises."""
    import anthropic

    return anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://example.invalid/v1"),
    )


_PAID_CONNECTION_ERRORS = [
    pytest.param(_openai_connection_error, id="openai_compat"),
    pytest.param(_anthropic_connection_error, id="anthropic"),
]

_ALL_CONNECTION_ERRORS = [
    pytest.param(_ollama_connection_error, id="ollama"),
    *_PAID_CONNECTION_ERRORS,
]


@pytest.mark.parametrize("build_error", _ALL_CONNECTION_ERRORS)
def test_connection_error_is_caught(build_error):
    """Every provider's connection failure reaches the loop.

    Classification is worth nothing if the error never gets to
    `_is_retryable`, which is how a whole provider went
    unretried before.
    """
    assert isinstance(build_error(), _all_provider_errors())


@pytest.mark.parametrize("build_error", _ALL_CONNECTION_ERRORS)
def test_connection_error_reaches_the_dispatcher(build_error):
    """A connection failure triggers fallback, not an escape."""
    assert isinstance(build_error(), _dispatch_errors())


@pytest.mark.parametrize("build_error", _PAID_CONNECTION_ERRORS)
def test_paid_connection_error_not_retried_by_default(build_error):
    """Reaching for a paid endpoint is opt-in, so this is off."""
    assert _is_retryable(build_error(), RetryConfig()) is False


@pytest.mark.parametrize("build_error", _PAID_CONNECTION_ERRORS)
def test_paid_connection_error_retried_when_asked(build_error):
    """Declaring the flag makes the three providers agree."""
    config = RetryConfig(retry_connection_errors=True)
    assert _is_retryable(build_error(), config) is True


def test_ollama_connection_error_retried_either_way():
    """A local server costs nothing to reach, so it always is."""
    error = _ollama_connection_error()
    assert _is_retryable(error, RetryConfig()) is True
    assert (
        _is_retryable(error, RetryConfig(retry_connection_errors=True))
        is True
    )


@pytest.mark.parametrize("build_error", _PAID_CONNECTION_ERRORS)
def test_paid_connection_error_retried_sync(build_error):
    """The opt-in reaches the sync loop, not just the classifier."""
    fn = Mock(side_effect=[build_error(), {"response": "recovered"}])

    with patch("agents7.llm.retry.time.sleep"):
        result = retry_sync(
            fn,
            RetryConfig(
                max_retries=3,
                base_delay=0.01,
                retry_connection_errors=True,
            ),
        )

    assert result == {"response": "recovered"}
    assert fn.call_count == 2


@pytest.mark.parametrize("build_error", _PAID_CONNECTION_ERRORS)
async def test_paid_connection_error_retried_async(build_error):
    """The async loop honours the opt-in too."""
    fn = AsyncMock(side_effect=[build_error(), {"response": "recovered"}])

    with patch("agents7.llm.retry.asyncio.sleep", new=AsyncMock()):
        result = await retry_async(
            fn,
            RetryConfig(
                max_retries=3,
                base_delay=0.01,
                retry_connection_errors=True,
            ),
        )

    assert result == {"response": "recovered"}
    assert fn.call_count == 2


@pytest.mark.parametrize("build_error", _PAID_CONNECTION_ERRORS)
def test_paid_connection_error_raises_without_the_opt_in(build_error):
    """Without it the caller learns immediately, as before."""
    fn = Mock(side_effect=build_error())

    with patch("agents7.llm.retry.time.sleep"):
        with pytest.raises(type(build_error())):
            retry_sync(fn, RetryConfig(max_retries=3, base_delay=0.01))

    assert fn.call_count == 1


@pytest.mark.parametrize("build_error", _PAID_CONNECTION_ERRORS)
def test_timeout_stays_retryable_without_the_opt_in(build_error):
    """Both SDKs subclass their timeout from their connection
    error, so the flag must not switch timeouts off with them.
    """
    import anthropic
    import openai

    timeout = (
        openai.APITimeoutError(
            request=httpx.Request("POST", "https://example.invalid/v1")
        )
        if isinstance(build_error(), openai.APIConnectionError)
        else anthropic.APITimeoutError(
            request=httpx.Request("POST", "https://example.invalid/v1")
        )
    )
    assert isinstance(timeout, type(build_error()))
    assert _is_retryable(timeout, RetryConfig()) is True
