"""Retry with exponential backoff for LLM transport errors."""

import asyncio
import time
from dataclasses import dataclass
from typing import Callable

import ollama

from agents7.exceptions import (
    LLMException,
    ModelUnavailableError,
    ProviderError,
    RateLimitError,
)
from agents7.logs import logger


def _openai_errors() -> tuple:
    """Return OpenAI error classes if the package is installed."""
    try:
        import openai

        return (openai.APIError,)
    except ImportError:
        return ()


def _anthropic_errors() -> tuple:
    """Return Anthropic error classes if the package is installed.

    `anthropic.APIError` is the root of the SDK's own hierarchy and
    is unrelated to `openai.APIError`, so listing the OpenAI one
    caught nothing raised by an Anthropic call. Until this existed,
    an `anthropic.InternalServerError` escaped both retry loops on
    the first failure even though `_is_retryable` would have said
    to retry it.

    :return: The Anthropic error classes, or empty when absent.
    """
    try:
        import anthropic

        return (anthropic.APIError,)
    except ImportError:
        return ()


def timeout_errors() -> tuple:
    """Return the timeout exception each provider raises.

    An engine built with a ``timeout`` sets it on the provider's
    own HTTP client, so an abandoned request surfaces as that
    SDK's timeout error rather than as anything agents7 raised.
    The three are unrelated classes, and Ollama's is httpx's own
    — `ollama.RequestError` does not wrap it — so a request that
    ran out of time would otherwise escape the retry loop
    entirely instead of being retried.

    :return: The timeout classes of every installed provider.
    """
    import httpx

    errors: list[type] = [httpx.TimeoutException]
    try:
        import openai

        errors.append(openai.APITimeoutError)
    except ImportError:
        pass
    try:
        import anthropic

        errors.append(anthropic.APITimeoutError)
    except ImportError:
        pass
    return tuple(errors)


@dataclass
class AttemptStats:
    """What the retry loop spent before a call succeeded.

    Passed into a retry loop, mutated by it, and read afterwards
    by the dispatcher, which puts the numbers on the response's
    metrics. Without it a call retried three times reached the
    caller looking exactly like one that succeeded first time.

    A failed attempt raises before the provider reports any
    usage, so it has no tokens and no price to record. What can
    be recorded is that it happened and how long it held the
    caller up, which is the part a slow run needs explaining.

    Attributes:
        retries: How many attempts failed and were retried.
        retry_duration: Seconds those attempts took, including
            the backoff wait after each.
    """

    retries: int = 0
    retry_duration: float = 0.0


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry with exponential backoff.

    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retries).
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Maximum delay cap in seconds.
        backoff_factor: Multiplier applied to delay after each retry.
        retry_connection_errors: Whether a failure to reach a
            *paid* provider at all — `openai.APIConnectionError`,
            `anthropic.APIConnectionError` — is retried. Ollama
            retries one either way, so this is the flag that
            makes the three agree. Default ``False``, which
            keeps what every provider did before v5.5.0.

            This is the one deliberate exception to the
            roadmap's provider-agnostic invariant, and the
            reason is that the providers are not alike in what
            a retry risks: Ollama is a local server, so an
            attempt against it costs nothing but time, and the
            server restarting between two calls is the ordinary
            case this covers. Reaching for a paid endpoint is a
            decision the caller makes explicitly.

            Worth knowing before setting it: a connection
            failure never reached the provider, so no tokens
            were billed for the attempt that failed. What a
            retry actually costs is the wait — a full backoff
            (1s + 2s under `DEFAULT_RETRY`) before a caller
            learns an endpoint is simply down.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    retry_connection_errors: bool = False

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise LLMException(
                f"max_retries must be >= 0, got {self.max_retries}"
            )
        if self.base_delay <= 0:
            raise LLMException(
                f"base_delay must be > 0, got {self.base_delay}"
            )
        if self.max_delay <= 0:
            raise LLMException(f"max_delay must be > 0, got {self.max_delay}")
        if self.backoff_factor < 1:
            raise LLMException(
                f"backoff_factor must be >= 1, got {self.backoff_factor}"
            )


#: What an engine retries with when its caller names nothing.
#:
#: The `openai` and `anthropic` clients both retry twice by
#: themselves, and until v5.3.0 the framework only switched that
#: off when a ``timeout`` was declared. So a config that declared
#: neither could have three attempts made on its behalf while
#: `LLMMetrics.retries` reported zero and ``retry_duration``
#: reported nothing — a measurement produced by a library and
#: recorded by nobody, which is the failure
#: ``.claude/rules/metrics-completeness.md`` exists to catch.
#:
#: Both SDKs are now switched off in every case, so this is what
#: takes their place: two retries, the same attempt count they
#: made, under a layer that counts and times them. A caller that
#: wants none says so with ``RetryConfig(max_retries=0)`` rather
#: than by leaving it unset.
DEFAULT_RETRY = RetryConfig(max_retries=2)


def connection_errors() -> tuple:
    """Return the "never reached the server" error of each SDK.

    A connection failure is the one transient that carries no
    status code, because there was no response to carry it — so
    `_is_retryable`'s fall-through to ``status_code`` could
    never classify one. Ollama's is `ollama.RequestError`,
    which that function already names; the other two are listed
    here so they can be named too.

    Both SDKs subclass their timeout error from their
    connection error, so this tuple contains the timeouts as
    well. `_is_retryable` checks `timeout_errors` first, which
    is what keeps a timeout retryable however
    ``retry_connection_errors`` is set.

    :return: The connection classes of every installed provider.
    """
    errors: list[type] = []
    try:
        import openai

        errors.append(openai.APIConnectionError)
    except ImportError:
        pass
    try:
        import anthropic

        errors.append(anthropic.APIConnectionError)
    except ImportError:
        pass
    return tuple(errors)


def _is_retryable(
    error: Exception,
    config: RetryConfig | None = None,
) -> bool:
    """Check whether an exception is retryable.

    Retryable: any provider's timeout (the engine's ``timeout``
    expiring is a stall, which is exactly what a second attempt
    is for), ollama.RequestError (connection failures),
    ollama.ResponseError with status_code >= 500 (server errors),
    and any OpenAI or Anthropic API error carrying a status of
    429 or >= 500. Non-retryable: client errors like 404.

    A failure to reach a paid provider at all is retryable only
    when *config* asks for it — see
    `RetryConfig.retry_connection_errors` for why the three
    providers deliberately differ here.

    :param error: The exception the provider call raised.
    :param config: The retry configuration in force, or
        ``None`` to answer under the defaults.
    :return: Whether another attempt is worth making.
    """
    if isinstance(error, timeout_errors()):
        return True
    if isinstance(error, ollama.RequestError):
        return True
    if isinstance(error, ollama.ResponseError):
        return error.status_code >= 500
    if isinstance(error, connection_errors()):
        return bool(config and config.retry_connection_errors)
    status = getattr(error, "status_code", None)
    if status is not None:
        return status >= 500 or status == 429
    return False


def _retry_delay(
    error: Exception,
    attempt: int,
    config: RetryConfig,
) -> float:
    """Compute backoff delay, log, or raise if exhausted.

    :param error: The retryable exception that was raised.
    :param attempt: Zero-based attempt number.
    :param config: Retry configuration.
    :return: Delay in seconds before the next retry.
    :raises LLMException: When all retries are exhausted.
    """
    if attempt >= config.max_retries:
        msg = (
            f"All {config.max_retries} retries exhausted. Last error: {error}"
        )
        if (
            isinstance(error, ollama.ResponseError)
            and error.status_code == 429
        ):
            raise RateLimitError(msg) from error
        if (
            isinstance(error, ollama.ResponseError)
            and error.status_code == 404
        ):
            raise ModelUnavailableError(msg) from error
        raise ProviderError(msg) from error
    delay = min(
        config.base_delay * config.backoff_factor**attempt,
        config.max_delay,
    )
    logger.warning(
        f"Retry {attempt + 1}/{config.max_retries}"
        f" after error: {error}. "
        f"Retrying in {delay:.1f}s..."
    )
    return delay


def _all_provider_errors() -> tuple:
    """Return all known provider error types."""
    return (
        ollama.RequestError,
        ollama.ResponseError,
        *_openai_errors(),
        *_anthropic_errors(),
        *timeout_errors(),
    )


def _count_failure(
    stats: AttemptStats | None,
    started: float,
) -> None:
    """Record one failed attempt and how long it took.

    Called before the delay is computed, because computing it is
    what raises once the retries run out — and an attempt that
    failed still failed. Nothing reads the stats after that
    raise unless a fallback model rescues the call, and then
    every attempt it took is wanted.

    :param stats: Where to record it, or ``None`` to discard.
    :param started: `time.perf_counter` when the attempt began.
    """
    if stats is None:
        return
    stats.retries += 1
    stats.retry_duration += time.perf_counter() - started


def _count_backoff(stats: AttemptStats | None, delay: float) -> None:
    """Add the backoff wait that follows a failed attempt.

    The wait is part of what the caller waited for the response,
    so it belongs in the same figure as the failed attempts
    themselves.

    :param stats: Where to record it, or ``None`` to discard.
    :param delay: Seconds about to be slept.
    """
    if stats is not None:
        stats.retry_duration += delay


def attach_attempt_stats(
    error: BaseException,
    stats: AttemptStats,
) -> BaseException:
    """Put what the retry loop counted onto the error ending a call.

    `AttemptStats` is local to one dispatch, so a call that never
    succeeded took its count with it: the failure reached the
    caller and every attempt it had cost was gone, leaving a failed
    call's record able only to say it made no retries. Every
    attempt the loop counted as failed is carried, the last one
    included.

    :param error: The exception about to be raised.
    :param stats: What the dispatch counted.
    :return: *error*, so a caller can ``raise`` the result.
    """
    error.retries = stats.retries
    error.retry_duration = stats.retry_duration
    return error


def _retry_impl(
    fn: Callable,
    sleep_fn: Callable[[float], None],
    retry_config: RetryConfig | None,
    stats: AttemptStats | None = None,
    **kwargs,
):
    """Shared sync retry loop with pluggable sleep callable."""
    if retry_config is None or retry_config.max_retries == 0:
        return fn(**kwargs)
    for attempt in range(retry_config.max_retries + 1):
        started = time.perf_counter()
        try:
            return fn(**kwargs)
        except _all_provider_errors() as e:
            if not _is_retryable(e, retry_config):
                raise
            _count_failure(stats, started)
            delay = _retry_delay(e, attempt, retry_config)
            _count_backoff(stats, delay)
            sleep_fn(delay)


def retry_sync(
    fn: Callable,
    retry_config: RetryConfig | None,
    stats: AttemptStats | None = None,
    **kwargs,
) -> dict:
    """Execute fn with retry and exponential backoff (sync).

    :param fn: The callable to execute.
    :param retry_config: Retry configuration. If None or
        max_retries == 0, fn is called once without retry.
    :param stats: Optional `AttemptStats` to record failed
        attempts into, so the response can report them.
    :param kwargs: Keyword arguments passed to fn.
    :return: The result of fn(**kwargs).
    :raises LLMException: When all retry attempts are exhausted.
    """
    return _retry_impl(fn, time.sleep, retry_config, stats, **kwargs)


async def retry_async(
    fn: Callable,
    retry_config: RetryConfig | None,
    stats: AttemptStats | None = None,
    **kwargs,
) -> dict:
    """Execute fn with retry and exponential backoff (async).

    :param fn: The async callable to execute.
    :param retry_config: Retry configuration. If None or
        max_retries == 0, fn is called once without retry.
    :param stats: Optional `AttemptStats` to record failed
        attempts into, so the response can report them.
    :param kwargs: Keyword arguments passed to fn.
    :return: The result of await fn(**kwargs).
    :raises LLMException: When all retry attempts are exhausted.
    """
    if retry_config is None or retry_config.max_retries == 0:
        return await fn(**kwargs)

    for attempt in range(retry_config.max_retries + 1):
        started = time.perf_counter()
        try:
            return await fn(**kwargs)
        except _all_provider_errors() as e:
            if not _is_retryable(e, retry_config):
                raise
            _count_failure(stats, started)
            delay = _retry_delay(e, attempt, retry_config)
            _count_backoff(stats, delay)
            await asyncio.sleep(delay)
