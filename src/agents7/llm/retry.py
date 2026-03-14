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


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry with exponential backoff.

    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retries).
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Maximum delay cap in seconds.
        backoff_factor: Multiplier applied to delay after each retry.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0

    def __post_init__(self):
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


def _is_retryable(error: Exception) -> bool:
    """Check whether an exception is retryable.

    Retryable: ollama.RequestError (connection failures),
    ollama.ResponseError with status_code >= 500 (server errors),
    and OpenAI API errors with status >= 500 or status 429.
    Non-retryable: client errors like 404.
    """
    if isinstance(error, ollama.RequestError):
        return True
    if isinstance(error, ollama.ResponseError):
        return error.status_code >= 500
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


def retry_sync(
    fn: Callable,
    retry_config: RetryConfig | None,
    **kwargs,
) -> dict:
    """Execute fn with retry and exponential backoff (sync).

    :param fn: The callable to execute.
    :param retry_config: Retry configuration. If None or
        max_retries == 0, fn is called once without retry.
    :param kwargs: Keyword arguments passed to fn.
    :return: The result of fn(**kwargs).
    :raises LLMException: When all retry attempts are exhausted.
    """
    if retry_config is None or retry_config.max_retries == 0:
        return fn(**kwargs)

    for attempt in range(retry_config.max_retries + 1):
        try:
            return fn(**kwargs)
        except (
            ollama.RequestError,
            ollama.ResponseError,
            *_openai_errors(),
        ) as e:
            if not _is_retryable(e):
                raise
            delay = _retry_delay(e, attempt, retry_config)
            time.sleep(delay)


async def retry_async(
    fn: Callable,
    retry_config: RetryConfig | None,
    **kwargs,
) -> dict:
    """Execute fn with retry and exponential backoff (async).

    :param fn: The async callable to execute.
    :param retry_config: Retry configuration. If None or
        max_retries == 0, fn is called once without retry.
    :param kwargs: Keyword arguments passed to fn.
    :return: The result of await fn(**kwargs).
    :raises LLMException: When all retry attempts are exhausted.
    """
    if retry_config is None or retry_config.max_retries == 0:
        return await fn(**kwargs)

    for attempt in range(retry_config.max_retries + 1):
        try:
            return await fn(**kwargs)
        except (
            ollama.RequestError,
            ollama.ResponseError,
            *_openai_errors(),
        ) as e:
            if not _is_retryable(e):
                raise
            delay = _retry_delay(e, attempt, retry_config)
            await asyncio.sleep(delay)
