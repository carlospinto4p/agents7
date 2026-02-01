"""Retry with exponential backoff for LLM transport errors."""

import asyncio
import time
from dataclasses import dataclass
from typing import Callable

import ollama

from agents7.exceptions import LLMException
from agents7.logs import logger


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

    Retryable: ollama.RequestError (connection failures) and
    ollama.ResponseError with status_code >= 500 (server errors).
    Non-retryable: ollama.ResponseError with status_code < 500
    (client errors like 404).
    """
    if isinstance(error, ollama.RequestError):
        return True
    if isinstance(error, ollama.ResponseError):
        return error.status_code >= 500
    return False


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

    last_error: Exception | None = None
    for attempt in range(retry_config.max_retries + 1):
        try:
            return fn(**kwargs)
        except (
            ollama.RequestError,
            ollama.ResponseError,
        ) as e:
            if not _is_retryable(e):
                raise
            last_error = e
            if attempt < retry_config.max_retries:
                delay = min(
                    retry_config.base_delay
                    * retry_config.backoff_factor**attempt,
                    retry_config.max_delay,
                )
                logger.warning(
                    f"Retry {attempt + 1}/{retry_config.max_retries}"
                    f" after error: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

    raise LLMException(
        f"All {retry_config.max_retries} retries exhausted. "
        f"Last error: {last_error}"
    ) from last_error


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

    last_error: Exception | None = None
    for attempt in range(retry_config.max_retries + 1):
        try:
            return await fn(**kwargs)
        except (
            ollama.RequestError,
            ollama.ResponseError,
        ) as e:
            if not _is_retryable(e):
                raise
            last_error = e
            if attempt < retry_config.max_retries:
                delay = min(
                    retry_config.base_delay
                    * retry_config.backoff_factor**attempt,
                    retry_config.max_delay,
                )
                logger.warning(
                    f"Retry {attempt + 1}/{retry_config.max_retries}"
                    f" after error: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

    raise LLMException(
        f"All {retry_config.max_retries} retries exhausted. "
        f"Last error: {last_error}"
    ) from last_error
