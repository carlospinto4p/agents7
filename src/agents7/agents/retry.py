from dataclasses import dataclass

from pydantic import BaseModel

from agents7.agents.parsing import ResponseParser
from agents7.exceptions import ParseError
from agents7.llm.base import LLMEngine
from agents7.logs import logger


@dataclass(frozen=True)
class ParseRetryHandler:
    """Handles parse-retry logic for a single LLM agent.

    Retries a failed parse by feeding the error back to the
    model and re-prompting up to ``max_retries`` times.

    :param engine: LLM engine used for retry calls.
    :param parser: ResponseParser used to parse responses.
    :param name: Agent name (used in log messages).
    :param max_retries: Maximum number of retry attempts.
    :param output_schema: JSON schema dict for the output
        model, passed as ``output_format`` to the engine.
    """

    engine: LLMEngine
    parser: ResponseParser
    name: str
    max_retries: int
    output_schema: dict

    def retry_sync(
        self,
        error: ParseError,
        messages: list[dict[str, str]],
        chat_kwargs: dict,
    ) -> "BaseModel | None":
        """Retry a failed parse synchronously.

        :param error: The ParseError from the first attempt.
        :param messages: Original messages sent to the LLM.
        :param chat_kwargs: Sanitized chat kwargs.
        :return: Parsed model on success, None if all retries
            fail.
        """
        if self.max_retries <= 0:
            return None
        retry_messages = list(messages)
        last_error = error
        for attempt in range(self.max_retries):
            self._log_retry(attempt, last_error)
            retry_messages.append(self._retry_message(last_error))
            raw = self.engine.chat(
                **self._chat_kwargs(retry_messages, chat_kwargs)
            )
            parsed, last_error = self._try_parse(raw)
            if parsed is not None:
                return parsed
        return None

    async def retry_async(
        self,
        error: ParseError,
        messages: list[dict[str, str]],
        chat_kwargs: dict,
    ) -> "BaseModel | None":
        """Retry a failed parse asynchronously.

        :param error: The ParseError from the first attempt.
        :param messages: Original messages sent to the LLM.
        :param chat_kwargs: Sanitized chat kwargs.
        :return: Parsed model on success, None if all retries
            fail.
        """
        if self.max_retries <= 0:
            return None
        retry_messages = list(messages)
        last_error = error
        for attempt in range(self.max_retries):
            self._log_retry(attempt, last_error)
            retry_messages.append(self._retry_message(last_error))
            raw = await self.engine.chat_async(
                **self._chat_kwargs(retry_messages, chat_kwargs)
            )
            parsed, last_error = self._try_parse(raw)
            if parsed is not None:
                return parsed
        return None

    def _log_retry(self, attempt: int, error: ParseError) -> None:
        logger.warning(
            "Agent `%s`: parse retry %d/%d after error: %s",
            self.name,
            attempt + 1,
            self.max_retries,
            error,
        )

    def _chat_kwargs(
        self,
        messages: list[dict[str, str]],
        chat_kwargs: dict,
    ) -> dict:
        return {
            "messages": messages,
            "temperature": 0,
            "output_format": self.output_schema,
            **chat_kwargs,
        }

    def _try_parse(
        self,
        raw_response: dict,
    ) -> "tuple[BaseModel | None, ParseError | None]":
        raw_response.pop("role", None)
        try:
            return self.parser.parse(raw_response["response"]), None
        except ParseError as e:
            return None, e

    @staticmethod
    def _retry_message(error: ParseError) -> dict[str, str]:
        return {
            "role": "user",
            "content": (
                "Your previous response could not be parsed. "
                f"Error: {error}\n"
                "Please respond again with valid JSON matching "
                "the schema exactly."
            ),
        }
