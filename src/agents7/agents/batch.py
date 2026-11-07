import asyncio
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import ceil
from typing import Callable

from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from agents7.agents.parsing import ResponseParser
from agents7.agents.utils import run_with_semaphore
from agents7.exceptions import ParseError
from agents7.llm.base import LLMEngine
from agents7.logs import logger
from agents7.models import AgentResponse
from agents7.settings import BatchSettings


@dataclass(frozen=True)
class BatchProcessor:
    """Handles batch LLM calls for a Labeler agent.

    Encapsulates chunking, prompting, parsing, and
    flattening for both sync and async batch execution.

    :param engine: LLM engine used for batch calls.
    :param parser: ResponseParser used to parse responses.
    :param name: Agent name (used in log messages).
    :param preprocessor: Optional callable applied to each
        input item before batching.
    :param output_list_schema: JSON schema dict for the
        list-wrapped output model.
    :param on_call: Optional callable invoked with each
        chunk-level `AgentResponse` — one per real LLM call,
        not one per input item. Used to emit call-log records
        for batch runs. Called as
        ``on_call(response, messages=messages)``, so the
        chunk's prompt reaches the record when it failed.
    """

    engine: LLMEngine
    parser: ResponseParser
    name: str
    preprocessor: Callable | None
    output_list_schema: dict
    on_call: Callable | None = None

    def act_in_batch(
        self,
        input_list: list,
        batch_prompt: str,
        chat_kwargs: dict | None = None,
        requests_per_prompt: int = 10,
        iter_func: Callable = tqdm,
        batch_settings: BatchSettings | None = None,
    ) -> list[AgentResponse]:
        """Sync batch processing with parallel chunk execution.

        Chunks are dispatched concurrently via a thread pool;
        results are re-ordered to match chunk submission order.

        :param input_list: List of input items.
        :param batch_prompt: System prompt for batch calls.
        :param chat_kwargs: Sanitized chat kwargs.
        :param requests_per_prompt: Items per LLM call.
        :param iter_func: Progress iterator (tqdm or None).
        :param batch_settings: Concurrency settings
            (max_concurrent controls thread pool size).
        :return: Flat list of AgentResponse, one per input.
        """
        if iter_func is None:
            iter_func = lambda x: x  # noqa: E731
        settings = batch_settings or BatchSettings()
        chunks = self._chunk_inputs(input_list, requests_per_prompt)
        max_workers = settings.max_concurrent or None

        def _req(chunk: list) -> AgentResponse:
            return self._request(chunk, batch_prompt, chat_kwargs)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            chunk_responses = list(iter_func(pool.map(_req, chunks)))
        return self._flatten(chunk_responses)

    async def act_in_batch_async(
        self,
        input_list: list,
        batch_prompt: str,
        chat_kwargs: dict | None = None,
        requests_per_prompt: int = 10,
        with_tqdm: bool = True,
        batch_settings: BatchSettings | None = None,
    ) -> list[AgentResponse]:
        """Async batch processing.

        :param input_list: List of input items.
        :param batch_prompt: System prompt for batch calls.
        :param chat_kwargs: Sanitized chat kwargs.
        :param requests_per_prompt: Items per LLM call.
        :param with_tqdm: Show async progress bar.
        :param batch_settings: Concurrency settings.
        :return: Flat list of AgentResponse, one per input.
        """
        settings = batch_settings or BatchSettings()
        sem = (
            asyncio.Semaphore(settings.max_concurrent)
            if settings.max_concurrent
            else None
        )
        chunks = self._chunk_inputs(input_list, requests_per_prompt)
        tasks = [
            run_with_semaphore(
                self._request_async(
                    chunk, batch_prompt, chat_kwargs, _idx=idx
                ),
                sem,
            )
            for idx, chunk in enumerate(chunks)
        ]
        chunk_responses: list[AgentResponse | None] = [None] * len(tasks)
        if with_tqdm:
            for coro in async_tqdm.as_completed(
                tasks,
                total=len(tasks),
                desc=f"Agent `{self.name}`: acting in batch",
            ):
                response = await coro
                chunk_responses[response.idx] = response
        else:
            for coro in as_completed(tasks):
                response = await coro
                chunk_responses[response.idx] = response
        return self._flatten(chunk_responses)

    def _request(
        self,
        chunk: list,
        batch_prompt: str,
        chat_kwargs: dict | None,
    ) -> AgentResponse:
        params = self._chat_params(chunk, batch_prompt, chat_kwargs)
        response = self._parse_response(self.engine.chat(**params))
        return self._emit(AgentResponse(**response), params["messages"])

    async def _request_async(
        self,
        chunk: list,
        batch_prompt: str,
        chat_kwargs: dict | None,
        _idx: int = 0,
    ) -> AgentResponse:
        params = self._chat_params(chunk, batch_prompt, chat_kwargs)
        response = self._parse_response(
            await self.engine.chat_async(**params)
        )
        response["idx"] = _idx
        return self._emit(AgentResponse(**response), params["messages"])

    def _emit(
        self,
        response: AgentResponse,
        messages: list[dict[str, str]],
    ) -> AgentResponse:
        """Notify `on_call` about one completed chunk request.

        :param response: The chunk-level response.
        :param messages: The messages this chunk sent, passed on
            so a failed chunk's record can carry its prompt.
        :return: The same response, for call chaining.
        """
        if self.on_call is not None:
            self.on_call(response, messages=messages)
        return response

    def _chat_params(
        self,
        chunk: list,
        batch_prompt: str,
        chat_kwargs: dict | None,
    ) -> dict:
        if self.preprocessor is not None:
            chunk = [self.preprocessor(item) for item in chunk]
        logger.debug(
            "Agent `%s`: processing batch input data:\n%s",
            self.name,
            chunk,
        )
        return {
            "messages": [
                {"role": "system", "content": batch_prompt},
                {
                    "role": "user",
                    "content": f"Process the following:\n{chunk}",
                },
            ],
            "temperature": 0,
            "output_format": self.output_list_schema,
            **(chat_kwargs or {}),
        }

    def _parse_response(self, response: dict) -> dict:
        response.pop("role", None)
        try:
            response["model"] = self.parser.parse(
                response["response"], for_batch=True
            )
        except ParseError:
            response["model"] = None
        return response

    def _flatten(
        self,
        chunk_responses: list[AgentResponse],
    ) -> list[AgentResponse]:
        agent_responses: list[AgentResponse] = []
        for chunk in chunk_responses:
            if chunk.model is None:
                logger.warning(
                    "Agent `%s`: skipping malformed chunk response",
                    self.name,
                )
                continue
            for item in chunk.model.items:
                agent_responses.append(
                    AgentResponse(
                        response=chunk.response,
                        model=self.parser.run_validators(item),
                        thinking=chunk.thinking,
                        metrics=chunk.metrics,
                        idx=len(agent_responses),
                    )
                )
        return agent_responses

    @staticmethod
    def _chunk_inputs(
        input_list: list,
        requests_per_prompt: int,
    ) -> list[list]:
        return [
            input_list[
                i * requests_per_prompt : (i + 1) * requests_per_prompt
            ]
            for i in range(ceil(len(input_list) / requests_per_prompt))
        ]
