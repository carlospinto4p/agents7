import asyncio
from asyncio import as_completed
from dataclasses import dataclass, field
from math import ceil
from typing import Callable

from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from pydantic import BaseModel, create_model

from cpinto.functions import identity_func

from agents7.agents.base import BaseAgent
from agents7.agents.config_loader import load_yaml_settings
from agents7.agents.parsing import ResponseParser
from agents7.agents.utils import (
    get_examples_str,
    model_class_to_string,
    run_with_semaphore,
)
from agents7.exceptions import AgentConfigurationError
from agents7.logs import logger
from agents7.models import AgentResponse
from agents7.settings import BatchSettings


@dataclass(frozen=True)
class Labeler(BaseAgent):
    """
    Labeler agent for processing and labeling data.
    This agent is designed to analyze and label data based on
    predefined rules.

    Attributes:
        name (str): The name of the agent.
        version (int): The version of the agent.
        engine (LLMEngine): The LLM engine used by the agent.
        system_prompt (str): The system prompt for the agent.
        output_model (BaseModel): The model used to validate and
            structure the output.
        examples (list[dict], optional): List of examples to include in the
            system prompt. Each example should be a dictionary with 'input'
            and 'output' keys. Defaults to None.
        validators (list[Callable], optional): List of callables that
            run after the LLM response is parsed. Each receives a
            BaseModel and returns it (possibly modified). Raising
            rejects the model (sets it to None). Defaults to None.
        think (bool, optional): Whether to enable thinking mode
            for LLM calls. Defaults to True. Can be overridden
            per call via `chat_kwargs`.
    """

    output_model: type[BaseModel]
    examples: list[dict] | None = None
    validators: list[Callable[[BaseModel], BaseModel]] | None = None
    think: bool = field(default=True, kw_only=True)

    def __post_init__(self):
        """
        Post-initialization method to validate the agent.
        This method is called after the agent is initialized.
        """
        super().__post_init__()
        if not issubclass(self.output_model, BaseModel):
            raise AgentConfigurationError(
                "`output_model` must be a subclass of `pydantic.BaseModel`"
            )
        output_model_list = create_model(
            f"{self.output_model.__name__}List",
            items=(list[self.output_model], ...),
            __base__=BaseModel,
        )
        object.__setattr__(self, "_output_model_list", output_model_list)
        object.__setattr__(
            self,
            "_output_schema",
            self.output_model.model_json_schema(),
        )
        object.__setattr__(
            self,
            "_output_list_schema",
            output_model_list.model_json_schema(),
        )
        object.__setattr__(
            self,
            "_parser",
            ResponseParser(
                output_model=self.output_model,
                output_model_list=output_model_list,
                name=self.name,
                validators=self.validators,
            ),
        )

    @classmethod
    def from_yaml(cls, file_path: str) -> "Labeler":
        """
        Create an instance of the agent from a YAML file configuration.
        This method should be overridden by subclasses to provide specific
        YAML loading logic.

        :param file_path: The path to the YAML file containing the agent
            configuration. Schema:
            ```yaml
            name: <name of the agent>
            version: <version of the agent>
            engine:
                provider: <llm provider>
                model: <llm model>
            system_prompt: <system prompt for the agent>
            prompt_params:
                <key>: <value to substitute $key in system_prompt>
                ...
            system_examples:
                - input:
                    <example 1 input>
                  output:
                    <example 1 output>
                ...
            output_model: <path to the pydantic model class, e.g.
                "agents7.models.MyOutputModel">
            ```
            The fields `prompt_params`, `system_examples`, and
            `think` are optional.
        :return: An instance of the agent.
        """
        config = load_yaml_settings(file_path)
        return Labeler(**config)

    @property
    def prompt(self) -> str:
        """Full prompt used by the agent."""
        return self._build_prompt()

    @property
    def prompt_for_batch(self) -> str:
        """Full prompt for batch processing."""
        return self._build_prompt(
            "You will receive multiple inputs. Process "
            "each one independently and return an array "
            "of outputs in the same order as the inputs."
        )

    def _build_prompt(
        self,
        instruction: str = "",
        extra_sections: list[str] | None = None,
    ) -> str:
        """Build the full prompt from system prompt, schema,
        instruction, extra sections, and examples.

        Subclasses pass their agent-specific instruction
        and optional extra sections (e.g., constraints).

        :param instruction: Agent-specific instruction
            appended after the output model schema.
        :param extra_sections: Additional text sections
            inserted between instruction and examples.
        :return: Full prompt string.
        """
        full_prompt = (
            f"{self.resolved_system_prompt}\n\n"
            f"## Output Model Schema\n\n"
            f"{model_class_to_string(self.output_model)}"
        )
        if instruction:
            full_prompt = f"{full_prompt}\n\n{instruction}"
        full_prompt = full_prompt.strip()

        for section in extra_sections or []:
            if section:
                full_prompt = f"{full_prompt}\n\n{section}"

        if self.examples:
            full_prompt = (
                f"{full_prompt}\n\n{get_examples_str(self.examples)}"
            )

        return full_prompt

    @property
    def output_model_list(self) -> type[BaseModel]:
        """Dynamic model wrapping ``output_model`` in a list."""
        return self._output_model_list

    def act(
        self, input_data: dict | list[dict], chat_kwargs: dict = None
    ) -> AgentResponse:
        """
        Perform the labeling action based on the provided input.
        :param input_data: The input data to be processed.
        :param chat_kwargs: Dictionary with extra keyword arguments to
            send to the `self.engine.chat` method. Note that:
            - The parameter `temperature` is set to 0 for agents, and
              won't be overridden by this parameter.
            - The parameter `think` is set to True by default, unless
              specified otherwise.
            - Available parameters:
                - max_tokens: The maximum number of tokens to generate in the
                    response.
                - think: If True, the model will think before generating the
                    response.
                - seed: Seed for random number generation. Default to None.
        :return: An AgentResponse model, containing:
            - response: The generated text response.
            - model: The structured output validated by the output model.
            - thinking: The thinking part of the response, if any.
            - metrics: LLMMetrics model, containing:
                - tokens: The number of tokens in the response.
                - prompt_tokens: The number of tokens in the prompt.
                - total_duration: The total duration of the response,
                 in seconds.
            - idx: Optional index for tracking purposes.
        """
        chat_kwargs = self._sanitize_chat_kwargs(chat_kwargs)

        logger.debug(
            f"Agent `{self.name}`: processing input data:\n{input_data}"
        )
        response = self.engine.chat(
            messages=[
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": f"Process the following:\n{input_data}",
                },
            ],
            temperature=0,
            output_format=self._output_schema,
            **chat_kwargs,
        )
        response.pop("role")
        response["model"] = self._parser.run_validators(
            self._parser.parse(response["response"])
        )
        return AgentResponse(**response)

    async def act_async(
        self,
        input_data: dict | list[dict],
        chat_kwargs: dict | None = None,
        _idx: int = 0,
    ) -> AgentResponse:
        """Async version of `act()`.

        :param input_data: The input data to be processed.
        :param chat_kwargs: See `act()` for details.
        :param _idx: Internal index for batch tracking.
            Should not be set by the user.
        :return: See `act()` for return schema.
        """
        chat_kwargs = self._sanitize_chat_kwargs(chat_kwargs)

        logger.debug(
            f"Agent `{self.name}`: processing input data:\n{input_data}"
        )
        response = await self.engine.chat_async(
            messages=[
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": f"Process the following:\n{input_data}",
                },
            ],
            temperature=0,
            output_format=self._output_schema,
            **chat_kwargs,
        )
        response.pop("role")
        response["model"] = self._parser.run_validators(
            self._parser.parse(response["response"])
        )
        response["idx"] = _idx
        return AgentResponse(**response)

    def act_in_batch(
        self,
        input_list: list[dict | list[dict]],
        chat_kwargs: dict | None = None,
        requests_per_prompt: int = 10,
        iter_func: Callable = tqdm,
    ) -> list[AgentResponse]:
        """Sync batch processing.

        :param input_list: List of input data to process.
        :param chat_kwargs: See `act()` for details.
        :param requests_per_prompt: Items per LLM call.
            Defaults to 10.
        :param iter_func: Progress iterator. Defaults to
            `tqdm.tqdm`. Pass `None` to disable.
        :return: List of `AgentResponse`, one per input.
            See `act()` for the response schema.
        """
        if iter_func is None:
            iter_func = identity_func
        input_chunks = self._chunk_inputs(input_list, requests_per_prompt)
        chunk_responses = [
            self._act_in_batch_request(input_data, chat_kwargs=chat_kwargs)
            for input_data in iter_func(input_chunks)
        ]
        return self._flatten_batch_responses(chunk_responses)

    async def act_in_batch_async(
        self,
        input_list: list[dict | list[dict]],
        chat_kwargs: dict | None = None,
        requests_per_prompt: int = 10,
        with_tqdm: bool = True,
        batch_settings: BatchSettings | None = None,
    ) -> list[AgentResponse]:
        """Async batch processing.

        :param input_list: List of input data to process.
        :param chat_kwargs: See `act()` for details.
        :param requests_per_prompt: Items per LLM call.
            Defaults to 10.
        :param with_tqdm: Show async progress bar.
        :param batch_settings: Optional `BatchSettings`
            controlling concurrency. Defaults to
            `BatchSettings()` (max 5 concurrent).
        :return: List of `AgentResponse`, one per input.
            See `act()` for the response schema.
        """
        settings = batch_settings or BatchSettings()
        sem = (
            asyncio.Semaphore(settings.max_concurrent)
            if settings.max_concurrent
            else None
        )

        input_chunks = self._chunk_inputs(input_list, requests_per_prompt)
        tasks = [
            run_with_semaphore(
                self._act_in_batch_async_request(
                    input_data,
                    chat_kwargs=chat_kwargs,
                    _idx=idx,
                ),
                sem,
            )
            for idx, input_data in enumerate(input_chunks)
        ]

        # Collect chunk responses in order
        chunk_responses: list[AgentResponse | None] = [None] * len(tasks)
        if with_tqdm:
            for coro in async_tqdm.as_completed(
                tasks,
                total=len(tasks),
                desc=(f"Agent `{self.name}`: acting in batch"),
            ):
                response = await coro
                chunk_responses[response.idx] = response
        else:
            for coro in as_completed(tasks):
                response = await coro
                chunk_responses[response.idx] = response

        return self._flatten_batch_responses(chunk_responses)

    async def _act_in_batch_async_request(
        self,
        input_list: list[dict | list[dict]],
        chat_kwargs: dict | None = None,
        _idx: int = 0,
    ) -> AgentResponse:
        """Send a single async batch request to the LLM.

        :param input_list: List of input data to process.
        :param chat_kwargs: See `act()` for details.
        :param _idx: Internal index for batch tracking.
        :return: An `AgentResponse` wrapping the batch
            output model.
        """
        params = self._batch_chat_params(input_list, chat_kwargs)
        response = self._parse_batch_response(
            await self.engine.chat_async(**params)
        )
        response["idx"] = _idx
        return AgentResponse(**response)

    def _act_in_batch_request(
        self,
        input_list: list[dict | list[dict]],
        chat_kwargs: dict | None = None,
    ) -> AgentResponse:
        """Send a single sync batch request to the LLM.

        :param input_list: List of input data to process.
        :param chat_kwargs: See `act()` for details.
        :return: An `AgentResponse` wrapping the batch
            output model.
        """
        params = self._batch_chat_params(input_list, chat_kwargs)
        response = self._parse_batch_response(self.engine.chat(**params))
        return AgentResponse(**response)

    @staticmethod
    def _chunk_inputs(
        input_list: list[dict | list[dict]],
        requests_per_prompt: int,
    ) -> list[list[dict | list[dict]]]:
        """Split an input list into chunks."""
        return [
            input_list[
                i * requests_per_prompt : (i + 1) * requests_per_prompt
            ]
            for i in range(ceil(len(input_list) / requests_per_prompt))
        ]

    def _batch_chat_params(
        self,
        input_list: list[dict | list[dict]],
        chat_kwargs: dict | None = None,
    ) -> dict:
        """Build kwargs for a batch LLM chat call."""
        chat_kwargs = self._sanitize_chat_kwargs(chat_kwargs)
        logger.debug(
            f"Agent `{self.name}`: processing batch input data:\n{input_list}"
        )
        return {
            "messages": [
                {
                    "role": "system",
                    "content": self.prompt_for_batch,
                },
                {
                    "role": "user",
                    "content": (f"Process the following:\n{input_list}"),
                },
            ],
            "temperature": 0,
            "output_format": self._output_list_schema,
            **chat_kwargs,
        }

    def _parse_batch_response(self, response: dict) -> dict:
        """Parse a raw batch LLM response dict."""
        response.pop("role")
        response["model"] = self._parser.parse(
            response["response"], for_batch=True
        )
        return response

    def _flatten_batch_responses(
        self,
        chunk_responses: list[AgentResponse],
    ) -> list[AgentResponse]:
        """Flatten batch chunk responses into individual
        AgentResponses."""
        agent_responses = []
        for chunk in chunk_responses:
            for item in chunk.model.items:
                agent_responses.append(
                    AgentResponse(
                        response=chunk.response,
                        model=self._parser.run_validators(item),
                        thinking=chunk.thinking,
                        metrics=chunk.metrics,
                        idx=len(agent_responses),
                    )
                )
        return agent_responses

    def _sanitize_chat_kwargs(self, chat_kwargs: dict | None) -> dict:
        """
        Prepare `chat_kwargs` for an LLM call.
        Sets `think` to the agent's default and removes keys
        managed internally (`messages`, `temperature`,
        `output_format`). Users can override `think` via
        `chat_kwargs`.
        :param chat_kwargs: Raw keyword arguments, or None.
        :return: Sanitized copy of the keyword arguments.
        """
        chat_kwargs = dict(chat_kwargs) if chat_kwargs else {}
        chat_kwargs.setdefault("think", self.think)
        for k in ["messages", "temperature", "output_format"]:
            chat_kwargs.pop(k, None)
        return chat_kwargs
