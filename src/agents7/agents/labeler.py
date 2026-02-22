import asyncio
from asyncio import as_completed
from dataclasses import dataclass
from math import ceil
from typing import Callable

from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from pydantic import BaseModel, create_model

from cpinto.files import load_yaml
from cpinto.functions import identity_func

from agents7.agents.base import BaseAgent
from agents7.agents.parsing import ResponseParser
from agents7.agents.utils import (
    get_examples_str,
    import_from_path,
    model_class_to_string,
    run_with_semaphore,
)
from agents7.exceptions import AgentConfigurationError
from agents7.llm import LLMEngine
from agents7.llm.cache import CacheConfig
from agents7.llm.retry import RetryConfig
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
    """

    output_model: type[BaseModel]
    examples: list[dict] = None
    validators: list[Callable[[BaseModel], BaseModel]] | None = None

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
        object.__setattr__(
            self,
            "_parser",
            ResponseParser(
                output_model=self.output_model,
                output_model_list=self.output_model_list,
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
            The fields `prompt_params` and `system_examples` are optional.
        :return: An instance of the agent.
        """
        config = cls._load_yaml_settings(file_path)
        return Labeler(**config)

    @property
    def prompt(self) -> str:
        """Get the full prompt used by the agent.
        :return: Full prompt string.
        """
        return self._build_prompt()

    @property
    def prompt_for_batch(self) -> str:
        """Get the full prompt for batch processing.
        :return: Full prompt string.
        """
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
        """
        Get a dynamic model that wraps the output_model in a list.
        :return: A dynamic pydantic model class with a single field `items`
            which is a list of `self.output_model`.
        """
        # Create a dynamic model that wraps the output_model in a list
        return create_model(
            f"{self.output_model.__name__}List",
            items=(list[self.output_model], ...),
            __base__=BaseModel,
        )

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
            output_format=self.output_model.model_json_schema(),
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
        chat_kwargs: dict = None,
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
            output_format=self.output_model.model_json_schema(),
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
        chat_kwargs: dict = None,
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
        input_chunks = [
            input_list[
                i * requests_per_prompt : (i + 1) * requests_per_prompt
            ]
            for i in range(ceil(len(input_list) / requests_per_prompt))
        ]
        agent_responses = []
        for input_data in iter_func(input_chunks):
            chunk_response = self._act_in_batch_request(
                input_data, chat_kwargs=chat_kwargs
            )
            # Flatten: create individual AgentResponse for each item
            for item in chunk_response.model.items:
                agent_responses.append(
                    AgentResponse(
                        response=chunk_response.response,
                        model=self._parser.run_validators(item),
                        thinking=chunk_response.thinking,
                        metrics=chunk_response.metrics,
                        idx=len(agent_responses),
                    )
                )
        return agent_responses

    async def act_in_batch_async(
        self,
        input_list: list[dict | list[dict]],
        chat_kwargs: dict = None,
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

        input_chunks = [
            input_list[
                i * requests_per_prompt : (i + 1) * requests_per_prompt
            ]
            for i in range(ceil(len(input_list) / requests_per_prompt))
        ]
        tasks = [
            run_with_semaphore(
                self._act_in_batch_async_request(
                    input_data, chat_kwargs=chat_kwargs, _idx=idx
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
                desc=f"Agent `{self.name}`: acting in batch",
            ):
                response = await coro
                chunk_responses[response.idx] = response
        else:
            for coro in as_completed(tasks):
                response = await coro
                chunk_responses[response.idx] = response

        # Flatten: create individual AgentResponse for each item
        agent_responses = []
        for chunk_response in chunk_responses:
            for item in chunk_response.model.items:
                agent_responses.append(
                    AgentResponse(
                        response=chunk_response.response,
                        model=self._parser.run_validators(item),
                        thinking=chunk_response.thinking,
                        metrics=chunk_response.metrics,
                        idx=len(agent_responses),
                    )
                )
        return agent_responses

    async def _act_in_batch_async_request(
        self,
        input_list: list[dict | list[dict]],
        chat_kwargs: dict = None,
        _idx: int = 0,
    ) -> AgentResponse:
        """Send a single async batch request to the LLM.

        :param input_list: List of input data to process.
        :param chat_kwargs: See `act()` for details.
        :param _idx: Internal index for batch tracking.
        :return: An `AgentResponse` wrapping the batch
            output model.
        """
        chat_kwargs = self._sanitize_chat_kwargs(chat_kwargs)

        logger.debug(
            f"Agent `{self.name}`: processing batch input data:\n{input_list}"
        )
        response = await self.engine.chat_async(
            messages=[
                {"role": "system", "content": self.prompt_for_batch},
                {
                    "role": "user",
                    "content": f"Process the following:\n{input_list}",
                },
            ],
            temperature=0,
            output_format=self.output_model_list.model_json_schema(),
            **chat_kwargs,
        )
        response.pop("role")
        response["model"] = self._parser.parse(
            response["response"], for_batch=True
        )
        response["idx"] = _idx
        return AgentResponse(**response)

    def _act_in_batch_request(
        self,
        input_list: list[dict | list[dict]],
        chat_kwargs: dict = None,
    ) -> AgentResponse:
        """Send a single sync batch request to the LLM.

        :param input_list: List of input data to process.
        :param chat_kwargs: See `act()` for details.
        :return: An `AgentResponse` wrapping the batch
            output model.
        """
        chat_kwargs = self._sanitize_chat_kwargs(chat_kwargs)

        logger.debug(
            f"Agent `{self.name}`: processing batch input data:\n{input_list}"
        )
        response = self.engine.chat(
            messages=[
                {"role": "system", "content": self.prompt_for_batch},
                {
                    "role": "user",
                    "content": f"Process the following:\n{input_list}",
                },
            ],
            temperature=0,
            output_format=self.output_model_list.model_json_schema(),
            **chat_kwargs,
        )
        response.pop("role")
        response["model"] = self._parser.parse(
            response["response"], for_batch=True
        )
        return AgentResponse(**response)

    @classmethod
    def _load_yaml_settings(cls, file_path: str) -> dict:
        """
        Load settings from a YAML file.
        :param file_path: The path to the YAML file. Schema:
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
            The fields `prompt_params` and `system_examples` are
            optional.
        :return: Loaded configuration, as a dictionary. Keys:
            - name (str): The name of the agent.
            - version (int): The version of the agent.
            - engine (str): The LLM engine used by the agent.
            - system_prompt (str): The system prompt for the agent.
            - prompt_params (dict, optional): Dictionary of parameters
                to substitute into system_prompt.
            - output_model (BaseModel): The model used to validate and
                structure the output.
            - examples (list[dict], optional): List of examples.
        """
        config = load_yaml(file_path)
        cls._validate_yaml_config(config, file_path)

        examples = cls._parse_yaml_examples(config, file_path)

        prompt_params = None
        if raw_params := config.get("prompt_params"):
            prompt_params = {str(k): str(v) for k, v in raw_params.items()}

        validators = None
        if validator_paths := config.get("validators"):
            validators = [import_from_path(path) for path in validator_paths]

        return {
            "name": config["name"],
            "version": config["version"],
            "engine": cls._build_engine(config["engine"]),
            "system_prompt": config["system_prompt"].strip(),
            "prompt_params": prompt_params,
            "output_model": import_from_path(config["output_model"]),
            "examples": examples,
            "validators": validators,
        }

    @staticmethod
    def _validate_yaml_config(config: dict, file_path: str) -> None:
        """
        Validate mandatory keys in a YAML configuration.
        :param config: Parsed YAML configuration dictionary.
        :param file_path: Path to the YAML file (for error
            messages).
        :raises AgentConfigurationError: If a mandatory key is
            missing at the top level or inside the engine section.
        """
        for key in (
            "name",
            "version",
            "engine",
            "system_prompt",
            "output_model",
        ):
            if key not in config:
                raise AgentConfigurationError(
                    f"Missing mandatory key `{key}` in the "
                    f"configuration yaml `{file_path}`"
                )
        engine = config["engine"]
        for key in ("provider", "model"):
            if key not in engine:
                raise AgentConfigurationError(
                    f"Missing mandatory key `{key}` in the "
                    f"engine configuration yaml "
                    f"`{file_path}`"
                )

    @staticmethod
    def _build_engine(engine_config: dict) -> LLMEngine:
        """
        Build an `LLMEngine` from the engine section of YAML config.
        :param engine_config: Dictionary with keys `provider`,
            `model`, and optional `retry` / `cache` sections.
        :return: Configured `LLMEngine` instance.
        """
        retry_config = None
        if retry_settings := engine_config.get("retry"):
            retry_config = RetryConfig(**retry_settings)

        cache_config = None
        cache_settings = engine_config.get("cache")
        if cache_settings is not None:
            cache_config = CacheConfig(**cache_settings)

        return LLMEngine(
            engine_config["provider"],
            engine_config["model"],
            retry=retry_config,
            cache=cache_config,
        )

    @staticmethod
    def _parse_yaml_examples(
        config: dict, file_path: str
    ) -> list[dict] | None:
        """
        Extract and validate examples from YAML config.
        :param config: Parsed YAML configuration dictionary.
        :param file_path: Path to the YAML file (for error
            messages).
        :return: List of example dicts, or `None` if absent.
        :raises AgentConfigurationError: If examples are malformed.
        """
        examples = config.get("system_examples")
        if examples is None:
            return None
        try:
            get_examples_str(examples)
        except (ValueError, AgentConfigurationError) as e:
            raise AgentConfigurationError(
                f"{e} in the configuration file `{file_path}`"
            ) from e
        return examples

    @staticmethod
    def _sanitize_chat_kwargs(chat_kwargs: dict | None) -> dict:
        """
        Prepare `chat_kwargs` for an LLM call.
        Sets `think=True` by default and removes keys managed
        internally (`messages`, `temperature`, `output_format`).
        :param chat_kwargs: Raw keyword arguments, or None.
        :return: Sanitized copy of the keyword arguments.
        """
        chat_kwargs = chat_kwargs or {"think": True}
        for k in ["messages", "temperature", "output_format"]:
            chat_kwargs.pop(k, None)
        return chat_kwargs
