import asyncio
import json
from asyncio import as_completed
from dataclasses import dataclass
from enum import StrEnum
from math import ceil
from typing import Callable

from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from pydantic import BaseModel, create_model

from cpinto.files import load_yaml
from cpinto.functions import identity_func

from agents7.agents.base import BaseAgent
from agents7.agents.utils import (
    import_from_path,
    get_examples_str,
    model_class_to_string,
)
from agents7.llm import LLMEngine
from agents7.llm.cache import CacheConfig
from agents7.llm.retry import RetryConfig
from agents7.logs import logger
from agents7.models import AgentResponse
from agents7.settings import DEFAULT_MAX_CONCURRENT


@dataclass(frozen=True)
class Labeler(BaseAgent):
    """
    Labeler agent for processing and labeling data.
    This agent is designed to analyze and label data based on predefined rules.

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
            raise TypeError(
                "`output_model` must be a subclass of `pydantic.BaseModel`"
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
        """
        Get the full prompt used by the agent, including the system prompt
        and examples if provided.
        :return: Full prompt string.
        """
        full_prompt = (
            f"{self.resolved_system_prompt}\n\n"
            f"## Output Model Schema\n\n"
            f"{model_class_to_string(self.output_model)}"
        ).strip()

        if self.examples:
            examples_str = get_examples_str(self.examples)
            full_prompt = f"{full_prompt}\n\n{examples_str}"

        return full_prompt

    @property
    def prompt_for_batch(self) -> str:
        """
        Get the full prompt used by the agent, including the system prompt
        and examples if provided. This prompt is used for batch processing.
        :return: Full prompt string.
        """
        full_prompt = (
            f"{self.resolved_system_prompt}\n\n"
            f"## Output Model Schema\n\n"
            f"{model_class_to_string(self.output_model)}\n\n"
            f"You will receive multiple inputs. Process each one independently "
            f"and return an array of outputs in the same order as the inputs."
        ).strip()

        if self.examples:
            examples_str = get_examples_str(self.examples)
            full_prompt = f"{full_prompt}\n\n{examples_str}"

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
        # Set default `chat_kwargs`
        chat_kwargs = chat_kwargs or {"think": True}
        # Ensure certain parameters are set correctly
        for k in ["messages", "temperature", "output_format"]:
            chat_kwargs.pop(k, None)

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
        response["model"] = self._run_validators(
            self._parse_model_response(response["response"])
        )
        return AgentResponse(**response)

    async def act_async(
        self,
        input_data: dict | list[dict],
        chat_kwargs: dict = None,
        _idx: int = 0,
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
        :param _idx: Internal parameter to keep track of the index of the
            input data when processing in batch mode. Default to 0.
            This parameter should not be set by the user.
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
        # Set default `chat_kwargs`
        chat_kwargs = chat_kwargs or {"think": True}
        # Ensure certain parameters are set correctly
        for k in ["messages", "temperature", "output_format"]:
            chat_kwargs.pop(k, None)

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
        response["model"] = self._run_validators(
            self._parse_model_response(response["response"])
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
        """
        Perform the labeling action based on the provided input, in batch mode.
        :param input_list: List of input data to be processed.
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
        :param requests_per_prompt: Number of requests to send per prompt.
            Number of LLM calls will be
                `ceil(len(input_list) / requests_per_prompt)`.
            Defaults to 10.
        :param iter_func: Function to keep track of the iterations.
            Defaults to `tqdm.tqdm`. If `None`, no function will be used.
        :return: A list of AgentResponse models, containing:
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
                        model=self._run_validators(item),
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
        max_concurrent: int | None = DEFAULT_MAX_CONCURRENT,
    ) -> list[AgentResponse]:
        """
        Perform the labeling action based on the provided input, in batch mode.
        :param input_list: List of input data to be processed.
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
        :param requests_per_prompt: Number of requests to send per prompt.
            Number of LLM calls will be
                `ceil(len(input_list) / requests_per_prompt)`.
            Defaults to 10.
        :param with_tqdm: Whether to use `tqdm.asyncio` to keep track of the
            iterations.
        :param max_concurrent: Maximum number of concurrent LLM
            requests. Defaults to ``DEFAULT_MAX_CONCURRENT`` (5).
            ``None`` means unlimited.
        :return: A list of AgentResponse models, containing:
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
        sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

        async def _limited(coro):
            if sem:
                async with sem:
                    return await coro
            return await coro

        input_chunks = [
            input_list[
                i * requests_per_prompt : (i + 1) * requests_per_prompt
            ]
            for i in range(ceil(len(input_list) / requests_per_prompt))
        ]
        tasks = [
            _limited(
                self._act_in_batch_async_request(
                    input_data, chat_kwargs=chat_kwargs, _idx=idx
                )
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
                        model=self._run_validators(item),
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
        """
        Perform the labeling action based on the provided input, in batch mode.
        Will send a single request to the LLM with a list of inputs.
        :param input_list: List of input data to be processed.
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
        :param _idx: Internal parameter to keep track of the index of the
            input data when processing in batch mode. Default to 0.
            This parameter should not be set by the user.
        :return: A list of AgentResponse models, containing:
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
        # Set default `chat_kwargs`
        chat_kwargs = chat_kwargs or {"think": True}
        # Ensure certain parameters are set correctly
        for k in ["messages", "temperature", "output_format"]:
            chat_kwargs.pop(k, None)

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
        response["model"] = self._parse_model_response(
            response["response"], for_batch=True
        )
        response["idx"] = _idx
        return AgentResponse(**response)

    def _act_in_batch_request(
        self,
        input_list: list[dict | list[dict]],
        chat_kwargs: dict = None,
    ) -> AgentResponse:
        """
        Perform the labeling action based on the provided input, in batch mode.
        Will send a single request to the LLM with a list of inputs.
        :param input_list: List of input data to be processed.
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
        # Set default `chat_kwargs`
        chat_kwargs = chat_kwargs or {"think": True}
        # Ensure certain parameters are set correctly
        for k in ["messages", "temperature", "output_format"]:
            chat_kwargs.pop(k, None)

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
        response["model"] = self._parse_model_response(
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

        mandatory_keys = [
            "name",
            "version",
            "engine",
            "system_prompt",
            "output_model",
        ]
        for key in mandatory_keys:
            if key not in config:
                raise ValueError(
                    f"Missing mandatory key `{key}` in the "
                    f"configuration yaml `{file_path}`"
                )
        engine = config["engine"]
        mandatory_keys = ["provider", "model"]
        for key in mandatory_keys:
            if key not in engine:
                raise ValueError(
                    f"Missing mandatory key `{key}` in the engine "
                    f"configuration yaml `{file_path}`"
                )

        retry_config = None
        if retry_settings := engine.get("retry"):
            retry_config = RetryConfig(**retry_settings)

        cache_config = None
        cache_settings = engine.get("cache")
        if cache_settings is not None:
            cache_config = CacheConfig(**cache_settings)

        system_prompt = config["system_prompt"].strip()
        if examples := config.get("system_examples"):
            try:
                get_examples_str(examples)
            except ValueError as e:
                raise ValueError(
                    f"{e} in the configuration file `{file_path}`"
                ) from e

        prompt_params = None
        if raw_params := config.get("prompt_params"):
            prompt_params = {str(k): str(v) for k, v in raw_params.items()}

        validators = None
        if validator_paths := config.get("validators"):
            validators = [import_from_path(path) for path in validator_paths]

        return {
            "name": config["name"],
            "version": config["version"],
            "engine": LLMEngine(
                engine["provider"],
                engine["model"],
                retry=retry_config,
                cache=cache_config,
            ),
            "system_prompt": system_prompt.strip(),
            "prompt_params": prompt_params,
            "output_model": import_from_path(config["output_model"]),
            "examples": examples,
            "validators": validators,
        }

    def _parse_model_response(
        self, response: str, for_batch: bool = False
    ) -> BaseModel | None:
        """
        Parse the response from the agent and validate it against the output
        model.
        :param response: The response string from the agent.
        :param for_batch: Whether the response is for batch processing.
            If True, will use `self.output_model_list`,
            instead of `self.output_model`, to parse the response.
            Defaults to False.
        :return: The structured output validated by the output model.
            Will return `None` if the response cannot be parsed.
        """
        if for_batch:
            model_cls = self.output_model_list
        else:
            model_cls = self.output_model
        model = None

        # Fix common JSON malformations from some models
        response = self._fix_malformed_json(response)

        try:
            model = model_cls.model_validate_json(response)
        except Exception:
            try:
                # Try converting enum names to values for models that
                # return enum names instead of values
                converted = self._convert_enum_names_to_values(response)
                model = model_cls.model_validate_json(converted)
            except Exception as e:
                logger.error(
                    f"Agent `{self.name}`: error parsing structured "
                    f"response:\n{e}"
                )
        return model

    def _run_validators(
        self,
        model: BaseModel | None,
    ) -> BaseModel | None:
        """
        Run the validator pipeline on a parsed model.
        Each validator receives the model and returns it (possibly
        modified). If a validator raises, the model is rejected
        (returns None) and remaining validators are skipped.
        :param model: The parsed model instance, or None.
        :return: The validated (possibly transformed) model, or None.
        """
        if model is None or not self.validators:
            return model
        for validator in self.validators:
            try:
                model = validator(model)
            except Exception as e:
                logger.error(
                    f"Agent `{self.name}`: validator "
                    f"`{validator.__name__}` failed: {e}"
                )
                return None
        return model

    @staticmethod
    def _fix_malformed_json(response: str) -> str:
        """
        Fix common JSON malformations from LLM outputs.
        Some models return malformed JSON like '{"{"reason":...' instead
        of '{"reason":...'.
        """
        response = response.strip()

        # Remove markdown code blocks: ```json ... ``` or ``` ... ```
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's closing ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines).strip()

        # Fix double opening brace: {"{ -> {
        if response.startswith('{"{'):
            response = response[2:]

        # Fix double opening bracket for arrays: ["[ -> [
        if response.startswith('["['):
            response = response[2:]

        return response

    def _get_enum_mappings(self) -> dict[str, dict[str, str]]:
        """
        Get enum name-to-value mappings for all StrEnum fields in output_model.
        :return: Dict mapping field names to their enum name->value dicts.
        """
        enum_mappings = {}
        for field_name, field_info in self.output_model.model_fields.items():
            field_type = field_info.annotation
            # Handle Optional types
            if hasattr(field_type, "__origin__"):
                args = getattr(field_type, "__args__", ())
                for arg in args:
                    if isinstance(arg, type) and issubclass(arg, StrEnum):
                        field_type = arg
                        break
            if isinstance(field_type, type) and issubclass(
                field_type, StrEnum
            ):
                enum_mappings[field_name] = {
                    member.name: member.value for member in field_type
                }
        return enum_mappings

    def _convert_enum_names_to_values(self, response: str) -> str:
        """
        Convert enum names to values in a JSON response string.
        Some models return enum names (e.g., 'TOO_COLD') instead of
        values (e.g., 'The temperature is too cold.').
        """
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return response

        enum_mappings = self._get_enum_mappings()
        data = self._apply_enum_mappings(data, enum_mappings)
        return json.dumps(data)

    @staticmethod
    def _apply_enum_mappings(
        data: dict | list, mappings: dict[str, dict[str, str]]
    ) -> dict | list:
        """
        Apply enum name-to-value mappings to data.
        Handles both single objects and lists.
        """
        if isinstance(data, list):
            return [
                Labeler._apply_enum_mappings_to_item(item, mappings)
                for item in data
            ]
        return Labeler._apply_enum_mappings_to_item(data, mappings)

    @staticmethod
    def _apply_enum_mappings_to_item(
        item: dict, mappings: dict[str, dict[str, str]]
    ) -> dict:
        """Apply enum name-to-value mappings to a single item."""
        if not isinstance(item, dict):
            return item
        for field_name, mapping in mappings.items():
            if field_name in item and item[field_name] in mapping:
                item[field_name] = mapping[item[field_name]]
        return item
