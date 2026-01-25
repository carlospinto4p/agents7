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
from agents7.agents.utils import (
    import_from_path,
    get_examples_str,
    model_class_to_string,
)
from agents7.llm import LLMEngine
from agents7.logs import logger
from agents7.models import AgentResponse


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
    """

    output_model: type[BaseModel]
    examples: list[dict] = None

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

    # TODO 2025.08.17 How can I include parameters in the prompts,
    #  like the current date?
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
            system_examples:
                - input:
                    <example 1 input, a dictionary or a list of dictionaries>
                  output:
                    <example 1 output, a dictionary or a list of dictionaries>
                - input:
                    <example 2 input, a dictionary or a list of dictionaries>
                  output:
                    <example 2 output, a dictionary or a list of dictionaries>
                ...
            output_model: <path to the pydantic model class, e.g.
                "agents7.models.MyOutputModel">
            ```
            The field `system_examples` is optional and can be used to
            provide examples to the agent. It will be added to the
            system prompt as examples for the agent to follow.
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
            f"{self.system_prompt}\n\n## Output Model Schema\n\n"
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
            f"{self.system_prompt}\n\n## Output Model Schema\n\n"
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
        response["model"] = self._parse_model_response(response["response"])
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
        response["model"] = self._parse_model_response(response["response"])
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
                        model=item,
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
        input_chunks = [
            input_list[
                i * requests_per_prompt : (i + 1) * requests_per_prompt
            ]
            for i in range(ceil(len(input_list) / requests_per_prompt))
        ]
        tasks = [
            self._act_in_batch_async_request(
                input_data, chat_kwargs=chat_kwargs, _idx=idx
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
                        model=item,
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
            system_examples:
                - input:
                    <example 1 input, a dictionary or a list of dictionaries>
                  output:
                    <example 1 output, a dictionary or a list of dictionaries>
                - input:
                    <example 2 input, a dictionary or a list of dictionaries>
                  output:
                    <example 2 output, a dictionary or a list of dictionaries>
                ...
            output_model: <path to the pydantic model class, e.g.
                "agents7.models.MyOutputModel">
            ```
            The field `system_examples` is optional and can be used to
            provide examples to the agent. It will be added to the
            system prompt as examples for the agent to follow.
        :return: Loaded configuration, as a dictionary. Keys:
            - name (str): The name of the agent.
            - version (int): The version of the agent.
            - engine (str): The LLM engine used by the agent.
            - system_prompt (str): The system prompt for the agent.
            - output_model (BaseModel): The model used to validate and
                structure the output.
            - examples (list[dict], optional): List of examples to include in
                the system prompt. Each example is a dictionary with 'input'
                and 'output' keys. Defaults to None.
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

        system_prompt = config["system_prompt"].strip()
        if examples := config.get("system_examples"):
            try:
                get_examples_str(examples)
            except ValueError as e:
                raise ValueError(
                    f"{e} in the configuration file `{file_path}`"
                ) from e
        return {
            "name": config["name"],
            "version": config["version"],
            "engine": LLMEngine(engine["provider"], engine["model"]),
            "system_prompt": system_prompt.strip(),
            "output_model": import_from_path(config["output_model"]),
            "examples": examples,
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
        try:
            model = model_cls.model_validate_json(response)
        except Exception:
            try:
                # Ad-hoc way to handle ollama's structured response
                model = model_cls.model_validate_json(response[2:])
            except Exception as e:
                logger.error(
                    f"Agent `{self.name}`: error parsing structured "
                    f"response:\n{e}"
                )
        return model
