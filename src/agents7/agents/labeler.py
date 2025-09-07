import importlib
from asyncio import as_completed
from dataclasses import dataclass
from typing import Callable

from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from pydantic import BaseModel

from cpinto.files import load_yaml
from cpinto.functions import identity_func

from agents7.agents.base import BaseAgent
from agents7.llm import LLMEngine
from agents7.logs import logger


@dataclass
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
    """

    output_model: type[BaseModel]

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
            if not isinstance(examples, list):
                raise ValueError(
                    f"Field `system_examples` must be a list in the "
                    f"configuration yaml `{file_path}`"
                )

            system_prompt = (
                config["system_prompt"].strip() + "\n\n## Examples:"
            )
            for example in examples:
                if not isinstance(example, dict):
                    raise ValueError(
                        f"Each example in `system_examples` must be a "
                        f"dictionary in the configuration yaml `{file_path}`"
                    )
                if set(example.keys()) != {"input", "output"}:
                    raise ValueError(
                        f"Each example in `system_examples` must contain "
                        f"`input` and `output` keys in the configuration "
                        f"yaml `{file_path}`"
                    )
                system_prompt += (
                    f"\n\n- Input: {example['input']}"
                    f"\n- Output: {example['output']}"
                )

            # TODO 2025.09.07 If `output_model`, add its schema to the prompt.
            # something like:
            # def model_to_class_string(model_cls: type[BaseModel]) -> str:
            #     lines = [f"class {model_cls.__name__}:"]
            #     for field_name, field_info in model_cls.model_fields.items():
            #         # annotation type
            #         ann = (
            #             field_info.annotation.__name__
            #             if hasattr(field_info.annotation, "__name__")
            #             else str(field_info.annotation)
            #         )
            #
            #         # default value
            #         default_repr = ""
            #         if field_info.default is not None:
            #             default_repr = f" = {field_info.default!r}"
            #
            #         # description (from Field or comment)
            #         description = ""
            #         if field_info.description:
            #             description = f"  # {field_info.description}"
            #
            #         # append line
            #         lines.append(
            #             f"    {field_name}: {ann}{default_repr}{description}")
            #
            #     return "\n".join(lines)

        return {
            "name": config["name"],
            "version": config["version"],
            "engine": LLMEngine(engine["provider"], engine["model"]),
            "system_prompt": system_prompt.strip(),
            "output_model": import_from_path(config["output_model"]),
        }

    def act(
        self, input_data: dict | list[dict], chat_kwargs: dict = None
    ) -> dict:
        """
        Perform the labeling action based on the provided input.
        :param input_data: The input data to be processed. The system prompt
            should specify the expected format of this data.
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
        :return: A dictionary containing the response from the agent:
            - response: The generated text response.
            - model: The structured output validated by the output model.
            - thinking: The thinking part of the response, if any.
            - metrics: A dictionary containing the following metrics:
             - tokens: The number of tokens in the response.
             - prompt_tokens: The number of tokens in the prompt.
             - total_duration: The total duration of the response,
                 in seconds.
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
                {"role": "system", "content": self.system_prompt},
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
        return response

    async def act_async(
        self,
        input_data: dict | list[dict],
        chat_kwargs: dict = None,
        _idx: int = 0,
    ) -> dict:
        """
        Perform the labeling action based on the provided input.
        :param input_data: The input data to be processed. The system prompt
            should specify the expected format of this data.
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
        :return: A dictionary containing the response from the agent:
            - response: The generated text response.
            - model: The structured output validated by the output model.
            - thinking: The thinking part of the response, if any.
            - metrics: A dictionary containing the following metrics:
             - tokens: The number of tokens in the response.
             - prompt_tokens: The number of tokens in the prompt.
             - total_duration: The total duration of the response,
                 in seconds.
            - _idx: The index of the input data when processing in batch mode.
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
                {"role": "system", "content": self.system_prompt},
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
        response["_idx"] = _idx
        return response

    def act_in_batch(
        self,
        input_list: list[dict | list[dict]],
        chat_kwargs: dict = None,
        iter_func: Callable = tqdm,
    ) -> list[dict]:
        """
        Perform the labeling action based on the provided input, in batch mode.
        :param input_list: List of input data to be processed.
            The system prompt should specify the expected format of
            each element of this data.
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
        :param iter_func: Function to keep track of the iterations.
            Defaults to `tqdm.tqdm`. If `None`, no function will be used.
        :return: A list of dictionaries containing the response from the agent:
            - response: The generated text response.
            - model: The structured output validated by the output model.
            - thinking: The thinking part of the response, if any.
            - metrics: A dictionary containing the following metrics:
             - tokens: The number of tokens in the response.
             - prompt_tokens: The number of tokens in the prompt.
             - total_duration: The total duration of the response,
                 in seconds.
        """
        if iter_func is None:
            iter_func = identity_func
        agent_response = []
        for input_data in iter_func(input_list):
            response = self.act(input_data, chat_kwargs=chat_kwargs)
            if response["model"] is not None:
                agent_response.append(response["model"])
        return agent_response

    async def act_in_batch_async(
        self,
        input_list: list[dict | list[dict]],
        chat_kwargs: dict = None,
        print_tqdm: bool = True,
    ) -> list[dict]:
        """
        Perform the labeling action based on the provided input, in batch mode.
        :param input_list: List of input data to be processed.
            The system prompt should specify the expected format of
            each element of this data.
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
        :param print_tqdm: Whether to use `tqdm.asyncio` to keep track of the
            iterations.
        :return: A list of dictionaries containing the response from the agent:
            - response: The generated text response.
            - model: The structured output validated by the output model.
            - thinking: The thinking part of the response, if any.
            - metrics: A dictionary containing the following metrics:
             - tokens: The number of tokens in the response.
             - prompt_tokens: The number of tokens in the prompt.
             - total_duration: The total duration of the response,
                 in seconds.
        """
        tasks = [
            self.act_async(input_data, chat_kwargs=chat_kwargs, _idx=idx)
            for idx, input_data in enumerate(input_list)
        ]

        agent_response: list[dict | None] = [None] * len(input_list)
        if print_tqdm:
            for coro in async_tqdm.as_completed(
                tasks,
                total=len(tasks),
                desc=f"Agent `{self.name}`: acting in batch",
            ):
                response = await coro
                if (model := response["model"]) is not None:
                    agent_response[response.pop("_idx")] = model
        else:
            for coro in as_completed(tasks):
                response = await coro
                if (model := response["model"]) is not None:
                    agent_response[response.pop("_idx")] = model

        return agent_response

    def _parse_model_response(self, response: str) -> BaseModel | None:
        """
        Parse the response from the agent and validate it against the output
        model.
        :param response: The response string from the agent.
        :return: The structured output validated by the output model.
            Will return `None` if the response cannot be parsed.
        """
        model = None
        try:
            model = self.output_model.model_validate_json(response)
        except Exception:
            try:
                # Ad-hoc way to handle ollama's structured response
                model = self.output_model.model_validate_json(response[2:])
            except Exception as e:
                logger.error(
                    f"Agent `{self.name}`: error parsing structured "
                    f"response:\n{e}"
                )
        return model


def import_from_path(path: str) -> type[BaseModel]:
    """
    Import a class from a given module path.
    :param path: Path to the module and class, e.g.
        "module.submodule.ClassName".
    :return: The class object imported from the specified path.
    """
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
