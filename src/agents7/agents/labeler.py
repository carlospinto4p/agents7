from pydantic import BaseModel

from agents7.agents.base import BaseAgent
from agents7.llm import LLMEngine
from agents7.logs import logger


class Labeler(BaseAgent):
    """
    Labeler agent for processing and labeling data.
    This agent is designed to analyze and label data based on predefined rules.

    Attributes:
        - name (str): The name of the agent.
        - engine (LLMEngine): The LLM engine used by the agent.
        - system_prompt (str): The system prompt for the agent.
        - output_model (BaseModel): The model used to validate and
            structure the output.
    """

    def __init__(
        self,
        name: str,
        engine: LLMEngine,
        system_prompt: str,
        output_model: type[BaseModel],
    ):
        """
        Initialize the Labeler agent with the given parameters.
        :param name: The name of the agent.
        :param engine: The LLM engine used by the agent.
        :param system_prompt: The system prompt for the agent.
        :param output_model: The model used to validate and structure the
            output.
        """
        super().__init__(
            name=name, engine=engine, system_prompt=system_prompt
        )
        self.output_model = output_model

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

        agent_str = f"Agent `{self.name}`:"
        logger.debug(f"{agent_str} processing input data:\n{input_data}")
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
        try:
            response["model"] = self.output_model.model_validate_json(
                response["response"]
            )
        except Exception:
            try:
                # Ad-hoc way to handle ollama's structured response
                response["model"] = self.output_model.model_validate_json(
                    response["response"][2:]
                )
            except Exception as e:
                logger.error(
                    f"{agent_str}: error parsing structured response:\n{e}"
                )
                response["model"] = None
        return response

    def act_in_batch(
        self, input_list: list[dict | list[dict]], chat_kwargs: dict = None
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
        agent_response = []
        for input_data in input_list:
            response = self.act(input_data, chat_kwargs=chat_kwargs)
            if response["model"] is not None:
                agent_response.append(response["model"])
        return agent_response
