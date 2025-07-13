from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from agents7.llm import LLMEngine


Agent = TypeVar("Agent", bound="BaseAgent")


@dataclass
class BaseAgent(ABC):
    """
    Base class for all agents.
    This class should be inherited by all agent classes.

    Attributes:
        - name (str): The name of the agent.
        - engine (LLMEngine): The LLM engine used by the agent.
        - system_prompt (str): The system prompt for the agent.
    """

    name: str
    engine: LLMEngine
    system_prompt: str

    @abstractmethod
    def act(self, *args, **kwargs) -> dict:
        """
        Perform an action based on the current state and input.
        This method should be overridden by subclasses.

        :param args: Positional arguments for the action.
        :param kwargs: Keyword arguments for the action.
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
        raise NotImplementedError("Subclasses must implement this method.")
