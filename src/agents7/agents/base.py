from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from string import Template
from typing import Any, TypeVar

from agents7.exceptions import PromptParameterError
from agents7.llm import LLMEngine
from agents7.models import AgentResponse

Agent = TypeVar("Agent", bound="BaseAgent")


@dataclass(frozen=True)
class BaseAgent(ABC):
    """
    Base class for all agents.
    This class should be inherited by all agent classes.

    Attributes:
        name (str): The name of the agent.
        version (int): The version of the agent.
        engine (LLMEngine): The LLM engine used by the agent.
        system_prompt (str): The system prompt for the agent.
        prompt_params (dict, optional): Dictionary of parameters to
            substitute into the system_prompt using
            $variable placeholders. Values can be strings or
            callables returning strings. Defaults to None.
    """

    name: str
    version: int
    engine: LLMEngine
    system_prompt: str
    prompt_params: dict[str, str | Callable[[], str]] | None = field(
        default=None, kw_only=True
    )

    def __post_init__(self) -> None:
        if self.prompt_params is not None:
            if not isinstance(self.prompt_params, dict):
                raise PromptParameterError(
                    "`prompt_params` must be a dict or None."
                )
            for key, value in self.prompt_params.items():
                if not isinstance(value, str) and not callable(value):
                    raise PromptParameterError(
                        f"prompt_params['{key}'] must be a str "
                        f"or callable, got {type(value).__name__}."
                    )

    @property
    def resolved_system_prompt(self) -> str:
        """Resolved system prompt with placeholders substituted."""
        if not self.prompt_params:
            return self.system_prompt
        resolved = {
            k: v() if callable(v) else v
            for k, v in self.prompt_params.items()
        }
        return Template(self.system_prompt).safe_substitute(resolved)

    @classmethod
    @abstractmethod
    def from_yaml(cls, file_path: str) -> Agent:
        """
        Create an instance of the agent from a YAML file configuration.
        This method should be overridden by subclasses to provide specific
        YAML loading logic.

        :param file_path: The path to the YAML file containing the agent
            configuration.
        :return: An instance of the agent.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def act(self, *args: Any, **kwargs: Any) -> AgentResponse:
        """
        Perform an action based on the current state and input.
        This method should be overridden by subclasses.

        :param args: Positional arguments for the action.
        :param kwargs: Keyword arguments for the action.
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
        raise NotImplementedError("Subclasses must implement this method.")
