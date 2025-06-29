from abc import ABC, abstractmethod


class LLMEngine:
    """
    Base class for LLM engines.
    This class defines the interface for LLM engines, including methods
    for generating text and retrieving model information.

    Attributes:
        provider (str): The provider of the LLM engine (e.g., "ollama").
        model_name (str): The name of the model being used.
    """
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        self.llm_provider = LLMProvider(provider, model_name)

    def generate(self, prompt: str) -> str:
        """
        Generate text based on the provided prompt.
        """
        return self.llm_provider.generate(prompt)

    def get_model_info(self) -> dict:
        """
        Get information about the model.
        """
        return self.llm_provider.get_model_info()


class LLMProvider(ABC):
    """
    Base class for LLM providers.
    This class defines the interface for LLM engines, including methods
    for generating text and retrieving model information.

    Attributes:
        provider (str): The provider of the LLM engine (e.g., "ollama").
        model_name (str): The name of the model being used.
    """

    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate text based on the provided prompt.
        """
        raise NotImplementedError("This method should be overridden "
                                  "by subclasses.")

    def get_model_info(self) -> dict:
        """
        Get information about the model.
        """
        return {
            "provider": self.provider,
            "model_name": self.model_name
        }
