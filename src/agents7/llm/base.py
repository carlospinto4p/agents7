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
        provider = provider.lower()
        self.provider = provider
        self.model_name = model_name

        if provider == "ollama":
            from agents7.llm.ollama import OllamaProvider

            self.llm_provider = OllamaProvider(model_name)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """
        Generate text based on the provided prompt using the Ollama API.
        :param prompt: The input prompt to send to the model.
        :param temperature: Temperature for the model.
        :param max_tokens: Maximum number of tokens to generate.
            Default is -1, which means no limit.
        :param output_format: Format of the response. Can be "json" or the json
            schema to use, e.g.:
            {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "integer"
                    },
                    "available": {
                        "type": "boolean"
                    }
                },
                "required": [
                  "age",
                  "available"
                ]
            }
        :param think: If True, the model will think before generating the
            response.
        :param seed: Seed for random number generation. Default is None.
        :return: A dictionary containing the parsed response. Schema:
            - response: The generated text response.
            - thinking: The thinking part of the response, if any.
            - total_duration: The total duration of the response, in seconds.
        """
        return self.llm_provider.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )

    async def generate_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """
        Asynchronously generate text based on the provided prompt using the
        Ollama API.
        :param prompt: The input prompt to send to the model.
        :param temperature: Temperature for the model.
        :param max_tokens: Maximum number of tokens to generate.
            Default is -1, which means no limit.
        :param output_format: Format of the response. Can be "json" or the json
            schema to use, e.g.:
            {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "integer"
                    },
                    "available": {
                        "type": "boolean"
                    }
                },
                "required": [
                  "age",
                  "available"
                ]
            }
        :param think: If True, the model will think before generating the
            response.
        :param seed: Seed for random number generation. Default is None.
        :return: A dictionary containing the parsed response. Schema:
            - response: The generated text response.
            - thinking: The thinking part of the response, if any.
            - total_duration: The total duration of the response, in seconds.
        """
        return await self.llm_provider.generate_async(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )

    def get_model_info(self) -> dict:
        """
        Get information about the model. Keys:
            - provider: The provider of the LLM engine (e.g., "ollama").
            - model_name: The name of the model being used.
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
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """
        Generate text based on the provided prompt using the Ollama API.
        :param prompt: The input prompt to send to the model.
        :param temperature: Temperature for the model.
        :param max_tokens: Maximum number of tokens to generate.
            Default is -1, which means no limit.
        :param output_format: Format of the response. Can be "json" or the json
            schema to use, e.g.:
            {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "integer"
                    },
                    "available": {
                        "type": "boolean"
                    }
                },
                "required": [
                  "age",
                  "available"
                ]
            }
        :param think: If True, the model will think before generating the
            response.
        :param seed: Seed for random number generation. Default is None.
        :return: A dictionary containing the parsed response. Schema:
            - response: The generated text response.
            - thinking: The thinking part of the response, if any.
            - total_duration: The total duration of the response, in seconds.
        """
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """
        Asynchronously generate text based on the provided prompt using the
        Ollama API.
        :param prompt: The input prompt to send to the model.
        :param temperature: Temperature for the model.
        :param max_tokens: Maximum number of tokens to generate.
            Default is -1, which means no limit.
        :param output_format: Format of the response. Can be "json" or the json
            schema to use, e.g.:
            {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "integer"
                    },
                    "available": {
                        "type": "boolean"
                    }
                },
                "required": [
                  "age",
                  "available"
                ]
            }
        :param think: If True, the model will think before generating the
            response.
        :param seed: Seed for random number generation. Default is None.
        :return: A dictionary containing the parsed response. Schema:
            - response: The generated text response.
            - thinking: The thinking part of the response, if any.
            - total_duration: The total duration of the response, in seconds.
        """
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )

    def get_model_info(self) -> dict:
        """
        Get information about the model. Keys:
            - provider: The provider of the LLM engine (e.g., "ollama").
            - model_name: The name of the model being used.
        """
        return {"provider": self.provider, "model_name": self.model_name}
