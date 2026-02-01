from __future__ import annotations

from abc import ABC, abstractmethod

from agents7.llm.retry import RetryConfig, retry_async, retry_sync
from agents7.llm.streaming import AsyncStreamResult, StreamResult


class LLMEngine:
    """
    Base class for LLM engines.
    This class defines the interface for LLM engines, including methods
    for generating text and retrieving model information.

    Attributes:
        provider (str): The provider of the LLM engine (e.g., "ollama").
        model_name (str): The name of the model being used.
        retry_config (RetryConfig | None): Optional retry configuration.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        retry: RetryConfig | None = None,
    ):
        provider = provider.lower()
        self.provider = provider
        self.model_name = model_name
        self.model_id: str = f"{provider}::{model_name}"
        self.retry_config = retry

        if provider == "ollama":
            from agents7.llm.ollama import OllamaProvider

            self.llm_provider = OllamaProvider(model_name)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """
         Generate a chat response based on the provided messages.
         :param messages: A list of messages to send to the model. Each message
             should be a dictionary with keys "role" and "content". For example:
             [
                 {"role": "user", "content": "Hello, how are you?"},
                 {"role": "assistant", "content": "I'm fine, thank you!"}
             ]
             Available roles: "system", "user", "assistant" or "tool".
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
             Easiest way to use this is to pass a Pydantic model
             `model_json_schema()` method output. See:
             https://github.com/ollama/ollama-python/blob/main/examples/structured-outputs.py
         :param think: If True, the model will think before generating the
             response.
         :param seed: Seed for random number generation. Default is None.
         :return: A dictionary containing the parsed response. Schema:
             - response: The generated text response.
             - role: The role of the message (e.g., "user", "assistant").
             - thinking: The thinking part of the response, if any.
             - metrics: A dictionary containing the following metrics:
                 - tokens: The number of tokens in the response.
                 - prompt_tokens: The number of tokens in the prompt.
                 - total_duration: The total duration of the response,
                     in seconds.
        """
        return retry_sync(
            self.llm_provider.chat,
            self.retry_config,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )

    async def chat_async(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """
         Generate a chat response based on the provided messages.
         :param messages: A list of messages to send to the model. Each message
             should be a dictionary with keys "role" and "content". For example:
             [
                 {"role": "user", "content": "Hello, how are you?"},
                 {"role": "assistant", "content": "I'm fine, thank you!"}
             ]
             Available roles: "system", "user", "assistant" or "tool".
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
             Easiest way to use this is to pass a Pydantic model
             `model_json_schema()` method output. See:
             https://github.com/ollama/ollama-python/blob/main/examples/structured-outputs.py
         :param think: If True, the model will think before generating the
             response.
         :param seed: Seed for random number generation. Default is None.
         :return: A dictionary containing the parsed response. Schema:
             - response: The generated text response.
             - role: The role of the message (e.g., "user", "assistant").
             - thinking: The thinking part of the response, if any.
             - metrics: A dictionary containing the following metrics:
                 - tokens: The number of tokens in the response.
                 - prompt_tokens: The number of tokens in the prompt.
                 - total_duration: The total duration of the response,
                     in seconds.
        """
        return await retry_async(
            self.llm_provider.chat_async,
            self.retry_config,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )

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
            Easiest way to use this is to pass a Pydantic model
            `model_json_schema()` method output. See:
            https://github.com/ollama/ollama-python/blob/main/examples/structured-outputs.py
        :param think: If True, the model will think before generating the
            response.
        :param seed: Seed for random number generation. Default is None.
        :return: A dictionary containing the parsed response. Schema:
            - response: The generated text response.
            - thinking: The thinking part of the response, if any.
            - metrics: A dictionary containing the following metrics:
                - tokens: The number of tokens in the response.
                - prompt_tokens: The number of tokens in the prompt.
                - total_duration: The total duration of the response,
                    in seconds.
        """
        return retry_sync(
            self.llm_provider.generate,
            self.retry_config,
            prompt=prompt,
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
            Easiest way to use this is to pass a Pydantic model
            `model_json_schema()` method output. See:
            https://github.com/ollama/ollama-python/blob/main/examples/structured-outputs.py
        :param think: If True, the model will think before generating the
            response.
        :param seed: Seed for random number generation. Default is None.
        :return: A dictionary containing the parsed response. Schema:
            - response: The generated text response.
            - thinking: The thinking part of the response, if any.
            - metrics: A dictionary containing the following metrics:
                - tokens: The number of tokens in the response.
                - prompt_tokens: The number of tokens in the prompt.
                - total_duration: The total duration of the response,
                    in seconds.
        """
        return await retry_async(
            self.llm_provider.generate_async,
            self.retry_config,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> StreamResult:
        """Stream a chat response, yielding text chunks.

        Bypasses retry logic — streaming calls cannot be retried
        mid-stream.
        """
        return self.llm_provider.chat_stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )

    async def chat_stream_async(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> AsyncStreamResult:
        """Stream a chat response asynchronously.

        Bypasses retry logic — streaming calls cannot be retried
        mid-stream.
        """
        return await self.llm_provider.chat_stream_async(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )

    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> StreamResult:
        """Stream a generate response, yielding text chunks.

        Bypasses retry logic — streaming calls cannot be retried
        mid-stream.
        """
        return self.llm_provider.generate_stream(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
            think=think,
            seed=seed,
        )

    async def generate_stream_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> AsyncStreamResult:
        """Stream a generate response asynchronously.

        Bypasses retry logic — streaming calls cannot be retried
        mid-stream.
        """
        return await self.llm_provider.generate_stream_async(
            prompt=prompt,
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
            - model_id: A unique identifier for the model in the format
                "<provider>::<model_name>".
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
        self.model_id: str = f"{provider}::{model_name}"

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """
         Generate a chat response based on the provided messages.
         :param messages: A list of messages to send to the model. Each message
             should be a dictionary with keys "role" and "content". For example:
             [
                 {"role": "user", "content": "Hello, how are you?"},
                 {"role": "assistant", "content": "I'm fine, thank you!"}
             ]
             Available roles: "system", "user", "assistant" or "tool".
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
             Easiest way to use this is to pass a Pydantic model
             `model_json_schema()` method output. See:
             https://github.com/ollama/ollama-python/blob/main/examples/structured-outputs.py
         :param think: If True, the model will think before generating the
             response.
         :param seed: Seed for random number generation. Default is None.
         :return: A dictionary containing the parsed response. Schema:
             - response: The generated text response.
             - role: The role of the message (e.g., "user", "assistant").
             - thinking: The thinking part of the response, if any.
             - metrics: A dictionary containing the following metrics:
                 - tokens: The number of tokens in the response.
                 - prompt_tokens: The number of tokens in the prompt.
                 - total_duration: The total duration of the response,
                     in seconds.
        """
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )

    @abstractmethod
    async def chat_async(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """
         Generate a chat response based on the provided messages.
         :param messages: A list of messages to send to the model. Each message
             should be a dictionary with keys "role" and "content". For example:
             [
                 {"role": "user", "content": "Hello, how are you?"},
                 {"role": "assistant", "content": "I'm fine, thank you!"}
             ]
             Available roles: "system", "user", "assistant" or "tool".
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
             Easiest way to use this is to pass a Pydantic model
             `model_json_schema()` method output. See:
             https://github.com/ollama/ollama-python/blob/main/examples/structured-outputs.py
         :param think: If True, the model will think before generating the
             response.
         :param seed: Seed for random number generation. Default is None.
         :return: A dictionary containing the parsed response. Schema:
             - response: The generated text response.
             - role: The role of the message (e.g., "user", "assistant").
             - thinking: The thinking part of the response, if any.
             - metrics: A dictionary containing the following metrics:
                 - tokens: The number of tokens in the response.
                 - prompt_tokens: The number of tokens in the prompt.
                 - total_duration: The total duration of the response,
                     in seconds.
        """
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )

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
            Easiest way to use this is to pass a Pydantic model
            `model_json_schema()` method output. See:
            https://github.com/ollama/ollama-python/blob/main/examples/structured-outputs.py
        :param think: If True, the model will think before generating the
            response.
        :param seed: Seed for random number generation. Default is None.
        :return: A dictionary containing the parsed response. Schema:
            - response: The generated text response.
            - thinking: The thinking part of the response, if any.
            - metrics: A dictionary containing the following metrics:
                - tokens: The number of tokens in the response.
                - prompt_tokens: The number of tokens in the prompt.
                - total_duration: The total duration of the response,
                    in seconds.
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
            Easiest way to use this is to pass a Pydantic model
            `model_json_schema()` method output. See:
            https://github.com/ollama/ollama-python/blob/main/examples/structured-outputs.py
        :param think: If True, the model will think before generating the
            response.
        :param seed: Seed for random number generation. Default is None.
        :return: A dictionary containing the parsed response. Schema:
            - response: The generated text response.
            - thinking: The thinking part of the response, if any.
            - metrics: A dictionary containing the following metrics:
                - tokens: The number of tokens in the response.
                - prompt_tokens: The number of tokens in the prompt.
                - total_duration: The total duration of the response,
                    in seconds.
        """
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> StreamResult:
        """Stream a chat response, yielding text chunks."""
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )

    @abstractmethod
    async def chat_stream_async(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> AsyncStreamResult:
        """Stream a chat response asynchronously."""
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> StreamResult:
        """Stream a generate response, yielding text chunks."""
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )

    @abstractmethod
    async def generate_stream_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> AsyncStreamResult:
        """Stream a generate response asynchronously."""
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )

    def get_model_info(self) -> dict:
        """
        Get information about the model. Keys:
            - provider: The provider of the LLM engine (e.g., "ollama").
            - model_name: The name of the model being used.
            - model_id: A unique identifier for the model in the format
                "<provider>::<model_name>".
        """
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "model_id": f"{self.model_id}",
        }
