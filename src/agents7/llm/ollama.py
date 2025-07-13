"""
See https://www.youtube.com/watch?v=d0o89z134CQ
(The video uses langchain, but the concepts are similar)

Ensure Ollama is installed and running locally:

```bash
ollama run <model_name>
```
This command will start the <model_name> model and keep the server
listening at http://localhost:11434.

See the docs at https://github.com/ollama/ollama/blob/main/docs/api.md.

Additional resources:
https://www.gpu-mart.com/blog/custom-llm-models-with-ollama-modelfile
"""

from ollama import AsyncClient, ChatResponse, GenerateResponse, generate, chat

from agents7.llm.base import LLMProvider


class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider for generating text using the Ollama API.
    """

    def __init__(self, model_name: str):
        super().__init__("ollama", model_name)

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
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if seed is not None:
            options["seed"] = seed

        response = chat(
            model=self.model_name,
            messages=messages,
            options=options,
            think=think,
            format=output_format,
        )
        return parse_ollama_chat_response(response)

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
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if seed is not None:
            options["seed"] = seed

        response = await AsyncClient().chat(
            model=self.model_name,
            messages=messages,
            options=options,
            think=think,
            format=output_format,
        )
        return parse_ollama_chat_response(response)

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
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if seed is not None:
            options["seed"] = seed

        response = generate(
            model=self.model_name,
            prompt=prompt,
            options=options,
            think=think,
            format=output_format,
        )
        return parse_ollama_response(response)

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
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if seed is not None:
            options["seed"] = seed

        response = await AsyncClient().generate(
            model=self.model_name,
            prompt=prompt,
            options=options,
            think=think,
            format=output_format,
        )
        return parse_ollama_response(response)


def parse_ollama_chat_response(chat_response: ChatResponse) -> dict:
    """
    Parse the response from the Ollama API.
    :param chat_response: The response object from the Ollama API.
    :return: A dictionary containing the parsed response. Schema:
        - response: The generated text response.
        - role: The role of the message (e.g., "user", "assistant").
        - thinking: The thinking part of the response, if any.
        - metrics: A dictionary containing the following metrics:
            - tokens: The number of tokens in the response.
            - prompt_tokens: The number of tokens in the prompt.
            - total_duration: The total duration of the response, in seconds.
    """
    return {
        "response": chat_response.message.content.strip(),
        "role": chat_response.message.role,
        "thinking": chat_response.message.thinking.strip()
        if chat_response.message.thinking
        else None,
        "metrics": {
            "tokens": chat_response.eval_count,
            "prompt_tokens": chat_response.prompt_eval_count,
            "total_duration": chat_response.total_duration / 10**9,
        },
    }


def parse_ollama_response(response: GenerateResponse) -> dict:
    """
    Parse the response from the Ollama API.
    :param response: The response object from the Ollama API.
    :return: A dictionary containing the parsed response. Schema:
        - response: The generated text response.
        - thinking: The thinking part of the response, if any.
        - metrics: A dictionary containing the following metrics:
            - tokens: The number of tokens in the response.
            - prompt_tokens: The number of tokens in the prompt.
            - total_duration: The total duration of the response, in seconds.
    """
    return {
        "response": response.response.split("</think>")[-1].strip(),
        "thinking": response.response.split("<think>")[-1]
        .split("</think>")[0]
        .strip(),
        "metrics": {
            "tokens": response.eval_count,
            "prompt_tokens": response.prompt_eval_count,
            "total_duration": response.total_duration / 10**9,
        },
    }
