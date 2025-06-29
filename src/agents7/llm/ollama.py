
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
"""
from ollama import AsyncClient, GenerateResponse, generate

from agents7.llm.base import LLMProvider


def parse_ollama_response(response: GenerateResponse) -> dict:
    """
    Parse the response from the Ollama API.
    :param response: The response dictionary from the Ollama API.
    :return: A dictionary containing the parsed response. Schema:
        - response: The generated text response.
        - thinking: The thinking part of the response, if any.
        - total_duration: The total duration of the response, in seconds.
    """
    return {
        "response": response.response.split("</think>")[-1].strip(),
        "thinking": response.response.split("<think>")[-1].split(
            "</think>")[0].strip(),
        "total_duration": response.total_duration / 10**9,
    }


class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider for generating text using the Ollama API.
    """

    def __init__(self, model_name: str):
        super().__init__("ollama", model_name)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None
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
            format=output_format
        )
        return parse_ollama_response(response)

    async def generate_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None
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
            format=output_format
        )
        return parse_ollama_response(response)
