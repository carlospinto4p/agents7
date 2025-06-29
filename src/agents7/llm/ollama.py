
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
from json import loads

import requests

from agents7.exceptions import LLMException
from agents7.llm.base import LLMProvider


def run_model(
    model_name: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = -1,
    output_format: str | dict = None,
    stop: str | None = None,
    seed: int | None = None,
) -> dict:
    """
    Get a response from the Ollama model.

    :param model_name: Name of the model.
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
    :param stop: List of stop sequences to end the generation. Once the model
        generates any of these sequences, it will stop generating further
        tokens. Default is None, which means no stop sequences.
    :param seed: Seed for random number generation. Default is None.
    Returns:
        str: The generated response from the model.
    """
    url = f"http://localhost:11434/api/generate/"

    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
    }
    if seed is not None:
        options["seed"] = seed

    payload = {
        "model": model_name,
        "prompt": prompt,
        "options": options,
    }
    if output_format:
        payload["output_format"] = output_format
    if stop:
        payload["stop"] = stop

    try:
        response = requests.post(url, json=payload)
    except requests.exceptions.RequestException as e:
        raise LLMException(f"Error connecting to Ollama API: {e}")

    if response.status_code != 200:
        raise LLMException(f"Error from Ollama API: {response.text}")

    return loads(response.text)


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
            stop: str | None = None,
            seed: int | None = None
    ) -> str:
        """
        Generate text based on the provided prompt using the Ollama API.
        """
        response = run_model(self.model_name, prompt)
        return response.get("text", "")
