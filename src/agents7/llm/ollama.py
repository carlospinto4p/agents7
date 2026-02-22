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
from agents7.llm.streaming import AsyncStreamResult, StreamResult


class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider for generating text using the Ollama API.
    """

    def __init__(self, model_name: str):
        super().__init__("ollama", model_name)

    @staticmethod
    def _build_options(
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> dict:
        """
        Build the Ollama options dictionary.
        :param temperature: Temperature for the model.
        :param max_tokens: Maximum number of tokens.
        :param seed: Seed for reproducibility, or None.
        :return: Options dictionary for the Ollama API.
        """
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if seed is not None:
            options["seed"] = seed
        return options

    @staticmethod
    def _needs_format_fallback(
        parsed: dict,
        output_format: str | dict | None,
    ) -> bool:
        """Check if a chat response needs retry without
        structured output format.

        Some models don't support structured output; when
        the response is empty and a format was requested,
        the call should be retried without format.
        """
        return not parsed["response"] and output_format is not None

    def _chat_call_kwargs(
        self,
        messages: list[dict[str, str]],
        options: dict,
        think: bool,
        output_format: str | dict | None,
    ) -> dict:
        """Build kwargs for an Ollama chat call."""
        return {
            "model": self.model_name,
            "messages": messages,
            "options": options,
            "think": think,
            "format": output_format,
        }

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """See `LLMProvider.chat` for documentation."""
        options = OllamaProvider._build_options(temperature, max_tokens, seed)
        kwargs = self._chat_call_kwargs(
            messages, options, think, output_format
        )

        response = chat(**kwargs)
        parsed = parse_ollama_chat_response(response)

        if self._needs_format_fallback(parsed, output_format):
            response = chat(**{**kwargs, "format": None})
            parsed = parse_ollama_chat_response(response)

        return parsed

    async def chat_async(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """See `LLMProvider.chat` for documentation."""
        options = OllamaProvider._build_options(temperature, max_tokens, seed)
        kwargs = self._chat_call_kwargs(
            messages, options, think, output_format
        )

        response = await AsyncClient().chat(**kwargs)
        parsed = parse_ollama_chat_response(response)

        if self._needs_format_fallback(parsed, output_format):
            response = await AsyncClient().chat(**{**kwargs, "format": None})
            parsed = parse_ollama_chat_response(response)

        return parsed

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
    ) -> dict:
        """See `LLMProvider.generate` for documentation."""
        options = OllamaProvider._build_options(temperature, max_tokens, seed)

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
        """See `LLMProvider.generate` for documentation."""
        options = OllamaProvider._build_options(temperature, max_tokens, seed)

        response = await AsyncClient().generate(
            model=self.model_name,
            prompt=prompt,
            options=options,
            think=think,
            format=output_format,
        )
        return parse_ollama_response(response)

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
        options = OllamaProvider._build_options(temperature, max_tokens, seed)

        iterator = chat(
            model=self.model_name,
            messages=messages,
            options=options,
            think=think,
            format=output_format,
            stream=True,
        )
        return StreamResult(iterator, "chat")

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
        options = OllamaProvider._build_options(temperature, max_tokens, seed)

        iterator = await AsyncClient().chat(
            model=self.model_name,
            messages=messages,
            options=options,
            think=think,
            format=output_format,
            stream=True,
        )
        return AsyncStreamResult(iterator, "chat")

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
        options = OllamaProvider._build_options(temperature, max_tokens, seed)

        iterator = generate(
            model=self.model_name,
            prompt=prompt,
            options=options,
            think=think,
            format=output_format,
            stream=True,
        )
        return StreamResult(iterator, "generate")

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
        options = OllamaProvider._build_options(temperature, max_tokens, seed)

        iterator = await AsyncClient().generate(
            model=self.model_name,
            prompt=prompt,
            options=options,
            think=think,
            format=output_format,
            stream=True,
        )
        return AsyncStreamResult(iterator, "generate")


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
    thinking = None
    if hasattr(response, "thinking") and response.thinking:
        thinking = response.thinking.strip()
    if "<think>" in response.response and "</think>" in response.response:
        thinking = (
            response.response.split("<think>")[-1]
            .split("</think>")[0]
            .strip()
        )

    return {
        "response": response.response.split("</think>")[-1].strip(),
        "thinking": thinking,
        "metrics": {
            "tokens": response.eval_count,
            "prompt_tokens": response.prompt_eval_count,
            "total_duration": response.total_duration / 10**9,
        },
    }
