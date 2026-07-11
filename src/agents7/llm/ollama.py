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

from collections.abc import Callable

from ollama import (
    AsyncClient,
    ChatResponse,
    GenerateResponse,
    ResponseError,
    chat,
    generate,
)

from agents7.llm.base import LLMProvider
from agents7.llm.streaming import AsyncStreamResult, StreamResult


class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider for generating text using the Ollama API.
    """

    def __init__(self, model_name: str):
        super().__init__("ollama", model_name)
        self._async_client = AsyncClient()

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
    def _think_kwargs(think: bool) -> dict:
        """Return `{'think': think}`, always passing the flag.

        Sending `think=False` explicitly disables Ollama's
        default thinking on thinking-capable models (qwen3,
        gemma3n, ...). Omitting the key lets those models think
        by default, which slows structured-output calls and can
        drop fields. Non-thinking models accept `think=False`
        without error (only `think=True` raises "does not
        support thinking", handled by `_with_think_fallback`).
        """
        return {"think": think}

    _THINKING_NOT_SUPPORTED_MSG = "does not support thinking"

    @staticmethod
    def _is_thinking_not_supported(error: ResponseError) -> bool:
        """Check if the error is due to thinking not supported."""
        return OllamaProvider._THINKING_NOT_SUPPORTED_MSG in str(error)

    def _with_think_fallback(
        self, fn: Callable, think: bool, kwargs: dict
    ) -> dict:
        """Call ``fn(**kwargs)``, retrying without ``think``
        if the model does not support it.
        """
        try:
            return fn(**kwargs)
        except ResponseError as e:
            if self._is_thinking_not_supported(e) and think:
                kwargs.pop("think", None)
                return fn(**kwargs)
            raise

    async def _with_think_fallback_async(
        self, fn: Callable, think: bool, kwargs: dict
    ) -> dict:
        """Async version of `_with_think_fallback`."""
        try:
            return await fn(**kwargs)
        except ResponseError as e:
            if self._is_thinking_not_supported(e) and think:
                kwargs.pop("think", None)
                return await fn(**kwargs)
            raise

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

    def _build_call_kwargs(
        self,
        *,
        primary_key: str,
        primary_value: str | list[dict[str, str]],
        options: dict,
        think: bool,
        output_format: str | dict | None,
        stream: bool = False,
        tools: list[dict] | None = None,
    ) -> dict:
        """Build kwargs for an Ollama chat or generate call.

        :param primary_key: ``"messages"`` for chat,
            ``"prompt"`` for generate.
        :param primary_value: The messages list or prompt str.
        :param tools: Optional list of tool schemas to pass
            to the model.
        """
        kwargs = {
            "model": self.model_name,
            primary_key: primary_value,
            "options": options,
            "format": output_format,
            **self._think_kwargs(think),
        }
        if stream:
            kwargs["stream"] = True
        if tools is not None:
            kwargs["tools"] = tools
        return kwargs

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = -1,
        output_format: str | dict = None,
        think: bool = False,
        seed: int | None = None,
        tools: list[dict] | None = None,
    ) -> dict:
        """See `LLMProvider.chat` for documentation."""
        options = self._build_options(temperature, max_tokens, seed)
        kwargs = self._build_call_kwargs(
            primary_key="messages",
            primary_value=messages,
            options=options,
            think=think,
            output_format=output_format,
            tools=tools,
        )
        response = self._with_think_fallback(chat, think, kwargs)
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
        tools: list[dict] | None = None,
    ) -> dict:
        """See `LLMProvider.chat` for documentation."""
        options = self._build_options(temperature, max_tokens, seed)
        kwargs = self._build_call_kwargs(
            primary_key="messages",
            primary_value=messages,
            options=options,
            think=think,
            output_format=output_format,
            tools=tools,
        )
        response = await self._with_think_fallback_async(
            self._async_client.chat, think, kwargs
        )
        parsed = parse_ollama_chat_response(response)

        if self._needs_format_fallback(parsed, output_format):
            response = await self._async_client.chat(
                **{**kwargs, "format": None}
            )
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
        options = self._build_options(temperature, max_tokens, seed)
        kwargs = self._build_call_kwargs(
            primary_key="prompt",
            primary_value=prompt,
            options=options,
            think=think,
            output_format=output_format,
        )
        response = self._with_think_fallback(generate, think, kwargs)
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
        options = self._build_options(temperature, max_tokens, seed)
        kwargs = self._build_call_kwargs(
            primary_key="prompt",
            primary_value=prompt,
            options=options,
            think=think,
            output_format=output_format,
        )
        response = await self._with_think_fallback_async(
            self._async_client.generate, think, kwargs
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
        options = self._build_options(temperature, max_tokens, seed)
        kwargs = self._build_call_kwargs(
            primary_key="messages",
            primary_value=messages,
            options=options,
            think=think,
            output_format=output_format,
            stream=True,
        )
        iterator = self._with_think_fallback(chat, think, kwargs)
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
        options = self._build_options(temperature, max_tokens, seed)
        kwargs = self._build_call_kwargs(
            primary_key="messages",
            primary_value=messages,
            options=options,
            think=think,
            output_format=output_format,
            stream=True,
        )
        iterator = await self._with_think_fallback_async(
            self._async_client.chat, think, kwargs
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
        options = self._build_options(temperature, max_tokens, seed)
        kwargs = self._build_call_kwargs(
            primary_key="prompt",
            primary_value=prompt,
            options=options,
            think=think,
            output_format=output_format,
            stream=True,
        )
        iterator = self._with_think_fallback(generate, think, kwargs)
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
        options = self._build_options(temperature, max_tokens, seed)
        kwargs = self._build_call_kwargs(
            primary_key="prompt",
            primary_value=prompt,
            options=options,
            think=think,
            output_format=output_format,
            stream=True,
        )
        iterator = await self._with_think_fallback_async(
            self._async_client.generate, think, kwargs
        )
        return AsyncStreamResult(iterator, "generate")

    def supports_tools(self) -> bool:
        """Ollama supports tool calling."""
        return True

    def build_tool_turn_messages(
        self,
        assistant_text: str,
        tool_calls: list[dict],
        results: list[str],
    ) -> list[dict]:
        """Build Ollama assistant + tool-result messages.

        See `LLMProvider.build_tool_turn_messages`.
        """
        assistant = {
            "role": "assistant",
            "content": assistant_text,
            "tool_calls": [
                {
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    }
                }
                for call in tool_calls
            ],
        }
        tool_msgs = [
            {
                "role": "tool",
                "content": result,
                "tool_name": call["name"],
            }
            for call, result in zip(tool_calls, results)
        ]
        return [assistant, *tool_msgs]


def _parse_tool_calls(message) -> list[dict]:
    """Extract tool calls from an Ollama message as plain dicts.

    :param message: The ``message`` object from an Ollama chat
        response.
    :return: A list of ``{"name", "arguments"}`` dicts (empty if
        the model requested no tools).
    """
    raw = getattr(message, "tool_calls", None) or []
    return [
        {
            "name": call.function.name,
            "arguments": dict(call.function.arguments),
        }
        for call in raw
    ]


def parse_ollama_chat_response(chat_response: ChatResponse) -> dict:
    """
    Parse the response from the Ollama API.
    :param chat_response: The response object from the Ollama API.
    :return: A dictionary containing the parsed response. Schema:
        - response: The generated text response.
        - role: The role of the message (e.g., "user", "assistant").
        - thinking: The thinking part of the response, if any.
        - tool_calls: List of ``{"name", "arguments"}`` dicts (empty
            if no tools were called).
        - metrics: A dictionary containing the following metrics:
            - tokens: The number of tokens in the response.
            - prompt_tokens: The number of tokens in the prompt.
            - total_duration: The total duration of the response,
                in seconds.
    """
    return {
        "response": (chat_response.message.content or "").strip(),
        "role": chat_response.message.role,
        "thinking": chat_response.message.thinking.strip()
        if chat_response.message.thinking
        else None,
        "tool_calls": _parse_tool_calls(chat_response.message),
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
