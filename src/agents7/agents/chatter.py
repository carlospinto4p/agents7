"""Conversational agent for multi-turn chat interactions.

`Chatter` extends `BaseAgent` to provide a stateful
conversation interface. Unlike structured-output agents
(Labeler, Detector, etc.), Chatter returns free-form text
responses and maintains message history across turns.

Persona, Scene, and extra_instructions are optional
composition plugins.  Passing a `Persona` (and optionally a
`Scene`) causes the system prompt to be auto-generated from
those objects, overriding any manually supplied
`system_prompt`.  Subclasses (`PersonalityChatter`,
`SceneChatter`, etc.) are thin wrappers that enforce field
presence via re-declaration.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from agents7.agents.base import BaseAgent
from agents7.agents.chatter_config_loader import (
    load_chatter_yaml_settings,
)
from agents7.agents.jitter import apply_typos
from agents7.agents.memory import MemoryBackend
from agents7.agents.persona import (
    Persona,
    Scene,
    _build_scene_system_prompt,
)
from agents7.exceptions import ChatEndedError
from agents7.llm.streaming import AsyncStreamResult, StreamResult
from agents7.logs import logger
from agents7.models import AgentResponse, ChatResponse, LLMMetrics


class _ChatterStreamBase:
    """Shared state and logic for chatter stream wrappers.

    Provides ``__iter__`` (sync) and ``__aiter__`` (async)
    iteration that yields text chunks and calls the
    appropriate finalize hook when done.  Subclasses
    override ``_finalize`` or ``_finalize_async`` to add
    post-stream logic (e.g. guard validation).
    """

    __slots__ = (
        "_stream",
        "_chatter",
        "_user_message",
        "_chat_response",
    )

    def __init__(
        self,
        stream: StreamResult | AsyncStreamResult,
        chatter: "Chatter",
        user_message: str,
    ) -> None:
        self._stream = stream
        self._chatter = chatter
        self._user_message = user_message
        self._chat_response: ChatResponse | None = None

    @property
    def chat_response(self) -> ChatResponse | None:
        """Full `ChatResponse`, available after iteration."""
        return self._chat_response

    def _finalize(self) -> None:
        """Build ChatResponse and update history."""
        self._chat_response = ChatResponse(
            response=self._stream.response,
            thinking=self._stream.thinking,
            metrics=LLMMetrics(**self._stream.metrics),
        )
        self._chatter._append_assistant(
            self._stream.response,
        )

    async def _finalize_async(self) -> None:
        """Async finalize hook. Defaults to sync."""
        self._finalize()

    def __iter__(self) -> Iterator[str]:
        for chunk in self._stream:
            yield chunk
        self._finalize()

    def __aiter__(self) -> AsyncIterator[str]:
        return self._async_iterate()

    async def _async_iterate(self) -> AsyncIterator[str]:
        async for chunk in self._stream:
            yield chunk
        await self._finalize_async()


class ChatterStreamResult(_ChatterStreamBase):
    """Sync streaming wrapper for Chatter responses.

    Yields text chunks during iteration. After iteration
    completes, access the `chat_response` property for the
    full `ChatResponse`. The Chatter's history is updated
    automatically when the stream finishes.

    Usage::

        stream = chatter.chat_stream("Hello!")
        for chunk in stream:
            print(chunk, end="", flush=True)
        print(stream.chat_response)
    """

    __slots__ = ()


class ChatterAsyncStreamResult(_ChatterStreamBase):
    """Async streaming wrapper for Chatter responses.

    Yields text chunks during async iteration. After
    iteration completes, access the `chat_response` property
    for the full `ChatResponse`.

    Usage::

        stream = await chatter.chat_stream_async("Hello!")
        async for chunk in stream:
            print(chunk, end="", flush=True)
        print(stream.chat_response)
    """

    __slots__ = ()


@dataclass(frozen=True)
class Chatter(BaseAgent):
    """Conversational agent for multi-turn chat.

    Maintains message history and uses the system prompt
    as the agent's personality. Unlike Labeler and its
    subclasses, Chatter returns free-form text (no
    structured output parsing).

    Attributes:
        name (str): The name of the agent.
        version (int): The version of the agent.
        engine (LLMEngine): The LLM engine used by the agent.
        system_prompt (str): The personality / system prompt.
            Overridden automatically when `persona` is set.
        prompt_params (dict, optional): Substitution params
            for `$variable` placeholders in system_prompt.
        temperature (float): Sampling temperature.
            Defaults to 0.7.
        think (bool): Enable thinking mode. Defaults to
            False.
        max_history (int, optional): Maximum number of message
            pairs (user + assistant) to keep. Older messages
            are trimmed from the front. None means unlimited.
            Defaults to 50.
        memory (MemoryBackend, optional): Pluggable memory
            backend for persisting conversation history
            across sessions. When set, history is loaded on
            init and auto-saved after each turn. Defaults
            to None (in-memory only).
        conversation_id (str): Identifier for the
            conversation in the memory backend. Defaults
            to ``"default"``.
        max_turns (int, optional): Maximum number of
            complete turns (user + assistant pairs). When
            reached, further chat calls raise
            ``ChatEndedError``. None means unlimited.
            Defaults to None.
        end_condition (Callable, optional): A callable
            that receives the conversation history
            (list of message dicts) and returns True to
            end the conversation. Checked before each
            turn. Defaults to None.
        typo_rate (float): Probability (0.0-1.0) that
            each word in the response gets a random typo.
            0.0 means perfect text, 0.03 means ~3% of
            words get typos. Applies to ``chat()`` and
            ``chat_async()`` only (not streaming).
            Defaults to 0.0.
        persona (Persona, optional): Character definition.
            When set, the system prompt is auto-generated
            from the persona (and scene if also set).
            Defaults to None.
        scene (Scene, optional): Scenario definition.
            When set alongside `persona`, the system prompt
            is built as a role-play scene. Defaults to None.
        extra_instructions (str): Additional instructions
            appended after the persona/scene prompt.
            Defaults to empty string.
    """

    temperature: float = field(default=0.7, kw_only=True)
    think: bool = field(default=False, kw_only=True)
    max_history: int | None = field(default=50, kw_only=True)
    memory: MemoryBackend | None = field(default=None, kw_only=True)
    conversation_id: str = field(default="default", kw_only=True)
    max_turns: int | None = field(default=None, kw_only=True)
    end_condition: Callable[[list[dict[str, str]]], bool] | None = field(
        default=None, kw_only=True
    )
    typo_rate: float = field(default=0.0, kw_only=True)
    persona: Persona | None = field(default=None, kw_only=True)
    scene: Scene | None = field(default=None, kw_only=True)
    extra_instructions: str = field(default="", kw_only=True)

    def __post_init__(self) -> None:
        if self.persona is not None:
            self._set_attr("system_prompt", self._build_plugin_prompt())
        super().__post_init__()
        if self.memory is not None:
            history = self.memory.load(self.conversation_id)
            turns = sum(1 for m in history if m["role"] == "assistant")
        else:
            history = []
            turns = 0
        self._set_attr("_history", history)
        self._set_attr("_turns", turns)

    @classmethod
    def from_yaml(cls, file_path: str) -> "Chatter":
        """Create a Chatter from a YAML configuration file.

        :param file_path: Path to the YAML file. Schema:
            ```yaml
            name: <name>
            version: <version>
            engine:
                provider: <provider>
                model: <model>
            system_prompt: <personality prompt>
            temperature: <optional, default 0.7>
            think: <optional, default false>
            max_history: <optional, default 50>
            prompt_params:
                <key>: <value>
            memory:
                backend: <json | sqlite>
                path: <file path>
            conversation_id: <optional, default "default">
            ```
        :return: A configured Chatter instance.
        """
        config = load_chatter_yaml_settings(file_path)
        return Chatter(**config)

    def act(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> AgentResponse:
        """Perform a single chat turn via the BaseAgent
        interface.

        Accepts a string as the first positional argument
        (the user message). Returns an `AgentResponse` with
        `model=None` (no structured output).

        :param args: First positional arg is the user message.
        :param kwargs: Passed to the engine's chat method.
        :return: AgentResponse with the assistant's reply.
        """
        message = args[0] if args else kwargs.get("message", "")
        chat_response = self.chat(str(message))
        return AgentResponse(
            response=chat_response.response,
            model=None,
            thinking=chat_response.thinking,
            metrics=chat_response.metrics,
        )

    def chat(self, message: str) -> ChatResponse:
        """Send a message and get a response.

        Appends the user message to history, calls the LLM
        with the full conversation context, appends the
        assistant response, and returns a `ChatResponse`.

        :param message: The user's message.
        :return: The assistant's ChatResponse.
        :raises ChatEndedError: If the conversation has ended.
        """
        self._check_ended()
        self._append_user(message)
        messages = self._build_messages()

        logger.debug(
            f"Chatter `{self.name}`: chat turn "
            f"({len(self._history)} messages in history)"
        )
        response = self.engine.chat(
            messages=messages,
            temperature=self.temperature,
            think=self.think,
        )
        response.pop("role", None)
        text = self._maybe_typo(response["response"])
        self._append_assistant(text)

        return ChatResponse(
            response=text,
            thinking=response.get("thinking"),
            metrics=LLMMetrics(**response["metrics"]),
        )

    async def chat_async(self, message: str) -> ChatResponse:
        """Async version of `chat`.

        :param message: The user's message.
        :return: The assistant's ChatResponse.
        :raises ChatEndedError: If the conversation has ended.
        """
        self._check_ended()
        self._append_user(message)
        messages = self._build_messages()

        logger.debug(
            f"Chatter `{self.name}`: async chat turn "
            f"({len(self._history)} messages in history)"
        )
        response = await self.engine.chat_async(
            messages=messages,
            temperature=self.temperature,
            think=self.think,
        )
        response.pop("role", None)
        text = self._maybe_typo(response["response"])
        self._append_assistant(text)

        return ChatResponse(
            response=text,
            thinking=response.get("thinking"),
            metrics=LLMMetrics(**response["metrics"]),
        )

    def chat_stream(
        self,
        message: str,
    ) -> ChatterStreamResult:
        """Stream a chat response, yielding text chunks.

        Returns a `ChatterStreamResult` that yields text
        chunks when iterated. After iteration, access the
        full `ChatResponse` via `stream.chat_response`.
        History is updated automatically after the stream
        completes.

        :param message: The user's message.
        :return: A ChatterStreamResult yielding text chunks.
        :raises ChatEndedError: If the conversation has ended.
        """
        self._check_ended()
        self._append_user(message)
        messages = self._build_messages()

        logger.debug(
            f"Chatter `{self.name}`: streaming chat turn "
            f"({len(self._history)} messages in history)"
        )
        stream = self.engine.chat_stream(
            messages=messages,
            temperature=self.temperature,
            think=self.think,
        )
        return ChatterStreamResult(stream, self, message)

    async def chat_stream_async(
        self,
        message: str,
    ) -> ChatterAsyncStreamResult:
        """Async version of `chat_stream`.

        :param message: The user's message.
        :return: A ChatterAsyncStreamResult yielding text
            chunks.
        :raises ChatEndedError: If the conversation has ended.
        """
        self._check_ended()
        self._append_user(message)
        messages = self._build_messages()

        logger.debug(
            f"Chatter `{self.name}`: async streaming chat "
            f"turn ({len(self._history)} messages in history)"
        )
        stream = await self.engine.chat_stream_async(
            messages=messages,
            temperature=self.temperature,
            think=self.think,
        )
        return ChatterAsyncStreamResult(stream, self, message)

    def reset(self) -> None:
        """Clear the conversation history.

        If a memory backend is configured, also clears the
        persisted history for this conversation.
        """
        self._history.clear()
        self._set_attr("_turns", 0)
        if self.memory is not None:
            self.memory.clear(self.conversation_id)

    @property
    def history(self) -> list[dict[str, str]]:
        """Copy of the current conversation history.

        Does not include the system prompt (which is
        prepended automatically at call time).

        :return: List of message dicts with 'role' and
            'content' keys.
        """
        return list(self._history)

    @property
    def turns(self) -> int:
        """Number of complete turns (user + assistant pairs).

        :return: Number of completed turns.
        """
        return self._turns

    @property
    def ended(self) -> bool:
        """Whether the conversation has ended.

        :return: True if max_turns reached or end_condition
            returns True.
        """
        if self.max_turns is not None and self.turns >= self.max_turns:
            return True
        if self.end_condition is not None and self.end_condition(
            self._history
        ):
            return True
        return False

    def describe(self) -> str:
        """Human-readable summary of the agent.

        :return: Multi-line string describing the Chatter.
        """
        lines = [
            f"Chatter: {self.name} (v{self.version})",
            f"Model: {self.engine.model_id}",
            f"Temperature: {self.temperature}",
            f"Max history: {self.max_history} turns",
        ]
        if self.max_turns is not None:
            lines.append(f"Max turns: {self.max_turns}")
        if self.end_condition is not None:
            lines.append("End condition: custom")
        return "\n".join(lines)

    # --- Private helpers ---

    def _build_plugin_prompt(self) -> str:
        """Build system prompt from persona and/or scene."""
        if self.scene is not None:
            return _build_scene_system_prompt(
                self.persona,
                self.scene,
                self.extra_instructions,
            )
        prompt = self.persona.to_prompt()
        if self.extra_instructions:
            prompt = (
                f"{prompt}\n\n"
                f"## Extra Instructions\n"
                f"{self.extra_instructions}"
            )
        return prompt

    def _check_ended(self) -> None:
        """Raise ChatEndedError if the conversation is over."""
        if self.ended:
            raise ChatEndedError(
                f"Chatter `{self.name}`: conversation "
                f"`{self.conversation_id}` has ended."
            )

    def _build_messages(self) -> list[dict[str, str]]:
        """Build the full message list for the LLM call.

        Prepends the system prompt and appends the
        (possibly trimmed) conversation history.

        :return: Messages list for engine.chat().
        """
        return [
            {
                "role": "system",
                "content": self.resolved_system_prompt,
            },
            *self._history,
        ]

    def _maybe_typo(self, text: str) -> str:
        """Apply typos if typo_rate > 0."""
        if self.typo_rate <= 0:
            return text
        return apply_typos(text, self.typo_rate)

    def _append_user(self, message: str) -> None:
        """Append a user message to history."""
        self._history.append({"role": "user", "content": message})
        self._trim_history()

    def _append_assistant(self, message: str) -> None:
        """Append an assistant message to history."""
        self._history.append({"role": "assistant", "content": message})
        self._set_attr("_turns", self._turns + 1)
        self._trim_history()
        self._persist()

    def _trim_history(self) -> None:
        """Trim history to max_history message pairs.

        Removes the oldest messages from the front,
        keeping at most ``max_history * 2`` individual
        messages (each turn = 1 user + 1 assistant).
        Does nothing when ``max_history`` is None.
        """
        if self.max_history is None:
            return
        max_messages = self.max_history * 2
        excess = len(self._history) - max_messages
        if excess <= 0:
            return
        # Compute removed assistant turns arithmetically from the first
        # message role; avoids scanning the excess slice.
        if self._history[0]["role"] == "assistant":
            removed = (excess + 1) // 2
        else:
            removed = excess // 2
        del self._history[:excess]
        if removed:
            self._set_attr("_turns", self._turns - removed)

    def _persist(self) -> None:
        """Save history to the memory backend if set."""
        if self.memory is not None:
            self.memory.save(
                self.conversation_id,
                self._history,
            )
