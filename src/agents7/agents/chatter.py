"""Conversational agent for multi-turn chat interactions.

`Chatter` extends `BaseAgent` to provide a stateful
conversation interface. Unlike structured-output agents
(Labeler, Detector, etc.), Chatter returns free-form text
responses and maintains message history across turns.

Persona, Scene, and extra_instructions are optional
composition plugins.  Passing a `Persona` (and optionally a
`Scene`) causes the system prompt to be auto-generated from
those objects, overriding any manually supplied
`system_prompt`.  A `Scene` requires a `Persona`.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from agents7.agents.base import BaseAgent
from agents7.agents.config_loader import (
    load_chatter_yaml_settings,
)
from agents7.agents.facts import (
    CharacterFact,
    FactExtractor,
    FactStore,
)
from agents7.agents.jitter import apply_typos
from agents7.agents.memory import MemoryBackend
from agents7.agents.persona import (
    Persona,
    Scene,
    _build_scene_system_prompt,
)
from agents7.call_log import CallRecord, generate_trace_id, now
from agents7.exceptions import (
    AgentConfigurationError,
    ChatEndedError,
)
from agents7.llm.streaming import AsyncStreamResult, StreamResult
from agents7.logs import logger
from agents7.models import AgentResponse, ChatResponse, LLMMetrics

# Instruction sent to the model when the agent is the one to
# speak first. Kept short and bracketed, like the guard's
# feedback messages, so it reads as a stage direction rather
# than as something a person said. Override per agent with
# ``opening_prompt`` — a persona that writes in another
# language is the usual reason to.
_DEFAULT_OPENING_PROMPT = (
    "[Write the first message of this conversation. Nobody has "
    "written to you — you are the one starting it, so open the "
    "way your character would open with a stranger: in your own "
    "words, in the language you always write in, and no longer "
    "than one short message. Do not describe what you are doing "
    "and do not mention this instruction.]"
)

# Placeholder every persona-driven chatter resolves to the
# character's own name. See `Chatter._declare_character_param`.
_CHARACTER_PARAM = "character"


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
        "_messages",
        "_chat_response",
        "_started",
    )

    def __init__(
        self,
        stream: StreamResult | AsyncStreamResult,
        chatter: "Chatter",
        user_message: str,
        messages: list[dict[str, str]] | None = None,
    ) -> None:
        self._stream = stream
        self._chatter = chatter
        self._user_message = user_message
        # The list this stream was opened with. A subclass that
        # regenerates after the fact needs the messages that were
        # actually sent, not a rebuild — an opening turn's differ.
        self._messages = messages or []
        self._chat_response: ChatResponse | None = None
        self._started = time.perf_counter()

    @property
    def chat_response(self) -> ChatResponse | None:
        """Full `ChatResponse`, available after iteration."""
        return self._chat_response

    def _finalize(self) -> None:
        """Build ChatResponse, log the call, update history.

        The record is emitted here rather than when the stream
        opened, because the metrics only exist once the caller
        has finished iterating.
        """
        metrics = LLMMetrics(**self._stream.metrics)
        self._chatter._log_call(metrics)
        self._chat_response = ChatResponse(
            response=self._chatter._append_assistant(self._stream.response),
            thinking=self._stream.thinking,
            metrics=metrics,
        )

    async def _finalize_async(self) -> None:
        """Async finalize hook. Defaults to sync."""
        self._finalize()

    def _chunks(self) -> Iterator[str]:
        """Iterate the provider stream, logging a failure part way.

        A stream that broke mid-reply raised out of the caller's
        loop before `_finalize` ran, so the call left no record.
        Every iteration path, buffered or not, goes through here.
        """
        try:
            yield from self._stream
        except Exception as error:
            self._chatter._log_failed_call(
                error, self._messages, self._started
            )
            raise

    async def _achunks(self) -> AsyncIterator[str]:
        """Async version of `_chunks`."""
        try:
            async for chunk in self._stream:
                yield chunk
        except Exception as error:
            self._chatter._log_failed_call(
                error, self._messages, self._started
            )
            raise

    def __iter__(self) -> Iterator[str]:
        yield from self._chunks()
        self._finalize()

    def __aiter__(self) -> AsyncIterator[str]:
        return self._async_iterate()

    async def _async_iterate(self) -> AsyncIterator[str]:
        async for chunk in self._achunks():
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
        context_params (tuple, optional): `ContextParam`
            declarations for `$variable` values supplied per
            conversation via `set_context` — the nickname of
            the person on the other side, the time of day.
        prompt_variants (dict, optional): Alternative texts for
            a `$variable`, one drawn per conversation: when the
            chatter is built, again on `reset` without a memory
            backend, and from `conversation_id` with one, so a
            conversation rebuilt each turn keeps its draw.
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
        facts (FactStore, optional): Store of durable
            character facts, keyed by `character_id` rather
            than by conversation. When set, the recorded
            facts are folded into the system prompt on every
            turn, so what the character improvised in one
            chat still holds in the next. Facts are stored in
            the character's own language. Defaults to None.
        character_id (str): Key for this character in the
            fact store. Defaults to the agent's `name`.
        fact_extractor (FactExtractor, optional): Agent that
            reads a finished transcript and returns the new
            facts to record. Required by `remember`.
            Defaults to None.
        banned_fact_phrases (tuple[str, ...]): Phrases the
            character must never say. A fact containing one
            is not written by `remember`, and one already in
            the store is left out of the prompt with a
            warning, since a fact outlives the config it was
            recorded under. The store itself is not edited.
            Matching ignores case and spacing. Defaults to
            no phrases.
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
        can_open (bool): Whether this agent may speak first.
            When True, `open_chat` (and its async / streaming
            siblings) produce a turn with nothing to answer —
            the greeting a character with written opening
            lines would send. When False those calls raise
            rather than inventing an opening the agent was
            never given. Defaults to False.
        opening_prompt (str): The instruction that asks the
            model for that first turn. Empty uses a built-in
            English default; set it to match a persona that
            writes in another language. Never stored in
            history. Defaults to empty string.
        opening_instructions (str): Guidance on *how* the
            character opens — the written opening lines, the
            length, the register — appended to the system
            prompt on the opening turn and on nothing else.
            It belongs here rather than in
            `extra_instructions` because a bank of opening
            lines the model can read on every turn is a bank
            it can copy on every turn: a character who has
            already said goodbye greets the same person again
            from the top. Defaults to empty string.
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
        call_logger (Callable, optional): Callable invoked
            with a `CallRecord` for every LLM call a turn
            makes. A plain chat turn is one call; a guarded
            one is at least two — the reply and the guard —
            and every call in a turn shares that turn's
            trace id. Defaults to None (no logging). Use
            `JsonCallLogger` for JSON output.
    """

    temperature: float = field(default=0.7, kw_only=True)
    think: bool = field(default=False, kw_only=True)
    max_history: int | None = field(default=50, kw_only=True)
    memory: MemoryBackend | None = field(default=None, kw_only=True)
    conversation_id: str = field(default="default", kw_only=True)
    facts: FactStore | None = field(default=None, kw_only=True)
    character_id: str = field(default="", kw_only=True)
    fact_extractor: FactExtractor | None = field(default=None, kw_only=True)
    banned_fact_phrases: tuple[str, ...] = field(default=(), kw_only=True)
    max_turns: int | None = field(default=None, kw_only=True)
    end_condition: Callable[[list[dict[str, str]]], bool] | None = field(
        default=None, kw_only=True
    )
    can_open: bool = field(default=False, kw_only=True)
    opening_prompt: str = field(default="", kw_only=True)
    opening_instructions: str = field(default="", kw_only=True)
    typo_rate: float = field(default=0.0, kw_only=True)
    persona: Persona | None = field(default=None, kw_only=True)
    scene: Scene | None = field(default=None, kw_only=True)
    extra_instructions: str = field(default="", kw_only=True)
    call_logger: Callable | None = field(default=None, kw_only=True)

    # Wrapper classes returned by the streaming calls. Not
    # annotated, so the dataclass leaves them alone; subclasses
    # point them at their own wrappers and inherit every
    # streaming call, including the opening one.
    _stream_result_class = ChatterStreamResult
    _async_stream_result_class = ChatterAsyncStreamResult

    def __post_init__(self) -> None:
        if (
            self.persona is None
            and self.scene is not None
            and self.scene != Scene()
        ):
            raise AgentConfigurationError(
                f"Chatter `{self.name}`: a `scene` requires a "
                f"`persona`. The scene frames the persona's "
                f"role-play, so without a persona it would be "
                f"silently ignored."
            )
        if self.persona is not None:
            self._set_attr("system_prompt", self._build_plugin_prompt())
            self._declare_character_param()
        if self.opening_instructions:
            # A YAML block scalar keeps its trailing newline, and
            # this text is spliced into the middle of a prompt.
            self._set_attr(
                "opening_instructions",
                self.opening_instructions.strip(),
            )
        super().__post_init__()
        if self.memory is not None:
            # A conversation kept in memory may be rebuilt for every
            # turn, and each rebuild must show the variants the first
            # turn showed. Its id is what the rebuilds share.
            self.draw_variants(seed=self.conversation_id)
            history = self.memory.load(self.conversation_id)
            turns = sum(1 for m in history if m["role"] == "assistant")
        else:
            history = []
            turns = 0
        self._set_attr("_history", history)
        self._set_attr("_turns", turns)
        if not self.character_id:
            self._set_attr("character_id", self.name)
        if any(not p.strip() for p in self.banned_fact_phrases):
            raise AgentConfigurationError(
                f"Chatter `{self.name}`: a blank entry in "
                f"`banned_fact_phrases` would match every fact "
                f"and leave the prompt with none."
            )
        self._set_attr("_facts", self._load_facts())
        self._set_attr("_trace_id", generate_trace_id())

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
            can_open: <optional, default false — may speak first>
            opening_prompt: <optional, asks for that first turn>
            opening_instructions: <optional, how the character
                    opens; in the prompt on that turn only>
            prompt_params:
                <key>: <value>
            context_params:      # supplied per conversation
                <key>:
                    default: <fallback value>
                    description: <what it means>
                    callable: <dotted path, called each turn>
                    from: <another key, passed to the callable>
            prompt_variants:     # one drawn per conversation
                <key>:
                    - <text>
                    - <another text>
            memory:
                backend: <json | sqlite>
                path: <file path>
            call_log:
                backend: <json | sqlite>
                path: <file path>
            conversation_id: <optional, default "default">
            character_facts:     # durable facts per character
                path: <folder for the fact files; a relative
                       one sits beside this config file>
                character_id: <optional, default the name>
                banned_phrases: <optional list; facts saying
                        one are never stored or prompted>
                extractor:
                    engine:
                        provider: <llm provider>
                        model: <llm model>
            persona:        # optional — auto-generates the prompt
                full_name: <name>
                ...
            scene:          # optional — requires a persona
                setting: <...>
                ...
            extra_instructions: <optional, appended to prompt>
            ```
            When a ``persona`` (and optionally ``scene``) section
            is present, ``system_prompt`` is auto-generated and
            need not be provided.
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
        self._log_turn("chat turn")
        return self._respond(self._build_messages(), message)

    async def chat_async(self, message: str) -> ChatResponse:
        """Async version of `chat`.

        :param message: The user's message.
        :return: The assistant's ChatResponse.
        :raises ChatEndedError: If the conversation has ended.
        """
        self._check_ended()
        self._append_user(message)
        self._log_turn("async chat turn")
        return await self._respond_async(
            self._build_messages(),
            message,
        )

    def open_chat(self) -> ChatResponse:
        """Speak first, with nothing to answer.

        Produces the opening turn of a conversation: the model
        is asked for a first message, and only its reply enters
        history, so the transcript starts with the agent. Use
        it before the first `chat` call, on an agent whose
        persona says how it opens.

        :return: The agent's opening ChatResponse.
        :raises AgentConfigurationError: If `can_open` is
            False.
        :raises ChatEndedError: If the conversation has ended.
        """
        self._check_can_open()
        self._check_ended()
        self._log_turn("opening turn")
        return self._respond(self._opening_messages(), "")

    async def open_chat_async(self) -> ChatResponse:
        """Async version of `open_chat`.

        :return: The agent's opening ChatResponse.
        :raises AgentConfigurationError: If `can_open` is
            False.
        :raises ChatEndedError: If the conversation has ended.
        """
        self._check_can_open()
        self._check_ended()
        self._log_turn("async opening turn")
        return await self._respond_async(
            self._opening_messages(),
            "",
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
        self._log_turn("streaming chat turn")
        return self._start_stream(self._build_messages(), message)

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
        self._log_turn("async streaming chat turn")
        return await self._start_stream_async(
            self._build_messages(),
            message,
        )

    def open_chat_stream(self) -> ChatterStreamResult:
        """Streaming version of `open_chat`.

        :return: A ChatterStreamResult yielding text chunks.
        :raises AgentConfigurationError: If `can_open` is
            False.
        :raises ChatEndedError: If the conversation has ended.
        """
        self._check_can_open()
        self._check_ended()
        self._log_turn("streaming opening turn")
        return self._start_stream(self._opening_messages(), "")

    async def open_chat_stream_async(
        self,
    ) -> ChatterAsyncStreamResult:
        """Async streaming version of `open_chat`.

        :return: A ChatterAsyncStreamResult yielding text
            chunks.
        :raises AgentConfigurationError: If `can_open` is
            False.
        :raises ChatEndedError: If the conversation has ended.
        """
        self._check_can_open()
        self._check_ended()
        self._log_turn("async streaming opening turn")
        return await self._start_stream_async(
            self._opening_messages(),
            "",
        )

    def reset(self) -> None:
        """Clear the conversation history.

        If a memory backend is configured, also clears the
        persisted history for this conversation, which keeps its
        id and so its `prompt_variants`. Without one, the next
        conversation draws its variants afresh.
        """
        self._history.clear()
        self._set_attr("_turns", 0)
        if self.memory is not None:
            self.memory.clear(self.conversation_id)
        else:
            self.draw_variants()

    def remember(self) -> list[CharacterFact]:
        """Record what this conversation revealed about the
        character.

        Runs `fact_extractor` over the current history and
        writes the new facts to `facts` under `character_id`,
        skipping anything already recorded. Call it when a
        conversation is finished — it makes an LLM call, so it
        does not belong inside a chat turn.

        The recorded facts are folded into the system prompt
        of every later turn, including the rest of this
        conversation.

        Facts are written in the character's own language,
        taken from `persona.language`. Without a persona, or
        without a declared language, the extractor follows
        the language of the transcript.

        :return: The facts that were written. Empty when the
            conversation revealed nothing new.
        :raises AgentConfigurationError: If no fact store or
            no fact extractor is configured.
        """
        self._check_can_remember()
        if not self._history:
            return []
        new_facts = self._logged_extractor().extract(
            character=self._character_name(),
            transcript=self._history,
            known_facts=[fact.text for fact in self._facts],
            language=self._character_language(),
        )
        return self._store_facts(new_facts)

    async def remember_async(self) -> list[CharacterFact]:
        """Async version of `remember`.

        :return: The facts that were written.
        :raises AgentConfigurationError: If no fact store or
            no fact extractor is configured.
        """
        self._check_can_remember()
        if not self._history:
            return []
        new_facts = await self._logged_extractor().extract_async(
            character=self._character_name(),
            transcript=self._history,
            known_facts=[fact.text for fact in self._facts],
            language=self._character_language(),
        )
        return self._store_facts(new_facts)

    def _logged_extractor(self) -> FactExtractor:
        """The fact extractor, reporting to this chatter's log.

        An extractor declared with its own `call_logger` keeps
        it. Otherwise it borrows this chatter's, read at call
        time rather than at construction, because an eval run
        swaps the logger on an agent that already exists.
        Without this, no extraction reached the call log, and
        an unparseable one — which returns no facts — looked
        exactly like a conversation that revealed nothing.

        :return: The extractor to run.
        """
        extractor = self.fact_extractor
        if extractor.call_logger is not None or self.call_logger is None:
            return extractor
        return extractor.with_overrides(call_logger=self.call_logger)

    def forget(self) -> None:
        """Erase every recorded fact for this character.

        Clears the fact store entry for `character_id` and
        drops the facts from the system prompt. Does not
        touch the conversation history — use `reset` for
        that.
        """
        if self.facts is None:
            return
        self.facts.clear(self.character_id)
        self._set_attr("_facts", [])

    @property
    def character_facts(self) -> list[CharacterFact]:
        """Facts recorded for this character.

        These are folded into the system prompt on every
        turn.

        :return: List of `CharacterFact`, oldest first.
        """
        return list(self._facts)

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
        if self.can_open:
            lines.append("Can open a chat: yes")
        if self.max_turns is not None:
            lines.append(f"Max turns: {self.max_turns}")
        if self.end_condition is not None:
            lines.append("End condition: custom")
        if self.facts is not None:
            lines.append(
                f"Character facts: {len(self._facts)} recorded "
                f"for `{self.character_id}`"
            )
        return "\n".join(lines)

    # --- Private helpers ---

    def _declare_character_param(self) -> None:
        """Make `$character` resolve to the persona's own name.

        Text that has to name the character — the speaker label
        on a worked example, most of all — can then be written
        once and stay right when the character is swapped. That
        is what makes a `chat_goal` file reusable: its examples
        show the moves without asserting who is making them, so
        the same goal reads correctly under David and under
        anyone else.

        A config that declares `character` itself wins, and so
        does a `context_params` entry of that name, since the
        agent was told about that one on purpose.
        """
        if not self.persona.full_name:
            return
        declared = {p.name for p in self.context_params or ()}
        if _CHARACTER_PARAM in declared:
            return
        params = dict(self.prompt_params or {})
        params.setdefault(_CHARACTER_PARAM, self.persona.full_name)
        self._set_attr("prompt_params", params)

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

    def _log_turn(self, what: str) -> None:
        """Log the start of a turn at debug level."""
        logger.debug(
            f"{type(self).__name__} `{self.name}`: {what} "
            f"({len(self._history)} messages in history)"
        )

    def _new_turn(self) -> None:
        """Start a new trace for the turn about to be made.

        Every LLM call a single turn makes shares one trace id
        — the reply, each guard check, and each regeneration —
        so a consumer can group a turn's records and read what
        the whole turn cost. Called once per turn, as the first
        call is about to go out.
        """
        self._set_attr("_trace_id", generate_trace_id())

    def _log_call(
        self,
        metrics: LLMMetrics,
        *,
        name: str | None = None,
        version: int | None = None,
        model_id: str | None = None,
        success: bool = True,
        prompt: str | None = None,
        error: str | None = None,
    ) -> None:
        """Emit a `CallRecord` for one LLM call, if logging.

        A chat reply has no parse step, so a call that returned
        is a success. The guard's calls are logged through here
        too — see `GuardedChatter` — which is why the agent
        identity is overridable: a guard usually runs its own
        model, and folding its tokens into the chatter's record
        would report one call made on two models.

        :param metrics: The call's own metrics.
        :param name: Agent name for the record, defaulting to
            this chatter's.
        :param version: Agent version, defaulting to this
            chatter's.
        :param model_id: Resolved model id, defaulting to this
            chatter's engine.
        :param success: Whether the call produced a usable
            result.
        :param prompt: The messages sent, as a JSON string, on
            a call that failed — see `CallRecord.prompt`.
        :param error: What went wrong, on a call that failed.
        """
        if self.call_logger is None:
            return
        self.call_logger(
            CallRecord(
                trace_id=self._trace_id,
                # The conversation is the run: a chatter always
                # knows which one it is in, so its records are
                # never the anonymous kind.
                run_id=self.conversation_id,
                recorded_at=now(),
                agent_name=name or self.name,
                agent_version=(self.version if version is None else version),
                model_id=model_id or self.engine.model_id,
                success=success,
                tokens=metrics.tokens,
                prompt_tokens=metrics.prompt_tokens,
                cached_prompt_tokens=metrics.cached_prompt_tokens,
                total_duration=metrics.total_duration,
                cost_cents=metrics.cost_cents,
                cached=metrics.cached,
                model=metrics.model,
                retries=metrics.retries,
                retry_duration=metrics.retry_duration,
                prompt=prompt,
                error=error,
            )
        )

    def _log_failed_call(
        self,
        error: BaseException,
        sent: Any,
        started: float,
        *,
        name: str | None = None,
        version: int | None = None,
        model_id: str | None = None,
    ) -> None:
        """Emit a `CallRecord` for a call that raised.

        Without this a call that raised left no row: a run could
        lose whole conversations to ``Connection error.`` while its
        call log showed every call as a success.

        Nothing came back, so no tokens were counted and the
        provider gave no duration. The duration recorded is how
        long this side waited before the error, and the cost is
        left unmeasured (``None``) rather than written as zero.
        Retries come from the error when the dispatcher attached
        them — see `attach_attempt_stats` — and are zero when it
        did not, for an error raised before any retry loop ran.

        :param error: What the call raised.
        :param sent: What was sent — the messages, or a guard's
            input — kept as the failed call's prompt.
        :param started: `time.perf_counter` when the call began.
        :param name: Agent name for the record, as in `_log_call`.
        :param version: Agent version, as in `_log_call`.
        :param model_id: Configured model id, as in `_log_call`.
        """
        self._log_call(
            LLMMetrics(
                tokens=0,
                prompt_tokens=0,
                total_duration=time.perf_counter() - started,
                cost_cents=None,
                retries=getattr(error, "retries", 0),
                retry_duration=getattr(error, "retry_duration", 0.0),
            ),
            name=name,
            version=version,
            model_id=model_id,
            success=False,
            prompt=json.dumps(sent, ensure_ascii=False, default=str),
            error=f"{type(error).__name__}: {error}",
        )

    def _respond(
        self,
        messages: list[dict[str, str]],
        user_message: str,
    ) -> ChatResponse:
        """Call the LLM, store the reply, and return it.

        The single point every turn goes through — answering
        one and opening one alike — so a subclass that
        post-processes replies (see `GuardedChatter`) overrides
        this instead of each public call.

        :param messages: The full message list for the engine.
        :param user_message: What was said to prompt this
            reply, or ``""`` for an opening turn. Unused here;
            subclasses use it as validation context.
        :return: The assistant's ChatResponse.
        """
        self._new_turn()
        started = time.perf_counter()
        try:
            response = self.engine.chat(
                messages=messages,
                temperature=self.temperature,
                think=self.think,
            )
        except Exception as error:
            self._log_failed_call(error, messages, started)
            raise
        response.pop("role", None)
        text = self._append_assistant(self._maybe_typo(response["response"]))
        metrics = LLMMetrics(**response["metrics"])
        self._log_call(metrics)
        return ChatResponse(
            response=text,
            thinking=response.get("thinking"),
            metrics=metrics,
        )

    async def _respond_async(
        self,
        messages: list[dict[str, str]],
        user_message: str,
    ) -> ChatResponse:
        """Async version of `_respond`."""
        self._new_turn()
        started = time.perf_counter()
        try:
            response = await self.engine.chat_async(
                messages=messages,
                temperature=self.temperature,
                think=self.think,
            )
        except Exception as error:
            self._log_failed_call(error, messages, started)
            raise
        response.pop("role", None)
        text = self._append_assistant(self._maybe_typo(response["response"]))
        metrics = LLMMetrics(**response["metrics"])
        self._log_call(metrics)
        return ChatResponse(
            response=text,
            thinking=response.get("thinking"),
            metrics=metrics,
        )

    def _start_stream(
        self,
        messages: list[dict[str, str]],
        user_message: str,
    ) -> ChatterStreamResult:
        """Open a stream and wrap it for this agent class."""
        self._new_turn()
        started = time.perf_counter()
        try:
            stream = self.engine.chat_stream(
                messages=messages,
                temperature=self.temperature,
                think=self.think,
            )
        except Exception as error:
            self._log_failed_call(error, messages, started)
            raise
        return self._stream_result_class(
            stream,
            self,
            user_message,
            messages,
        )

    async def _start_stream_async(
        self,
        messages: list[dict[str, str]],
        user_message: str,
    ) -> ChatterAsyncStreamResult:
        """Async version of `_start_stream`."""
        self._new_turn()
        started = time.perf_counter()
        try:
            stream = await self.engine.chat_stream_async(
                messages=messages,
                temperature=self.temperature,
                think=self.think,
            )
        except Exception as error:
            self._log_failed_call(error, messages, started)
            raise
        return self._async_stream_result_class(
            stream,
            self,
            user_message,
            messages,
        )

    def _check_can_open(self) -> None:
        """Raise unless this agent is allowed to speak first."""
        if not self.can_open:
            raise AgentConfigurationError(
                f"Chatter `{self.name}`: opening a chat needs "
                f"`can_open=True`. This agent is configured to "
                f"answer, not to start a conversation."
            )

    def _opening_messages(self) -> list[dict[str, str]]:
        """Build the message list for a turn the agent starts.

        The instruction goes in as a user message rather than
        being left out: a call carrying only a system prompt is
        rejected by some providers, and every one of them
        answers a conversation whose last turn is the user's.
        It is never appended to history, so the stored
        transcript still begins with the agent's own line.

        `opening_instructions` joins the system prompt here and
        only here — see the field's own note for why it is not
        left in `extra_instructions`.

        :return: Messages list for the engine call.
        """
        return [
            *self._build_messages(self.opening_instructions),
            {
                "role": "user",
                "content": self.substitute(
                    self.opening_prompt or _DEFAULT_OPENING_PROMPT
                ),
            },
        ]

    def _check_ended(self) -> None:
        """Raise ChatEndedError if the conversation is over."""
        if self.ended:
            raise ChatEndedError(
                f"Chatter `{self.name}`: conversation "
                f"`{self.conversation_id}` has ended."
            )

    def _build_messages(
        self,
        extra_system: str = "",
    ) -> list[dict[str, str]]:
        """Build the full message list for the LLM call.

        Prepends the system prompt and appends the
        (possibly trimmed) conversation history.

        :param extra_system: Text appended to the system
            prompt for this call only. Used by the opening
            turn; empty on every other one.
        :return: Messages list for engine.chat().
        """
        system_prompt = self.resolved_system_prompt
        if facts_section := self._facts_section():
            system_prompt = f"{system_prompt}\n\n{facts_section}"
        if extra_system:
            resolved_extra = self.substitute(extra_system)
            system_prompt = f"{system_prompt}\n\n{resolved_extra}"
        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            *self._history,
        ]

    def _check_can_remember(self) -> None:
        """Raise unless a store and an extractor are set."""
        missing = []
        if self.facts is None:
            missing.append("`facts` store")
        if self.fact_extractor is None:
            missing.append("`fact_extractor`")
        if missing:
            raise AgentConfigurationError(
                f"Chatter `{self.name}`: remember() needs "
                f"{' and '.join(missing)}. Nothing was "
                f"recorded."
            )

    def _character_name(self) -> str:
        """Name to describe the character to the extractor."""
        if self.persona is not None and self.persona.full_name:
            return self.persona.full_name
        return self.character_id

    def _character_language(self) -> str:
        """Language the extractor should write the facts in.

        Taken from the persona, which already declares the
        language the character always answers in. Empty when
        there is no persona or it declares none, which leaves
        the extractor to follow the transcript.
        """
        if self.persona is None:
            return ""
        return self.persona.language

    def _store_facts(self, texts: list[str]) -> list[CharacterFact]:
        """Write extracted facts and refresh the prompt copy.

        Facts containing a banned phrase are dropped first.
        """
        texts = [t for t in texts if self._banned_phrase_in(t) is None]
        if not texts:
            return []
        added = self.facts.add(
            self.character_id,
            [
                CharacterFact(
                    text=text,
                    source_conversation=self.conversation_id,
                )
                for text in texts
            ],
        )
        if added:
            self._set_attr("_facts", self._facts + added)
            logger.info(
                f"Chatter `{self.name}`: recorded "
                f"{len(added)} new fact(s) about "
                f"`{self.character_id}`."
            )
        return added

    def _load_facts(self) -> list[CharacterFact]:
        """Read this character's facts from the store.

        A stored fact containing a banned phrase is left out
        and logged, so the operator can remove it by hand.
        """
        if self.facts is None:
            return []
        kept = []
        for fact in self.facts.load(self.character_id):
            phrase = self._banned_phrase_in(fact.text)
            if phrase is None:
                kept.append(fact)
                continue
            logger.warning(
                f"Chatter `{self.name}`: left fact `{fact.text}` "
                f"about `{self.character_id}` out of the prompt "
                f"because it contains the banned phrase "
                f"`{phrase}`. Remove it from the fact store."
            )
        return kept

    def _banned_phrase_in(self, text: str) -> str | None:
        """Return the first banned phrase *text* contains.

        :param text: A fact sentence.
        :return: The matching phrase as configured, or None.
        """
        haystack = " ".join(text.lower().split())
        for phrase in self.banned_fact_phrases:
            if " ".join(phrase.lower().split()) in haystack:
                return phrase
        return None

    def _facts_section(self) -> str:
        """Render the recorded facts as a system-prompt section.

        :return: The section text, or ``""`` when the
            character has no recorded facts.
        """
        if not self._facts:
            return ""
        lines = [
            "## Things You Have Already Said About Yourself",
            "These came up in earlier conversations. They are "
            "true of you, so stay consistent with them and "
            "never contradict them. Do not list them, and do "
            "not announce that you are remembering something "
            "— just talk as somebody for whom they are "
            "already true.",
        ]
        lines.extend(f"- {fact.text}" for fact in self._facts)
        return "\n".join(lines)

    def _maybe_typo(self, text: str) -> str:
        """Apply typos if typo_rate > 0."""
        if self.typo_rate <= 0:
            return text
        return apply_typos(text, self.typo_rate)

    def _append_user(self, message: str) -> None:
        """Append a user message to history."""
        self._history.append({"role": "user", "content": message})
        self._trim_history()

    def _append_assistant(self, message: str) -> str:
        """Store a finished reply in history, without empty lines.

        Every reply path ends here, so this is where the reply
        takes its final shape. A model that separates two short
        messages with an empty line would otherwise hand a chat
        client an empty message between them; the lines a client
        renders are kept exactly as written.

        :param message: The reply as generated.
        :return: The reply as stored — what the caller returns.
        """
        message = "\n".join(
            line for line in message.split("\n") if line.strip()
        )
        self._history.append({"role": "assistant", "content": message})
        self._set_attr("_turns", self._turns + 1)
        self._trim_history()
        self._persist()
        return message

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
