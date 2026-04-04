"""Guarded conversational agent with context validation.

`GuardedChatter` extends `PersonalityChatter` with a built-in
`ContextGuard` that validates every response before it enters
conversation history.  Rejected responses are retried with
feedback; if all retries fail the guard's suggested fix is
used instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from agents7.agents.chatter import (
    _ChatterStreamBase,
)
from agents7.agents.context_guard import (
    ContextGuard,
    GuardVerdict,
    _PLACEHOLDER,
)
from agents7.agents.personality import (
    Persona,
    PersonalityChatter,
)
from agents7.llm.streaming import (
    AsyncStreamResult,
    StreamResult,
)
from agents7.logs import logger
from agents7.models import ChatResponse, LLMMetrics


# --- Guarded stream wrappers ---


class GuardedChatterStreamResult(_ChatterStreamBase):
    """Sync streaming wrapper with post-stream guard.

    Yields text chunks normally.  After iteration the guard
    evaluates the full response.  If rejected, the
    ``chat_response.response`` contains the corrected text
    (which may differ from the streamed chunks).  History
    always stores the authoritative version.

    Usage::

        stream = agent.chat_stream("Hello!")
        for chunk in stream:
            print(chunk, end="", flush=True)
        # chat_response.response is the guarded version
        print(stream.chat_response)
    """

    __slots__ = ("_guard", "_user_msg")

    def __init__(
        self,
        stream: StreamResult,
        chatter: "GuardedChatter",
        user_message: str,
        guard: ContextGuard,
    ) -> None:
        super().__init__(stream, chatter, user_message)
        self._guard = guard
        self._user_msg = user_message

    def __iter__(self) -> Iterator[str]:
        for chunk in self._stream:
            yield chunk
        self._finalize()

    def _finalize(self) -> None:
        text = self._stream.response
        text = self._chatter._guard_and_fix(
            self._user_msg,
            text,
        )
        self._chat_response = ChatResponse(
            response=text,
            thinking=self._stream.thinking,
            metrics=LLMMetrics(**self._stream.metrics),
        )
        self._chatter._append_assistant(text)


class GuardedChatterAsyncStreamResult(
    _ChatterStreamBase,
):
    """Async streaming wrapper with post-stream guard.

    Same semantics as `GuardedChatterStreamResult` but for
    async iteration.
    """

    __slots__ = ("_guard", "_user_msg")

    def __init__(
        self,
        stream: AsyncStreamResult,
        chatter: "GuardedChatter",
        user_message: str,
        guard: ContextGuard,
    ) -> None:
        super().__init__(stream, chatter, user_message)
        self._guard = guard
        self._user_msg = user_message

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        async for chunk in self._stream:
            yield chunk
        await self._finalize_async()

    async def _finalize_async(self) -> None:
        text = self._stream.response
        text = await self._chatter._guard_and_fix_async(
            self._user_msg,
            text,
        )
        self._chat_response = ChatResponse(
            response=text,
            thinking=self._stream.thinking,
            metrics=LLMMetrics(**self._stream.metrics),
        )
        self._chatter._append_assistant(text)


# --- GuardedChatter ---


@dataclass(frozen=True)
class GuardedChatter(PersonalityChatter):
    """PersonalityChatter with a built-in context guard.

    Every response is validated by a `ContextGuard` before
    it enters conversation history.  When the guard rejects
    a response the chatter is asked to regenerate (with the
    guard's reason as feedback) up to ``max_guard_retries``
    times.  If all retries fail the guard's suggested
    ``fix`` is used.

    :param guard: The `ContextGuard` instance.
    :param max_guard_retries: How many times to retry on
        rejection.  Defaults to ``2``.
    :param guard_context_turns: Number of recent history
        turns sent to the guard for context.
        Defaults to ``3``.
    """

    guard: ContextGuard = field(kw_only=True)
    max_guard_retries: int = field(
        default=2,
        kw_only=True,
    )
    guard_context_turns: int = field(
        default=3,
        kw_only=True,
    )

    # --- Public API overrides ---

    def chat(self, message: str) -> ChatResponse:
        """Send a message and guard the response.

        :param message: The user's message.
        :return: The guarded ChatResponse.
        :raises ChatEndedError: If the conversation ended.
        """
        self._check_ended()
        self._append_user(message)
        messages = self._build_messages()

        text, response = self._generate(messages)
        text = self._guard_and_fix(message, text)
        text = self._maybe_typo(text)
        self._append_assistant(text)

        return ChatResponse(
            response=text,
            thinking=response.get("thinking"),
            metrics=LLMMetrics(**response["metrics"]),
        )

    async def chat_async(
        self,
        message: str,
    ) -> ChatResponse:
        """Async version of `chat`.

        :param message: The user's message.
        :return: The guarded ChatResponse.
        :raises ChatEndedError: If the conversation ended.
        """
        self._check_ended()
        self._append_user(message)
        messages = self._build_messages()

        text, response = await self._generate_async(
            messages,
        )
        text = await self._guard_and_fix_async(
            message,
            text,
        )
        text = self._maybe_typo(text)
        self._append_assistant(text)

        return ChatResponse(
            response=text,
            thinking=response.get("thinking"),
            metrics=LLMMetrics(**response["metrics"]),
        )

    def chat_stream(
        self,
        message: str,
    ) -> GuardedChatterStreamResult:
        """Stream a chat response with post-stream guard.

        Text chunks are yielded normally during iteration.
        After the stream completes the guard evaluates the
        full response.  If rejected,
        ``chat_response.response`` contains the corrected
        text (which may differ from the streamed chunks).

        :param message: The user's message.
        :return: A GuardedChatterStreamResult.
        :raises ChatEndedError: If the conversation ended.
        """
        self._check_ended()
        self._append_user(message)
        messages = self._build_messages()

        logger.debug(f"GuardedChatter `{self.name}`: streaming chat turn")
        stream = self.engine.chat_stream(
            messages=messages,
            temperature=self.temperature,
            think=self.think,
        )
        return GuardedChatterStreamResult(
            stream,
            self,
            message,
            self.guard,
        )

    async def chat_stream_async(
        self,
        message: str,
    ) -> GuardedChatterAsyncStreamResult:
        """Async version of `chat_stream`.

        :param message: The user's message.
        :return: A GuardedChatterAsyncStreamResult.
        :raises ChatEndedError: If the conversation ended.
        """
        self._check_ended()
        self._append_user(message)
        messages = self._build_messages()

        logger.debug(
            f"GuardedChatter `{self.name}`: async streaming chat turn"
        )
        stream = await self.engine.chat_stream_async(
            messages=messages,
            temperature=self.temperature,
            think=self.think,
        )
        return GuardedChatterAsyncStreamResult(
            stream,
            self,
            message,
            self.guard,
        )

    @classmethod
    def from_yaml(
        cls,
        file_path: str,
    ) -> GuardedChatter:
        """Create from a YAML configuration file.

        Extends the PersonalityChatter YAML schema with a
        ``guard`` section (nested ContextGuard config) and
        optional ``max_guard_retries`` /
        ``guard_context_turns`` fields::

            name: guarded-david
            version: 1
            engine:
                provider: ollama
                model: qwen3:4b
            persona:
                full_name: David Garcia
                ...
            guard:
                engine:
                    provider: ollama
                    model: qwen3:4b
            max_guard_retries: 2
            guard_context_turns: 3

        :param file_path: Path to the YAML file.
        :return: A GuardedChatter instance.
        """
        return _load_guarded_yaml(file_path)

    def describe(self) -> str:
        """Human-readable summary of the agent.

        :return: Multi-line description.
        """
        base = super().describe()
        lines = [
            base,
            f"Guard: {self.guard.name} (v{self.guard.version})",
            f"Guard retries: {self.max_guard_retries}",
            f"Guard context: {self.guard_context_turns} turns",
        ]
        return "\n".join(lines)

    # --- Private helpers ---

    def _generate(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[str, dict[str, Any]]:
        """Call the LLM and return (text, raw_response)."""
        logger.debug(f"GuardedChatter `{self.name}`: generating response")
        response = self.engine.chat(
            messages=messages,
            temperature=self.temperature,
            think=self.think,
        )
        response.pop("role", None)
        return response["response"], response

    async def _generate_async(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[str, dict[str, Any]]:
        """Async version of `_generate`."""
        logger.debug(
            f"GuardedChatter `{self.name}`: async generating response"
        )
        response = await self.engine.chat_async(
            messages=messages,
            temperature=self.temperature,
            think=self.think,
        )
        response.pop("role", None)
        return response["response"], response

    def _guard_check(
        self,
        user_message: str,
        assistant_response: str,
    ) -> GuardVerdict:
        """Run the guard on a response.

        :return: The parsed `GuardVerdict`.
        """
        guard_input = self._build_guard_input(
            user_message,
            assistant_response,
        )
        result = self.guard.act(guard_input)
        return result.model

    async def _guard_check_async(
        self,
        user_message: str,
        assistant_response: str,
    ) -> GuardVerdict:
        """Async version of `_guard_check`."""
        guard_input = self._build_guard_input(
            user_message,
            assistant_response,
        )
        result = await self.guard.act_async(guard_input)
        return result.model

    def _guard_and_fix(
        self,
        user_message: str,
        text: str,
    ) -> str:
        """Guard a response; retry or fix if rejected.

        :return: The accepted (or fixed) response text.
        """
        verdict = self._guard_check(user_message, text)
        if verdict.value:
            return text

        messages = self._build_messages()
        for i in range(self.max_guard_retries):
            logger.debug(
                f"GuardedChatter `{self.name}`: "
                f"guard retry {i + 1}/"
                f"{self.max_guard_retries} "
                f"(reason: {verdict.reason.name})"
            )
            messages = self._add_feedback(
                messages,
                text,
                verdict,
            )
            text, _ = self._generate(messages)
            verdict = self._guard_check(
                user_message,
                text,
            )
            if verdict.value:
                return text

        # All retries exhausted — use fix if available.
        if verdict.fix:
            logger.debug(f"GuardedChatter `{self.name}`: using guard fix")
            return verdict.fix
        return text

    async def _guard_and_fix_async(
        self,
        user_message: str,
        text: str,
    ) -> str:
        """Async version of `_guard_and_fix`."""
        verdict = await self._guard_check_async(
            user_message,
            text,
        )
        if verdict.value:
            return text

        messages = self._build_messages()
        for i in range(self.max_guard_retries):
            logger.debug(
                f"GuardedChatter `{self.name}`: "
                f"async guard retry {i + 1}/"
                f"{self.max_guard_retries} "
                f"(reason: {verdict.reason.name})"
            )
            messages = self._add_feedback(
                messages,
                text,
                verdict,
            )
            text, _ = await self._generate_async(
                messages,
            )
            verdict = await self._guard_check_async(
                user_message,
                text,
            )
            if verdict.value:
                return text

        if verdict.fix:
            logger.debug(
                f"GuardedChatter `{self.name}`: using guard fix (async)"
            )
            return verdict.fix
        return text

    def _build_guard_input(
        self,
        user_message: str,
        assistant_response: str,
    ) -> dict[str, Any]:
        """Build the input dict for the guard agent."""
        recent: list[dict[str, str]] = []
        if self.guard_context_turns > 0:
            max_msgs = self.guard_context_turns * 2
            recent = list(self._history[-max_msgs:])
        return {
            "user_message": user_message,
            "assistant_response": assistant_response,
            "recent_history": recent,
        }

    @staticmethod
    def _add_feedback(
        messages: list[dict[str, str]],
        rejected_text: str,
        verdict: GuardVerdict,
    ) -> list[dict[str, str]]:
        """Append feedback about a rejected response.

        Returns a **new** list — the original is not
        mutated.  The feedback is temporary and never
        stored in conversation history.
        """
        feedback = (
            f"[Your previous response was rejected by "
            f"the context guard. Reason: "
            f"{verdict.reason.value} "
            f"Please regenerate your response while "
            f"staying fully in character.]"
        )
        return [
            *messages,
            {"role": "assistant", "content": rejected_text},
            {"role": "user", "content": feedback},
        ]


# --- YAML loading ---


def _load_guarded_yaml(
    file_path: str,
) -> GuardedChatter:
    """Load a GuardedChatter from a YAML file."""
    from agents7.agents.config_loader import (
        _resolve_inheritance,
        build_engine,
        validate_chatter_yaml_config,
    )
    from agents7.agents.memory import (
        MemoryConfig,
        create_memory,
    )

    raw = _resolve_inheritance(file_path)
    raw.setdefault("system_prompt", "(auto-generated)")
    validate_chatter_yaml_config(raw, file_path)

    # --- Persona ---
    persona_data = raw.get("persona", {})
    persona = Persona.from_dict(dict(persona_data))

    extra_instructions = raw.get(
        "extra_instructions",
        "",
    )
    if extra_instructions:
        extra_instructions = str(extra_instructions).strip()

    # --- Guard ---
    guard_raw = raw.get("guard", {})
    guard_engine = build_engine(
        guard_raw.get("engine", raw["engine"]),
    )
    persona_summary = guard_raw.get(
        "persona_summary",
        _derive_persona_summary(persona),
    )
    guard = ContextGuard(
        name=f"{raw['name']}-guard",
        version=raw["version"],
        engine=guard_engine,
        system_prompt=_PLACEHOLDER,
        persona_summary=persona_summary,
    )

    # --- Main config ---
    config: dict[str, Any] = {
        "name": raw["name"],
        "version": raw["version"],
        "engine": build_engine(raw["engine"]),
        "system_prompt": "",
        "persona": persona,
        "extra_instructions": extra_instructions,
        "guard": guard,
    }

    if "temperature" in raw:
        config["temperature"] = float(raw["temperature"])
    if "think" in raw:
        config["think"] = bool(raw["think"])
    if "max_history" in raw:
        config["max_history"] = int(raw["max_history"])
    if "max_turns" in raw:
        config["max_turns"] = int(raw["max_turns"])
    if "typo_rate" in raw:
        config["typo_rate"] = float(raw["typo_rate"])
    if "conversation_id" in raw:
        config["conversation_id"] = str(raw["conversation_id"])
    if "max_guard_retries" in raw:
        config["max_guard_retries"] = int(raw["max_guard_retries"])
    if "guard_context_turns" in raw:
        config["guard_context_turns"] = int(raw["guard_context_turns"])
    if memory_settings := raw.get("memory"):
        memory_config = MemoryConfig(
            backend=memory_settings.get("backend", "json"),
            path=memory_settings.get("path", ""),
        )
        config["memory"] = create_memory(memory_config)

    return GuardedChatter(**config)


def _derive_persona_summary(persona: Persona) -> str:
    """Build a one-line persona summary for the guard."""
    parts = [persona.full_name]
    if persona.nationality:
        parts.append(persona.nationality)
    if persona.occupation:
        parts.append(persona.occupation)
    if persona.personality:
        parts.append(", ".join(persona.personality[:3]))
    return ", ".join(parts)
