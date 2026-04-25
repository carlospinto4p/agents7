"""Guarded conversational agent with context validation.

`GuardedChatter` extends `PersonalityChatter` with a built-in
`ContextGuard` that validates every response before it enters
conversation history.  Rejected responses are retried with
feedback; if all retries fail the guard's suggested fix is
used instead.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agents7.agents.chatter import (
    _ChatterStreamBase,
)
from agents7.agents.context_guard import (
    ContextGuard,
    GuardReason,
    GuardVerdict,
)
from agents7.agents.personality import (
    PersonalityChatter,
)
from agents7.logs import logger
from agents7.models import ChatResponse, LLMMetrics


@dataclass(frozen=True)
class GuardEvent:
    """Event emitted by `GuardedChatter` on each guard check.

    :param attempt: Attempt number (``0`` = initial check,
        ``1+`` = retry).
    :param accepted: Whether the guard accepted the response.
    :param reason: The guard's reason enum value, or
        ``None`` when the guard failed to parse.
    :param response_text: The response that was evaluated.
    :param fix: The guard's suggested rewrite (if rejected).
    :param used_fix: ``True`` when the final response came
        from the guard's ``fix`` field rather than the
        chatter.
    :param duration: Guard LLM call duration in seconds.
    :param parse_failed: ``True`` when the guard LLM
        returned unparseable output.
    """

    attempt: int
    accepted: bool
    reason: GuardReason | None
    response_text: str
    fix: str | None = None
    used_fix: bool = False
    duration: float = 0.0
    parse_failed: bool = False


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

    __slots__ = ()

    def _finalize(self) -> None:
        text = self._stream.response
        text = self._chatter._guard_and_fix(
            self._user_message,
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

    __slots__ = ()

    async def _finalize_async(self) -> None:
        text = self._stream.response
        text = await self._chatter._guard_and_fix_async(
            self._user_message,
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
    :param guard_callback: Optional callable invoked on
        every guard check with a `GuardEvent`.  Useful for
        logging or debugging the guard flow.
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
    guard_callback: Callable[[GuardEvent], None] | None = field(
        default=None, kw_only=True
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
        return GuardedChatterStreamResult(stream, self, message)

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
        return GuardedChatterAsyncStreamResult(stream, self, message)

    @classmethod
    def from_yaml(
        cls,
        file_path: str,
    ) -> "GuardedChatter":
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
        from agents7.agents.config_loader import (
            load_guarded_yaml_settings,
        )

        return load_guarded_yaml_settings(file_path)

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
    ) -> tuple[GuardVerdict | None, float]:
        """Run the guard on a response.

        :return: ``(verdict, duration)`` where verdict is
            ``None`` when the guard failed to parse.
        """
        guard_input = self._build_guard_input(
            user_message,
            assistant_response,
        )
        result = self.guard.act(guard_input)
        dur = result.metrics.total_duration
        if result.model is None:
            logger.warning(
                f"GuardedChatter `{self.name}`: "
                f"guard parse failed, passing through"
            )
        return result.model, dur

    async def _guard_check_async(
        self,
        user_message: str,
        assistant_response: str,
    ) -> tuple[GuardVerdict | None, float]:
        """Async version of `_guard_check`."""
        guard_input = self._build_guard_input(
            user_message,
            assistant_response,
        )
        result = await self.guard.act_async(guard_input)
        dur = result.metrics.total_duration
        if result.model is None:
            logger.warning(
                f"GuardedChatter `{self.name}`: "
                f"guard parse failed, passing through"
            )
        return result.model, dur

    def _emit(
        self,
        attempt: int,
        verdict: GuardVerdict | None,
        text: str,
        duration: float = 0.0,
        used_fix: bool = False,
    ) -> None:
        """Emit a `GuardEvent` via the callback."""
        if self.guard_callback is None:
            return
        if verdict is None:
            self.guard_callback(
                GuardEvent(
                    attempt=attempt,
                    accepted=True,
                    reason=None,
                    response_text=text,
                    duration=duration,
                    parse_failed=True,
                )
            )
            return
        self.guard_callback(
            GuardEvent(
                attempt=attempt,
                accepted=bool(verdict.value),
                reason=verdict.reason,
                response_text=text,
                fix=verdict.fix,
                used_fix=used_fix,
                duration=duration,
            )
        )

    @staticmethod
    def _is_accepted(
        verdict: GuardVerdict | None,
    ) -> bool:
        """Check if a verdict passed (or parse failed)."""
        return verdict is None or bool(verdict.value)

    def _try_accept(
        self,
        attempt: int,
        verdict: GuardVerdict | None,
        text: str,
        duration: float,
    ) -> str | None:
        """Emit a guard event and return *text* if accepted.

        :return: The accepted text, or ``None`` if rejected.
        """
        self._emit(attempt, verdict, text, duration=duration)
        if self._is_accepted(verdict):
            return text
        return None

    def _use_fix_or_text(
        self,
        verdict: GuardVerdict | None,
        text: str,
    ) -> str:
        """Return the guard's fix if available, else *text*.

        Called after all retries are exhausted.
        """
        if verdict is not None and verdict.fix:
            logger.debug(f"GuardedChatter `{self.name}`: using guard fix")
            self._emit(
                self.max_guard_retries + 1,
                verdict,
                verdict.fix,
                duration=0.0,
                used_fix=True,
            )
            return verdict.fix
        return text

    def _guard_and_fix(
        self,
        user_message: str,
        text: str,
    ) -> str:
        """Guard a response; retry or fix if rejected.

        When the guard fails to parse (verdict is ``None``),
        the response passes through (fail-open).

        :return: The accepted (or fixed) response text.
        """
        verdict, dur = self._guard_check(
            user_message,
            text,
        )
        result = self._try_accept(0, verdict, text, dur)
        if result is not None:
            return result

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
            verdict, dur = self._guard_check(
                user_message,
                text,
            )
            result = self._try_accept(
                i + 1,
                verdict,
                text,
                dur,
            )
            if result is not None:
                return result

        return self._use_fix_or_text(verdict, text)

    async def _guard_and_fix_async(
        self,
        user_message: str,
        text: str,
    ) -> str:
        """Async version of `_guard_and_fix`."""
        verdict, dur = await self._guard_check_async(
            user_message,
            text,
        )
        result = self._try_accept(0, verdict, text, dur)
        if result is not None:
            return result

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
            verdict, dur = await self._guard_check_async(
                user_message,
                text,
            )
            result = self._try_accept(
                i + 1,
                verdict,
                text,
                dur,
            )
            if result is not None:
                return result

        return self._use_fix_or_text(verdict, text)

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
            {
                "role": "assistant",
                "content": rejected_text,
            },
            {"role": "user", "content": feedback},
        ]
