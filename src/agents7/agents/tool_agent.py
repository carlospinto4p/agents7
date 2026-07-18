"""An agent that calls registered tools in a bounded loop."""

from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from agents7.agents.base import Agent, BaseAgent
from agents7.call_log import CallRecord, generate_trace_id
from agents7.exceptions import AgentConfigurationError, ToolLoopError
from agents7.models import AgentResponse
from agents7.models.base import LLMMetrics
from agents7.tools.registry import ToolRegistry


class ToolStepKind(StrEnum):
    """The kind of a `ToolStepEvent`.

    A `StrEnum`, so members compare equal to their string
    value — ``event.kind == "tool_call"`` works as well as
    ``event.kind == ToolStepKind.TOOL_CALL``.
    """

    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ANSWER = "answer"


@dataclass(frozen=True)
class ToolStepEvent:
    """A single step emitted while a `ToolAgent` runs its loop.

    Yielded by `ToolAgent.act_stream` / `act_stream_async` so a
    caller can observe intermediate tool calls and their results
    before the final answer arrives.

    :param kind: The step kind — `ToolStepKind.TOOL_CALL` when the
        model requests a tool, `TOOL_RESULT` after the tool runs,
        or `ANSWER` for the final text reply.
    :param iteration: Zero-based loop iteration the event belongs
        to.
    :param name: The tool name (``TOOL_CALL`` and ``TOOL_RESULT``).
    :param arguments: The tool arguments (``TOOL_CALL`` only).
    :param result: The stringified tool result (``TOOL_RESULT``
        only).
    :param text: The final answer text (``ANSWER`` only).
    :param thinking: The model's thinking tokens, if any
        (``ANSWER`` only).
    """

    kind: ToolStepKind
    iteration: int
    name: str | None = None
    arguments: dict | None = None
    result: str | None = None
    text: str | None = None
    thinking: str | None = None


class _ToolStreamBase:
    """Shared state for the `ToolAgent` stream wrappers."""

    __slots__ = ("_agent", "_user_input", "_agent_response")

    def __init__(self, agent: "ToolAgent", user_input: str) -> None:
        self._agent = agent
        self._user_input = user_input
        self._agent_response: AgentResponse | None = None

    @property
    def agent_response(self) -> AgentResponse | None:
        """The final `AgentResponse`, available after iteration.

        ``None`` until iteration reaches the ``ANSWER`` event.
        """
        return self._agent_response


class ToolStreamResult(_ToolStreamBase):
    """Sync iterable of `ToolStepEvent` for a tool loop.

    Iterating runs the agent's tool-calling loop, yielding a
    `ToolStepEvent` for each tool call, each tool result, and the
    final answer. After iteration, the parsed result is available
    via `agent_response`.

    Usage::

        stream = agent.act_stream("What is 2 + 2?")
        for event in stream:
            if event.kind == "tool_call":
                print(event.name, event.arguments)
            elif event.kind == "tool_result":
                print(event.result)
            elif event.kind == "answer":
                print(event.text)
        print(stream.agent_response.metrics)

    :raises ToolLoopError: If the loop exceeds the agent's
        ``max_iterations`` before a final answer.
    """

    __slots__ = ()

    def __iter__(self) -> Iterator[ToolStepEvent]:
        return self._iterate()

    def _iterate(self) -> Iterator[ToolStepEvent]:
        agent = self._agent
        messages = agent._seed_messages(self._user_input)
        totals = agent._new_totals()
        for iteration in range(agent.max_iterations):
            result = agent.engine.chat(
                messages,
                temperature=agent.temperature,
                tools=agent._registry.schemas(),
            )
            agent._record_turn(totals, result["metrics"])
            calls = result.get("tool_calls") or []
            if not calls:
                self._agent_response = agent._final(result, totals)
                yield agent._answer_event(self._agent_response, iteration)
                return
            yield from agent._call_events(calls, iteration)
            results = agent._execute_calls(calls)
            yield from agent._result_events(calls, results, iteration)
            messages += agent.engine.build_tool_turn_messages(
                result["response"], calls, results
            )
        raise agent._loop_error()


class ToolAsyncStreamResult(_ToolStreamBase):
    """Async iterable of `ToolStepEvent` for a tool loop.

    Same semantics as `ToolStreamResult` but for async iteration.

    Usage::

        stream = await agent.act_stream_async("What is 2 + 2?")
        async for event in stream:
            ...
        print(stream.agent_response.metrics)
    """

    __slots__ = ()

    def __aiter__(self) -> AsyncIterator[ToolStepEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[ToolStepEvent]:
        agent = self._agent
        messages = agent._seed_messages(self._user_input)
        totals = agent._new_totals()
        for iteration in range(agent.max_iterations):
            result = await agent.engine.chat_async(
                messages,
                temperature=agent.temperature,
                tools=agent._registry.schemas(),
            )
            agent._record_turn(totals, result["metrics"])
            calls = result.get("tool_calls") or []
            if not calls:
                self._agent_response = agent._final(result, totals)
                yield agent._answer_event(self._agent_response, iteration)
                return
            for event in agent._call_events(calls, iteration):
                yield event
            results = agent._execute_calls(calls)
            for event in agent._result_events(calls, results, iteration):
                yield event
            messages += agent.engine.build_tool_turn_messages(
                result["response"], calls, results
            )
        raise agent._loop_error()


@dataclass(frozen=True)
class ToolAgent(BaseAgent):
    """An agent that can call Python functions to answer.

    Runs a bounded call -> execute-tool -> call loop: the model
    may request tool calls, which are executed and fed back, until
    it returns a final answer or ``max_iterations`` is hit.

    :param tools: Callables or `Tool` instances the agent may call.
    :param max_iterations: Maximum model turns before raising
        `ToolLoopError`.
    :param temperature: Sampling temperature for each model turn.
    :param call_logger: Optional callable receiving a `CallRecord`
        per **model turn** — not per `act`, since one answer may
        take several turns and the log measures real calls.
    """

    tools: list[Any] = field(default_factory=list, kw_only=True)
    max_iterations: int = field(default=5, kw_only=True)
    temperature: float = field(default=0.7, kw_only=True)
    call_logger: Callable | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.engine.supports_tools():
            raise AgentConfigurationError(
                f"Provider `{self.engine.provider}` does not support tools."
            )
        object.__setattr__(self, "_registry", ToolRegistry(self.tools))

    @classmethod
    def from_yaml(cls, file_path: str) -> Agent:
        """Build a `ToolAgent` from a YAML config file.

        :param file_path: Path to the YAML configuration.
        :return: A configured `ToolAgent`.
        """
        from agents7.agents.config_loader import (
            load_tool_agent_yaml_settings,
        )

        return cls(**load_tool_agent_yaml_settings(file_path))

    def act(self, user_input: str) -> AgentResponse:
        """Answer *user_input*, calling tools as needed.

        :param user_input: The user's message.
        :return: An `AgentResponse` with the final text and metrics
            summed across every model turn.
        :raises ToolLoopError: If the loop exceeds
            ``max_iterations``.
        """
        messages = self._seed_messages(user_input)
        totals = self._new_totals()
        for _ in range(self.max_iterations):
            result = self.engine.chat(
                messages,
                temperature=self.temperature,
                tools=self._registry.schemas(),
            )
            self._record_turn(totals, result["metrics"])
            calls = result.get("tool_calls") or []
            if not calls:
                return self._final(result, totals)
            messages += self._run_tools(result, calls)
        raise self._loop_error()

    async def act_async(self, user_input: str) -> AgentResponse:
        """Async version of `act`."""
        messages = self._seed_messages(user_input)
        totals = self._new_totals()
        for _ in range(self.max_iterations):
            result = await self.engine.chat_async(
                messages,
                temperature=self.temperature,
                tools=self._registry.schemas(),
            )
            self._record_turn(totals, result["metrics"])
            calls = result.get("tool_calls") or []
            if not calls:
                return self._final(result, totals)
            messages += self._run_tools(result, calls)
        raise self._loop_error()

    def act_stream(self, user_input: str) -> ToolStreamResult:
        """Answer *user_input*, streaming each loop step.

        Returns a `ToolStreamResult` that yields a `ToolStepEvent`
        for every tool call, every tool result, and the final
        answer as the loop runs — so a caller can observe
        intermediate steps rather than waiting for the whole loop.
        After iteration, the final `AgentResponse` is available via
        ``stream.agent_response``.

        The final answer is delivered as a single ``ANSWER`` event
        (whole text), not token-by-token: streaming raw tokens
        through a tool loop needs provider-specific streaming tool
        APIs, which this method deliberately avoids.

        :param user_input: The user's message.
        :return: A `ToolStreamResult` yielding `ToolStepEvent`s.
        :raises ToolLoopError: If the loop exceeds
            ``max_iterations``.
        """
        return ToolStreamResult(self, user_input)

    async def act_stream_async(
        self, user_input: str
    ) -> ToolAsyncStreamResult:
        """Async version of `act_stream`.

        :param user_input: The user's message.
        :return: A `ToolAsyncStreamResult` yielding
            `ToolStepEvent`s when async-iterated.
        """
        return ToolAsyncStreamResult(self, user_input)

    def _seed_messages(self, user_input: str) -> list[dict]:
        return [
            {"role": "system", "content": self.resolved_system_prompt},
            {"role": "user", "content": user_input},
        ]

    def _execute_calls(self, calls: list[dict]) -> list[str]:
        return [
            self._registry.execute(call["name"], call["arguments"])
            for call in calls
        ]

    def _run_tools(self, result: dict, calls: list[dict]) -> list[dict]:
        results = self._execute_calls(calls)
        return self.engine.build_tool_turn_messages(
            result["response"], calls, results
        )

    @staticmethod
    def _call_events(
        calls: list[dict], iteration: int
    ) -> list[ToolStepEvent]:
        return [
            ToolStepEvent(
                kind=ToolStepKind.TOOL_CALL,
                iteration=iteration,
                name=call["name"],
                arguments=call.get("arguments"),
            )
            for call in calls
        ]

    @staticmethod
    def _result_events(
        calls: list[dict], results: list[str], iteration: int
    ) -> list[ToolStepEvent]:
        return [
            ToolStepEvent(
                kind=ToolStepKind.TOOL_RESULT,
                iteration=iteration,
                name=call["name"],
                result=result,
            )
            for call, result in zip(calls, results)
        ]

    @staticmethod
    def _answer_event(
        response: AgentResponse, iteration: int
    ) -> ToolStepEvent:
        return ToolStepEvent(
            kind=ToolStepKind.ANSWER,
            iteration=iteration,
            text=response.response,
            thinking=response.thinking,
        )

    def _final(self, result: dict, totals: dict) -> AgentResponse:
        return AgentResponse(
            response=result["response"],
            model=None,
            thinking=result.get("thinking"),
            metrics=LLMMetrics(**totals),
        )

    def _loop_error(self) -> ToolLoopError:
        return ToolLoopError(
            f"ToolAgent `{self.name}` exceeded "
            f"{self.max_iterations} iterations."
        )

    def _record_turn(self, totals: dict, metrics: dict) -> None:
        """Fold one model turn into *totals* and log it.

        The single choke point every loop (sync, async, and both
        streams) routes each model turn through, so accumulation
        and call logging can never diverge between them.

        :param totals: The running metric totals to fold into.
        :param metrics: The turn's raw metrics from the engine.
        """
        self._accumulate(totals, metrics)
        self._emit_call_log(metrics)

    def _emit_call_log(self, metrics: dict) -> None:
        """Emit a `CallRecord` for one model turn, if logging.

        Only turns that returned are recorded — when the engine
        raises, the exception propagates to the caller rather than
        becoming a logged failure.

        :param metrics: The turn's raw metrics from the engine.
        """
        if self.call_logger is None:
            return
        self.call_logger(
            CallRecord(
                trace_id=generate_trace_id(),
                agent_name=self.name,
                agent_version=self.version,
                model_id=self.engine.model_id,
                success=True,
                tokens=metrics.get("tokens") or 0,
                prompt_tokens=metrics.get("prompt_tokens") or 0,
                total_duration=metrics.get("total_duration") or 0.0,
            )
        )

    @staticmethod
    def _new_totals() -> dict:
        return {"tokens": 0, "prompt_tokens": 0, "total_duration": 0.0}

    @staticmethod
    def _accumulate(totals: dict, metrics: dict) -> None:
        totals["tokens"] += metrics.get("tokens") or 0
        totals["prompt_tokens"] += metrics.get("prompt_tokens") or 0
        totals["total_duration"] += metrics.get("total_duration") or 0.0
        cost = metrics.get("cost_cents")
        if cost is not None:
            totals["cost_cents"] = (totals.get("cost_cents") or 0.0) + cost
