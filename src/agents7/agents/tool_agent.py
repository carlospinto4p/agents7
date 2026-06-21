"""An agent that calls registered tools in a bounded loop."""

from dataclasses import dataclass, field
from typing import Any

from agents7.agents.base import Agent, BaseAgent
from agents7.exceptions import AgentConfigurationError, ToolLoopError
from agents7.models import AgentResponse
from agents7.models.base import LLMMetrics
from agents7.tools.registry import ToolRegistry


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
    """

    tools: list[Any] = field(default_factory=list, kw_only=True)
    max_iterations: int = field(default=5, kw_only=True)
    temperature: float = field(default=0.7, kw_only=True)

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
            self._accumulate(totals, result["metrics"])
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
            self._accumulate(totals, result["metrics"])
            calls = result.get("tool_calls") or []
            if not calls:
                return self._final(result, totals)
            messages += self._run_tools(result, calls)
        raise self._loop_error()

    def _seed_messages(self, user_input: str) -> list[dict]:
        return [
            {"role": "system", "content": self.resolved_system_prompt},
            {"role": "user", "content": user_input},
        ]

    def _run_tools(self, result: dict, calls: list[dict]) -> list[dict]:
        results = [
            self._registry.execute(call["name"], call["arguments"])
            for call in calls
        ]
        return self.engine.build_tool_turn_messages(
            result["response"], calls, results
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

    @staticmethod
    def _new_totals() -> dict:
        return {"tokens": 0, "prompt_tokens": 0, "total_duration": 0.0}

    @staticmethod
    def _accumulate(totals: dict, metrics: dict) -> None:
        totals["tokens"] += metrics.get("tokens") or 0
        totals["prompt_tokens"] += metrics.get("prompt_tokens") or 0
        totals["total_duration"] += metrics.get("total_duration") or 0.0
