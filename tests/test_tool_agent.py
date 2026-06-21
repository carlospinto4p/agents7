"""Unit tests for agents7.agents.tool_agent.ToolAgent."""

import pytest

from helpers import (
    chat_response_dict,
    make_engine,
    make_tool_engine,
    tool_call_response,
)

from agents7.agents.tool_agent import ToolAgent
from agents7.exceptions import AgentConfigurationError, ToolLoopError


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _agent(responses, **kwargs):
    return ToolAgent(
        name="calc",
        version=1,
        engine=make_tool_engine(responses),
        system_prompt="You can add.",
        tools=[add],
        **kwargs,
    )


def test_returns_final_answer_without_tools():
    agent = _agent([chat_response_dict("4")])
    out = agent.act("hi")
    assert out.response == "4"


def test_executes_tool_then_returns_answer():
    agent = _agent(
        [
            tool_call_response("add", {"a": 1, "b": 2}),
            chat_response_dict("The sum is 3"),
        ]
    )
    out = agent.act("add 1 and 2")
    assert out.response == "The sum is 3"


def test_metrics_accumulate_across_loop():
    agent = _agent(
        [
            tool_call_response("add", {"a": 1, "b": 2}),
            chat_response_dict("3"),
        ]
    )
    out = agent.act("add")
    # DUMMY_METRICS.tokens == 10 per call, two calls.
    assert out.metrics.tokens == 20
    assert out.metrics.prompt_tokens == 10


def test_max_iterations_raises():
    # Every turn asks for a tool again → never converges.
    responses = [
        tool_call_response("add", {"a": 1, "b": 1}) for _ in range(5)
    ]
    agent = _agent(responses, max_iterations=3)
    with pytest.raises(ToolLoopError, match="exceeded 3"):
        agent.act("loop forever")


def test_non_tool_provider_rejected():
    engine = make_engine()
    engine.llm_provider.supports_tools.return_value = False
    with pytest.raises(AgentConfigurationError, match="does not support"):
        ToolAgent(
            name="x",
            version=1,
            engine=engine,
            system_prompt="p",
            tools=[add],
        )


async def test_act_async_executes_tool():
    agent = _agent(
        [
            tool_call_response("add", {"a": 2, "b": 2}),
            chat_response_dict("4"),
        ]
    )
    out = await agent.act_async("add")
    assert out.response == "4"
