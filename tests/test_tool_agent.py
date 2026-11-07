"""Unit tests for agents7.agents.tool_agent.ToolAgent."""

import pytest

from helpers import (
    chat_response_dict,
    make_engine,
    make_tool_engine,
    tool_call_response,
)

from agents7.agents.tool_agent import (
    ToolAgent,
    ToolStepEvent,
    ToolStepKind,
)
from agents7.exceptions import (
    AgentConfigurationError,
    ToolLoopError,
    ToolSchemaError,
)


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
    # Two turns: tokens 10+10=20; prompt_tokens 5+5=10.
    assert out.metrics.tokens == 20
    assert out.metrics.prompt_tokens == 10
    assert out.metrics.cost_cents is None


def test_cost_cents_accumulates_across_turns():
    """cost_cents sums across turns when present in metrics."""
    turn1 = {
        "role": "assistant",
        "response": "",
        "thinking": None,
        "tool_calls": [{"name": "add", "arguments": {"a": 1, "b": 2}}],
        "metrics": {
            "tokens": 10,
            "prompt_tokens": 5,
            "total_duration": 0.1,
            "cost_cents": 0.2,
        },
    }
    turn2 = {
        "role": "assistant",
        "response": "3",
        "thinking": None,
        "metrics": {
            "tokens": 10,
            "prompt_tokens": 5,
            "total_duration": 0.1,
            "cost_cents": 0.2,
        },
    }
    agent = _agent([turn1, turn2])
    out = agent.act("add 1 and 2")
    assert abs(out.metrics.cost_cents - 0.4) < 1e-9


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


def test_unregistered_tool_raises_schema_error():
    """Requesting an unregistered tool name is a hard error."""
    agent = _agent([tool_call_response("nope", {})])
    with pytest.raises(ToolSchemaError):
        agent.act("use nope")


# ── Streaming Tests ────────────────────────────────────────


def test_stream_answer_only_yields_single_answer_event():
    agent = _agent([chat_response_dict("4")])
    stream = agent.act_stream("hi")
    events = list(stream)
    assert len(events) == 1
    assert events[0].kind == ToolStepKind.ANSWER
    assert events[0].kind == "answer"  # StrEnum compares to str
    assert events[0].text == "4"
    assert events[0].iteration == 0


def test_stream_emits_call_result_then_answer():
    agent = _agent(
        [
            tool_call_response("add", {"a": 1, "b": 2}),
            chat_response_dict("The sum is 3"),
        ]
    )
    events = list(agent.act_stream("add 1 and 2"))
    kinds = [e.kind for e in events]
    assert kinds == [
        ToolStepKind.TOOL_CALL,
        ToolStepKind.TOOL_RESULT,
        ToolStepKind.ANSWER,
    ]
    call, result, answer = events
    assert call.name == "add"
    assert call.arguments == {"a": 1, "b": 2}
    assert call.iteration == 0
    assert result.name == "add"
    assert result.result == "3"
    assert result.error is None
    assert result.iteration == 0
    assert answer.text == "The sum is 3"
    assert answer.iteration == 1


def test_stream_result_event_carries_the_failure():
    """A rejected call is flagged, not just described in text.

    `result` holds the `"Error: ..."` string fed back to the
    model; `error` is what says the call failed, since a tool may
    legitimately return text starting with `Error:`.
    """
    agent = _agent(
        [
            tool_call_response("add", {"a": {"type": "string"}, "b": 1}),
            chat_response_dict("I could not work that out."),
        ]
    )
    _, result, _ = list(agent.act_stream("add"))
    assert result.result.startswith("Error: Invalid arguments")
    assert "Invalid arguments" in result.error


def test_stream_agent_response_available_after_iteration():
    agent = _agent(
        [
            tool_call_response("add", {"a": 1, "b": 2}),
            chat_response_dict("3"),
        ]
    )
    stream = agent.act_stream("add")
    assert stream.agent_response is None
    list(stream)
    # Metrics accumulate across both turns (10+10 tokens).
    assert stream.agent_response.response == "3"
    assert stream.agent_response.metrics.tokens == 20
    assert stream.agent_response.model is None


def test_stream_max_iterations_raises():
    responses = [
        tool_call_response("add", {"a": 1, "b": 1}) for _ in range(5)
    ]
    agent = _agent(responses, max_iterations=2)
    with pytest.raises(ToolLoopError, match="exceeded 2"):
        list(agent.act_stream("loop forever"))


async def test_act_stream_async_emits_events():
    agent = _agent(
        [
            tool_call_response("add", {"a": 2, "b": 2}),
            chat_response_dict("4"),
        ]
    )
    stream = await agent.act_stream_async("add")
    events = [event async for event in stream]
    kinds = [e.kind for e in events]
    assert kinds == [
        ToolStepKind.TOOL_CALL,
        ToolStepKind.TOOL_RESULT,
        ToolStepKind.ANSWER,
    ]
    assert events[-1].text == "4"
    assert stream.agent_response.response == "4"


def test_tool_step_event_is_frozen():
    event = ToolStepEvent(kind=ToolStepKind.ANSWER, iteration=0, text="x")
    with pytest.raises(Exception):
        event.text = "y"
