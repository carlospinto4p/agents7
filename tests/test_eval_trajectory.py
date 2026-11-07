"""Tests for trajectory scoring of tool-calling agents.

Hermetic: the scorer is pure, and the runner is driven by a
scripted provider replaying canned turns, so nothing here
contacts a model.
"""

import pytest
from agents7.agents.tool_agent import ToolAgent
from agents7.eval.trajectory import (
    TRAJECTORY_FIELD,
    ExpectedCall,
    ToolCall,
    TrajectoryMode,
    collect_trajectory,
    compute_trajectory_metrics,
    match_trajectory,
    parse_expected_calls,
    run_trajectory_evaluation,
)
from agents7.exceptions import ToolLoopError
from agents7.llm import LLMEngine
from agents7.llm.base import LLMProvider

DUMMY_METRICS = {"tokens": 10, "prompt_tokens": 5, "total_duration": 0.1}


# ── Scripted provider ───────────────────────────────────────


class ScriptedProvider(LLMProvider):
    """A tool-capable provider replaying pre-built turns."""

    def __init__(self, responses: list[dict]):
        super().__init__("ollama", "scripted")
        self._responses = list(responses)

    def supports_tools(self) -> bool:
        return True

    def build_tool_turn_messages(self, assistant_text, tool_calls, results):
        return [{"role": "assistant", "content": assistant_text}]

    def chat(self, messages, tools=None, **kwargs):
        return self._responses.pop(0)

    async def chat_async(self, messages, tools=None, **kwargs):
        return self.chat(messages, tools=tools, **kwargs)

    def chat_stream(self, *args, **kwargs):
        raise NotImplementedError

    async def chat_stream_async(self, *args, **kwargs):
        raise NotImplementedError


def tool_turn(calls: list[tuple[str, dict]]) -> dict:
    """Build a response dict requesting tool calls."""
    return {
        "role": "assistant",
        "response": "",
        "thinking": None,
        "tool_calls": [
            {"name": name, "arguments": args} for name, args in calls
        ],
        "metrics": DUMMY_METRICS,
    }


def answer_turn(text: str) -> dict:
    """Build a response dict carrying a final answer."""
    return {
        "role": "assistant",
        "response": text,
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }


def echo(value: str) -> str:
    """Echo a value back.

    :param value: Anything.
    :return: The same text.
    """
    return value


def add(a: int, b: int) -> int:
    """Add two numbers.

    :param a: First number.
    :param b: Second number.
    :return: Their sum.
    """
    return a + b


@pytest.fixture
def scripted_agent(monkeypatch):
    """Return a factory building a tool agent on a script."""

    def build(responses: list[dict]) -> ToolAgent:
        provider = ScriptedProvider(responses)
        engine = LLMEngine(provider="ollama", model_name="scripted")
        monkeypatch.setattr(engine, "llm_provider", provider)
        return ToolAgent(
            name="Scripted",
            version=1,
            engine=engine,
            system_prompt="You help.",
            tools=[echo, add],
            max_iterations=4,
        )

    return build


# ── parse_expected_calls ────────────────────────────────────


def test_parse_accepts_dicts():
    """A list of dicts becomes expectations with arguments."""
    calls = parse_expected_calls(
        [{"name": "add", "arguments": {"a": 1}}, {"name": "echo"}]
    )
    assert [c.name for c in calls] == ["add", "echo"]
    assert calls[0].arguments == {"a": 1}
    assert calls[1].arguments is None


def test_parse_accepts_defaults():
    """A `defaults` key parses alongside `arguments`."""
    calls = parse_expected_calls(
        [
            {
                "name": "add",
                "arguments": {"a": 1},
                "defaults": {"b": 0},
            }
        ]
    )
    assert calls[0].arguments == {"a": 1}
    assert calls[0].defaults == {"b": 0}


def test_parse_leaves_defaults_unset_when_absent():
    """A call with no `defaults` key pins none."""
    calls = parse_expected_calls([{"name": "add"}])
    assert calls[0].defaults is None


def test_parse_accepts_bare_tool_names():
    """A list of strings names tools without pinning arguments."""
    calls = parse_expected_calls(["echo", "add"])
    assert [c.name for c in calls] == ["echo", "add"]
    assert all(c.arguments is None for c in calls)


def test_parse_accepts_a_json_string():
    """A CSV cell arrives as JSON text and still parses."""
    calls = parse_expected_calls('[{"name": "echo"}]')
    assert calls[0].name == "echo"


def test_parse_rejects_invalid_json():
    """A malformed JSON cell raises, naming the column."""
    with pytest.raises(ValueError, match=TRAJECTORY_FIELD):
        parse_expected_calls("not json")


def test_parse_rejects_a_non_list():
    """A scalar is not a trajectory."""
    with pytest.raises(ValueError, match="must be a list"):
        parse_expected_calls({"name": "echo"})


def test_parse_rejects_a_call_without_a_name():
    """Every expected call must name a tool."""
    with pytest.raises(ValueError, match="`name`"):
        parse_expected_calls([{"arguments": {"a": 1}}])


# ── Matching: order ─────────────────────────────────────────


def test_matches_an_exact_sequence():
    """The same calls in the same order pass."""
    result = match_trajectory(
        [ExpectedCall("echo"), ExpectedCall("add")],
        [ToolCall("echo"), ToolCall("add")],
    )
    assert result.passed
    assert result.missing == []
    assert result.extra == []


def test_wrong_order_fails():
    """Expected calls must appear in the expected order."""
    result = match_trajectory(
        [ExpectedCall("echo"), ExpectedCall("add")],
        [ToolCall("add"), ToolCall("echo")],
    )
    assert not result.passed
    assert [e.name for e in result.missing] == ["add"]


def test_missing_call_fails():
    """An expected call that never happened fails the row."""
    result = match_trajectory(
        [ExpectedCall("echo"), ExpectedCall("add")],
        [ToolCall("echo")],
    )
    assert not result.passed
    assert [e.name for e in result.missing] == ["add"]


def test_no_calls_at_all_fails():
    """An agent that called nothing fails a row expecting calls."""
    result = match_trajectory([ExpectedCall("echo")], [])
    assert not result.passed


def test_empty_expectation_passes_with_no_calls():
    """A row expecting no tools passes when none were called."""
    assert match_trajectory([], []).passed


def test_empty_expectation_fails_a_call_under_in_order():
    """A row expecting no tools fails when the agent called one.

    `in_order` tolerates extras, so with nothing to miss this row
    used to pass unconditionally — the agent could reach for any
    tool and still score a pass.
    """
    result = match_trajectory([], [ToolCall("echo")])
    assert not result.passed
    assert [c.name for c in result.extra] == ["echo"]


def test_empty_expectation_fails_a_call_under_exact():
    """The same row fails under exact, as it always did."""
    result = match_trajectory([], [ToolCall("echo")], TrajectoryMode.EXACT)
    assert not result.passed


# ── Matching: modes ─────────────────────────────────────────


def test_in_order_tolerates_extra_calls():
    """Extra calls are allowed but reported under in_order."""
    result = match_trajectory(
        [ExpectedCall("echo"), ExpectedCall("add")],
        [ToolCall("echo"), ToolCall("other"), ToolCall("add")],
    )
    assert result.passed
    assert [c.name for c in result.extra] == ["other"]


def test_exact_rejects_extra_calls():
    """The same trajectory fails under exact."""
    result = match_trajectory(
        [ExpectedCall("echo"), ExpectedCall("add")],
        [ToolCall("echo"), ToolCall("other"), ToolCall("add")],
        TrajectoryMode.EXACT,
    )
    assert not result.passed
    assert result.missing == []


def test_exact_passes_a_clean_trajectory():
    """Exact still passes when nothing extra was called."""
    result = match_trajectory(
        [ExpectedCall("echo")],
        [ToolCall("echo")],
        TrajectoryMode.EXACT,
    )
    assert result.passed


# ── Matching: arguments ─────────────────────────────────────


def test_arguments_match_on_stated_keys_only():
    """Unstated arguments are ignored."""
    result = match_trajectory(
        [ExpectedCall("add", {"a": 1})],
        [ToolCall("add", {"a": 1, "b": 2})],
    )
    assert result.passed


def test_a_wrong_stated_argument_fails():
    """A pinned argument that differs fails the call."""
    result = match_trajectory(
        [ExpectedCall("add", {"a": 1})],
        [ToolCall("add", {"a": 9})],
    )
    assert not result.passed


def test_a_missing_stated_argument_fails():
    """A pinned argument the agent never sent fails the call."""
    result = match_trajectory(
        [ExpectedCall("add", {"a": 1})],
        [ToolCall("add", {"b": 2})],
    )
    assert not result.passed


def test_arguments_compare_case_insensitively_across_types():
    """A CSV-loaded string scores like a typed JSON value."""
    result = match_trajectory(
        [ExpectedCall("add", {"a": "1"})],
        [ToolCall("add", {"a": 1})],
    )
    assert result.passed


def test_omitting_arguments_scores_the_name_alone():
    """No pinned arguments means any arguments match."""
    result = match_trajectory(
        [ExpectedCall("add")],
        [ToolCall("add", {"a": 1, "b": 2})],
    )
    assert result.passed


def test_the_right_tool_with_wrong_args_can_match_a_later_call():
    """Matching skips a same-named call whose arguments differ."""
    result = match_trajectory(
        [ExpectedCall("add", {"a": 2})],
        [ToolCall("add", {"a": 1}), ToolCall("add", {"a": 2})],
    )
    assert result.passed
    assert [c.arguments for c in result.extra] == [{"a": 1}]


# ── Matching: defaults ──────────────────────────────────────


def test_a_default_passes_when_the_argument_is_absent():
    """Omitting an optional argument is correct behaviour."""
    result = match_trajectory(
        [ExpectedCall("add", {"a": 1}, defaults={"b": 0})],
        [ToolCall("add", {"a": 1})],
    )
    assert result.passed


def test_a_default_passes_when_the_argument_equals_it():
    """Passing the default explicitly is also correct."""
    result = match_trajectory(
        [ExpectedCall("add", {"a": 1}, defaults={"b": 0})],
        [ToolCall("add", {"a": 1, "b": 0})],
    )
    assert result.passed


def test_a_default_fails_when_the_argument_differs():
    """An invented optional argument no longer matches.

    Left out of `arguments` entirely it would be ignored, and
    pinned under `arguments` it would fail a model that correctly
    omitted it — `defaults` is what rejects only the invention.
    """
    result = match_trajectory(
        [ExpectedCall("add", {"a": 1}, defaults={"b": 0})],
        [ToolCall("add", {"a": 1, "b": 9})],
    )
    assert not result.passed


def test_a_default_compares_case_insensitively_across_types():
    """A CSV-loaded `"false"` scores like a JSON `false`."""
    result = match_trajectory(
        [ExpectedCall("add", defaults={"b": "false"})],
        [ToolCall("add", {"b": False})],
    )
    assert result.passed


def test_defaults_alone_still_check_the_name():
    """A call with only defaults pinned still matches by name."""
    result = match_trajectory(
        [ExpectedCall("add", defaults={"b": 0})],
        [ToolCall("echo", {})],
    )
    assert not result.passed


def test_format_marks_a_default_with_a_question_mark():
    """`k?=v` reads as "absent, or this" in a report line."""
    expected = ExpectedCall("add", {"a": 1}, defaults={"b": 0})
    assert expected.format() == "add(a=1, b?=0)"


def test_format_falls_back_to_the_name_with_nothing_pinned():
    """Neither arguments nor defaults renders as `name(…)`."""
    assert ExpectedCall("add").format() == "add(…)"


# ── Matching: rejected calls ────────────────────────────────


def test_a_rejected_call_satisfies_nothing():
    """A call the tool threw out does not count as made.

    The scorer used to check which tools were called, not whether
    the call worked, so a malformed call satisfied an expectation
    pinned by name alone.
    """
    result = match_trajectory(
        [ExpectedCall("add")],
        [ToolCall("add", {"a": 1}, error="Invalid arguments")],
    )
    assert not result.passed
    assert [e.name for e in result.missing] == ["add"]


def test_a_rejected_call_is_reported_as_extra():
    """It still shows in the trajectory, marked as failed."""
    result = match_trajectory(
        [ExpectedCall("add")],
        [ToolCall("add", {"a": 1}, error="Invalid arguments")],
    )
    assert [c.name for c in result.extra] == ["add"]
    assert result.extra[0].failed


def test_a_later_successful_call_still_matches():
    """Retrying the tool correctly satisfies the expectation."""
    result = match_trajectory(
        [ExpectedCall("add")],
        [
            ToolCall("add", {"a": "x"}, error="Invalid arguments"),
            ToolCall("add", {"a": 1}, result="3"),
        ],
    )
    assert result.passed


def test_a_call_with_no_recorded_outcome_still_matches():
    """A hand-built call carries no outcome and is not failed."""
    assert match_trajectory([ExpectedCall("add")], [ToolCall("add")]).passed


def test_allow_failure_accepts_a_rejected_call():
    """A row testing the error path passes on the rejection.

    `check_stock` on an unknown SKU is meant to raise, and the
    agent is meant to relay it — the call counts as made.
    """
    result = match_trajectory(
        [ExpectedCall("check_stock", {"sku": "NOPE"}, allow_failure=True)],
        [ToolCall("check_stock", {"sku": "NOPE"}, error="Unknown SKU")],
    )
    assert result.passed
    assert result.extra == []


def test_allow_failure_still_accepts_a_call_that_returned():
    """It tolerates a failure, it does not require one."""
    result = match_trajectory(
        [ExpectedCall("check_stock", allow_failure=True)],
        [ToolCall("check_stock", result="42 units")],
    )
    assert result.passed


def test_allow_failure_still_checks_the_arguments():
    """Tolerating a failure does not loosen anything else."""
    result = match_trajectory(
        [ExpectedCall("check_stock", {"sku": "NOPE"}, allow_failure=True)],
        [ToolCall("check_stock", {"sku": "OTHER"}, error="Unknown SKU")],
    )
    assert not result.passed


def test_parse_reads_allow_failure():
    """A row opts in with an `allow_failure` key."""
    calls = parse_expected_calls(
        [{"name": "check_stock", "allow_failure": True}, {"name": "add"}]
    )
    assert calls[0].allow_failure
    assert not calls[1].allow_failure


def test_format_marks_an_expectation_that_may_fail():
    """`[may fail]` says the row is testing the error path."""
    expected = ExpectedCall("check_stock", allow_failure=True)
    assert expected.format() == "check_stock(…) [may fail]"


def test_format_marks_a_rejected_call():
    """`[failed]` says the tool was reached for and never ran."""
    call = ToolCall("add", {"a": 1}, error="Invalid arguments")
    assert call.format() == "add(a=1) [failed]"


def test_result_format_marks_a_rejected_call():
    """The one-line summary names the call that did not run."""
    result = match_trajectory(
        [ExpectedCall("add")],
        [ToolCall("add", {"a": 1}, error="Invalid arguments")],
    )
    assert "add (failed)" in result.format()


# ── collect_trajectory ──────────────────────────────────────


def test_collect_records_calls_in_order(scripted_agent):
    """The recorded trajectory is what the agent really did."""
    agent = scripted_agent(
        [
            tool_turn([("echo", {"value": "hi"})]),
            tool_turn([("add", {"a": 1, "b": 2})]),
            answer_turn("Done."),
        ]
    )
    calls, response = collect_trajectory(agent, "go")
    assert [c.name for c in calls] == ["echo", "add"]
    assert calls[1].arguments == {"a": 1, "b": 2}
    assert response.response == "Done."


def test_collect_records_parallel_calls_in_one_turn(scripted_agent):
    """Two calls in a single turn are both recorded."""
    agent = scripted_agent(
        [
            tool_turn([("echo", {"value": "a"}), ("echo", {"value": "b"})]),
            answer_turn("Done."),
        ]
    )
    calls, _ = collect_trajectory(agent, "go")
    assert [c.arguments["value"] for c in calls] == ["a", "b"]


def test_collect_on_a_toolless_answer_is_empty(scripted_agent):
    """An answer needing no tool records an empty trajectory."""
    agent = scripted_agent([answer_turn("No tools needed.")])
    calls, response = collect_trajectory(agent, "go")
    assert calls == []
    assert response.response == "No tools needed."


def test_collect_records_what_each_tool_returned(scripted_agent):
    """The outcome travels with the call, not just its name."""
    agent = scripted_agent(
        [tool_turn([("add", {"a": 1, "b": 2})]), answer_turn("Done.")]
    )
    calls, _ = collect_trajectory(agent, "go")
    assert calls[0].result == "3"
    assert calls[0].error is None
    assert not calls[0].failed


def test_collect_records_a_rejected_call_as_failed(scripted_agent):
    """A call the tool threw out is recorded as such."""
    agent = scripted_agent(
        [
            tool_turn([("add", {"a": {"type": "string"}, "b": 1})]),
            answer_turn("Sorry, I could not work that out."),
        ]
    )
    calls, _ = collect_trajectory(agent, "go")
    assert calls[0].failed
    assert "Invalid arguments" in calls[0].error


def test_collect_pairs_outcomes_with_parallel_calls(scripted_agent):
    """Two calls in one turn keep their own outcomes."""
    agent = scripted_agent(
        [
            tool_turn(
                [
                    ("add", {"a": {"type": "string"}, "b": 1}),
                    ("add", {"a": 2, "b": 3}),
                ]
            ),
            answer_turn("Done."),
        ]
    )
    calls, _ = collect_trajectory(agent, "go")
    assert calls[0].failed
    assert not calls[1].failed
    assert calls[1].result == "5"


# ── run_trajectory_evaluation ───────────────────────────────


def test_run_scores_a_matching_row(scripted_agent):
    """A row whose trajectory matches passes."""
    agent = scripted_agent(
        [tool_turn([("echo", {"value": "hi"})]), answer_turn("Done.")]
    )
    results = run_trajectory_evaluation(
        agent,
        [{"input": "go", TRAJECTORY_FIELD: [{"name": "echo"}]}],
    )
    assert results[0].passed
    assert results[0].field_results == {TRAJECTORY_FIELD: True}


def test_run_scores_a_mismatching_row(scripted_agent):
    """A row whose expected tool was never called fails."""
    agent = scripted_agent([answer_turn("Guessed it.")])
    results = run_trajectory_evaluation(
        agent,
        [{"input": "go", TRAJECTORY_FIELD: [{"name": "echo"}]}],
    )
    assert not results[0].passed
    assert [e.name for e in results[0].trajectory.missing] == ["echo"]


def test_run_fails_a_row_whose_call_the_tool_rejected(scripted_agent):
    """The end-to-end shape of the defect this closed.

    A model sends a malformed argument, the tool rejects it, and
    the agent answers without ever getting a result — the row now
    fails instead of passing because a call by that name was made.
    """
    agent = scripted_agent(
        [
            tool_turn([("add", {"a": {"type": "string"}, "b": 1})]),
            answer_turn("Sorry, I could not work that out."),
        ]
    )
    results = run_trajectory_evaluation(
        agent,
        [{"input": "go", TRAJECTORY_FIELD: [{"name": "add"}]}],
    )
    assert not results[0].passed
    assert [e.name for e in results[0].trajectory.missing] == ["add"]
    assert results[0].trajectory.extra[0].failed


def test_run_records_the_answer_on_the_result(scripted_agent):
    """The final answer travels with the result for review."""
    agent = scripted_agent([answer_turn("We open at 9am.")])
    results = run_trajectory_evaluation(
        agent, [{"input": "go", TRAJECTORY_FIELD: []}]
    )
    assert results[0].response.response == "We open at 9am."


def test_run_turns_a_loop_error_into_a_failed_row(scripted_agent):
    """A runaway loop fails its row instead of the whole run."""
    agent = scripted_agent([tool_turn([("echo", {"value": "x"})])] * 5)
    results = run_trajectory_evaluation(
        agent, [{"input": "go", TRAJECTORY_FIELD: [{"name": "echo"}]}]
    )
    assert not results[0].passed
    assert "iterations" in results[0].error


def test_run_turns_a_bad_row_into_a_failed_row(scripted_agent):
    """An unparseable expectation fails only its own row."""
    agent = scripted_agent([answer_turn("Done.")])
    results = run_trajectory_evaluation(
        agent, [{"input": "go", TRAJECTORY_FIELD: "not json"}]
    )
    assert not results[0].passed
    assert results[0].error is not None


def test_run_rejects_a_non_string_input(scripted_agent):
    """A tool agent answers a message, not a field dict."""
    agent = scripted_agent([answer_turn("Done.")])
    results = run_trajectory_evaluation(
        agent, [{"input": {"text": "go"}, TRAJECTORY_FIELD: []}]
    )
    assert "must be a string" in results[0].error


def test_run_reports_progress(scripted_agent):
    """The progress callback fires once per row."""
    agent = scripted_agent([answer_turn("a"), answer_turn("b")])
    seen: list[tuple[int, int]] = []
    run_trajectory_evaluation(
        agent,
        [
            {"input": "one", TRAJECTORY_FIELD: []},
            {"input": "two", TRAJECTORY_FIELD: []},
        ],
        progress_callback=lambda cur, total: seen.append((cur, total)),
    )
    assert seen == [(1, 2), (2, 2)]


# ── compute_trajectory_metrics ──────────────────────────────


def test_metrics_count_calls_and_extras(scripted_agent):
    """Call-level totals reflect what matched and what did not."""
    agent = scripted_agent(
        [
            tool_turn([("echo", {"value": "x"})]),
            tool_turn([("add", {"a": 1, "b": 2})]),
            answer_turn("Done."),
        ]
    )
    results = run_trajectory_evaluation(
        agent,
        [
            {
                "input": "go",
                TRAJECTORY_FIELD: [{"name": "echo"}, {"name": "missing"}],
            }
        ],
    )
    metrics = compute_trajectory_metrics(results)
    assert metrics["expected_calls"] == 2
    assert metrics["matched_calls"] == 1
    assert metrics["extra_calls"] == 1
    assert metrics["mode"] == "in_order"


def test_metrics_report_per_tool_recall(scripted_agent):
    """Per-tool stats say which tool the agent misses."""
    agent = scripted_agent(
        [tool_turn([("echo", {"value": "x"})]), answer_turn("Done.")]
    )
    results = run_trajectory_evaluation(
        agent,
        [
            {
                "input": "go",
                TRAJECTORY_FIELD: [{"name": "echo"}, {"name": "add"}],
            }
        ],
    )
    per_tool = compute_trajectory_metrics(results)["per_tool"]
    assert per_tool["echo"] == {"expected": 1, "matched": 1}
    assert per_tool["add"] == {"expected": 1, "matched": 0}


def test_metrics_count_rejected_calls(scripted_agent):
    """A run says how many calls the tools threw out."""
    agent = scripted_agent(
        [
            tool_turn([("add", {"a": {"type": "string"}, "b": 1})]),
            answer_turn("Done."),
        ]
    )
    results = run_trajectory_evaluation(
        agent,
        [{"input": "go", TRAJECTORY_FIELD: [{"name": "add"}]}],
    )
    metrics = compute_trajectory_metrics(results)
    assert metrics["failed_calls"] == 1
    assert metrics["matched_calls"] == 0


def test_metrics_on_no_scored_rows_are_empty():
    """Metrics degrade cleanly when nothing was scored."""
    metrics = compute_trajectory_metrics([])
    assert metrics["expected_calls"] == 0
    assert metrics["failed_calls"] == 0
    assert metrics["per_tool"] == {}


# ── Call logging ────────────────────────────────────────────


def test_call_logger_fires_once_per_model_turn(scripted_agent):
    """Each model turn is one logged call, not each `act`."""
    agent = scripted_agent(
        [tool_turn([("echo", {"value": "x"})]), answer_turn("Done.")]
    )
    records = []
    agent.with_overrides(call_logger=records.append).act("go")
    assert len(records) == 2
    assert all(r.agent_name == "Scripted" for r in records)
    assert records[0].tokens == 10


def test_call_logger_fires_on_the_streaming_path(scripted_agent):
    """Streaming logs the same turns as the blocking path."""
    agent = scripted_agent(
        [tool_turn([("echo", {"value": "x"})]), answer_turn("Done.")]
    )
    records = []
    logged = agent.with_overrides(call_logger=records.append)
    list(logged.act_stream("go"))
    assert len(records) == 2


def test_without_a_logger_nothing_is_recorded(scripted_agent):
    """The logger is opt-in — the default path stays silent."""
    agent = scripted_agent([answer_turn("Done.")])
    assert agent.call_logger is None
    assert agent.act("go").response == "Done."


# ── Loop errors still propagate to a direct caller ──────────


def test_loop_error_propagates_from_collect(scripted_agent):
    """`collect_trajectory` does not swallow a runaway loop."""
    agent = scripted_agent([tool_turn([("echo", {"value": "x"})])] * 5)
    with pytest.raises(ToolLoopError):
        collect_trajectory(agent, "go")
