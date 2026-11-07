"""Trajectory scoring for tool-calling agents.

A `ToolAgent` answers in free text, so the label-vs-label scoring
in `agents7.eval.metrics` does not apply — two correct answers to
the same question rarely share a wording. What *is* checkable is
the agent's **trajectory**: which tools it called, in what order,
and with what arguments.

A golden-set row pairs an input with the calls the agent is
expected to make::

    {"input": "Can I order 3 blue mugs?",
     "tools": [
        {"name": "find_product", "arguments": {"query": "blue mug"}},
        {"name": "order_total", "arguments": {"quantity": 3},
         "defaults": {"express": false}}
     ]}

Matching semantics:

- **Order matters.** Expected calls must appear in the expected
  relative order.
- **`in_order` (default)** treats the expected calls as an ordered
  *subsequence*: extra calls the agent made are tolerated, but
  counted and reported. **`exact`** additionally requires that no
  extra call was made.
- **An empty ``tools`` list means no call at all**, and is scored
  under ``exact`` whatever mode the run is in — a row that expects
  nothing is asserting the agent must not reach for a tool, and
  under ``in_order`` it would otherwise have nothing to miss and so
  pass unconditionally.
- **Arguments match on the stated keys only.** A row pins the
  arguments it cares about; unstated keys are ignored, and omitting
  ``arguments`` scores the call on its name alone.
- **``defaults`` pins an argument as absent, or equal to a value.**
  An optional argument left out of ``arguments`` can be invented by
  the model and still match, while pinning it under ``arguments``
  fails a model that correctly omits it. ``defaults`` covers both:
  ``{"defaults": {"express": false}}`` accepts an omitted
  ``express`` and rejects ``express=true``.
- **A call the tool rejected satisfies no expectation.** A row
  naming ``find_product`` is asserting the agent looked the product
  up, and a call whose arguments the tool threw out did not do
  that. Such a call is reported as an extra one and marked
  ``[failed]``, so the row says the tool was reached for and never
  ran rather than passing on the attempt. Pinning the arguments is
  not an alternative: a row leaves an argument unpinned precisely
  when its wording is the model's to choose, and that is the case
  where a malformed value can still slip through.
- **``allow_failure`` is for a row that tests the error path.**
  ``{"name": "check_stock", "arguments": {"sku": "NOPE-99"},
  "allow_failure": true}`` is a row where the tool is *meant* to
  raise and the agent is meant to relay it, so the call counts as
  made whether it returned or not.

Values are compared with the same case-insensitive rule the
field scorer uses, so a golden set loaded from CSV (all strings)
scores the same as one loaded from JSONL.
"""

import json
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Callable

from agents7.agents.tool_agent import ToolAgent, ToolStepKind
from agents7.eval.metrics import EvalResult, _compare_field
from agents7.models.base import AgentResponse

# The golden-set column holding the expected trajectory, and the
# field name the run is scored under.
TRAJECTORY_FIELD = "tools"


class TrajectoryMode(StrEnum):
    """How strictly an actual trajectory must match the expected
    one.

    A `StrEnum`, so members compare equal to their string value —
    ``mode == "exact"`` works as well as
    ``mode == TrajectoryMode.EXACT``.
    """

    IN_ORDER = "in_order"
    EXACT = "exact"


@dataclass(frozen=True)
class ToolCall:
    """A tool call the agent actually made.

    :param name: The tool's name.
    :param arguments: The arguments it was called with.
    :param result: The text the tool returned, or None when the
        call was built without running one.
    :param error: The failure message when the tool rejected the
        arguments or raised, else None.
    """

    name: str
    arguments: dict = field(default_factory=dict)
    result: str | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        """Whether the tool rejected this call or raised on it."""
        return self.error is not None

    def format(self) -> str:
        """Render the call as ``name(k=v, ...)``.

        A call the tool rejected is marked ``[failed]``, so a
        report line says why an expectation went unsatisfied.

        :return: A compact single-line rendering.
        """
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        suffix = " [failed]" if self.failed else ""
        return f"{self.name}({args}){suffix}"


@dataclass(frozen=True)
class ExpectedCall:
    """A tool call a golden-set row expects.

    :param name: The tool name that must be called.
    :param arguments: Arguments that must be present and match.
        Only these keys are compared; ``None`` or empty scores the
        call on its name alone.
    :param defaults: Arguments that must be *absent, or equal to
        this*. Use it for an optional argument the agent should
        leave alone unless asked: pinning it under *arguments*
        would fail a model that correctly omits it, while leaving
        it out altogether lets a model invent it and still match.
    :param allow_failure: Whether a call the tool rejected still
        satisfies this expectation. False by default, so a row
        cannot pass on a call that never returned; set it on a row
        that deliberately exercises the error path, where the tool
        is meant to raise and the agent is meant to relay it.
    """

    name: str
    arguments: dict | None = None
    defaults: dict | None = None
    allow_failure: bool = False

    def matches(self, call: ToolCall) -> bool:
        """Return whether *call* satisfies this expectation.

        :param call: An actual tool call.
        :return: True when the call ran, the name matches, every
            stated argument matches, and every stated default is
            absent or equal. A call the tool rejected satisfies
            nothing — the agent reached for the tool but never got
            an answer out of it — unless *allow_failure* says the
            row is testing that very path.
        """
        if call.failed and not self.allow_failure:
            return False
        if not _compare_field(call.name, self.name):
            return False
        if self.arguments and not all(
            key in call.arguments
            and _compare_field(call.arguments[key], value)
            for key, value in self.arguments.items()
        ):
            return False
        return all(
            key not in call.arguments
            or _compare_field(call.arguments[key], value)
            for key, value in (self.defaults or {}).items()
        )

    def format(self) -> str:
        """Render the expectation as ``name(k=v, ...)``.

        Pinned arguments render as ``k=v``; defaults render as
        ``k?=v``, the ``?`` reading as "absent, or this". A row
        that tolerates a rejected call is marked ``[may fail]``.

        :return: A compact single-line rendering, with ``(…)`` when
            nothing is pinned.
        """
        parts = [f"{k}={v!r}" for k, v in (self.arguments or {}).items()]
        parts += [f"{k}?={v!r}" for k, v in (self.defaults or {}).items()]
        suffix = " [may fail]" if self.allow_failure else ""
        if not parts:
            return f"{self.name}(…){suffix}"
        return f"{self.name}({', '.join(parts)}){suffix}"


@dataclass
class TrajectoryResult:
    """The outcome of matching one actual trajectory.

    Attributes:
        expected: The calls the row expected, in order.
        actual: The calls the agent made, in order.
        matched: For each expected call, the index into *actual*
            that satisfied it, or None when unmatched.
        mode: The mode the match was scored under.
        passed: True when every expected call matched and, under
            `TrajectoryMode.EXACT` or an empty *expected*, no extra
            call was made.
    """

    expected: list[ExpectedCall]
    actual: list[ToolCall]
    matched: list[int | None]
    mode: TrajectoryMode
    passed: bool

    @property
    def missing(self) -> list[ExpectedCall]:
        """The expected calls that never matched."""
        return [
            exp
            for exp, idx in zip(self.expected, self.matched)
            if idx is None
        ]

    @property
    def extra(self) -> list[ToolCall]:
        """The actual calls that satisfied no expectation."""
        used = {idx for idx in self.matched if idx is not None}
        return [call for i, call in enumerate(self.actual) if i not in used]

    def format(self) -> str:
        """Render a one-line ``expected -> actual`` summary.

        :return: A compact summary naming the called tools in
            order, with rejected calls marked and any missing or
            extra calls flagged.
        """
        names = (
            f"{c.name} (failed)" if c.failed else c.name for c in self.actual
        )
        actual = " -> ".join(names) or "<none>"
        parts = [f"called: {actual}"]
        if self.missing:
            parts.append(
                "missing: " + ", ".join(e.format() for e in self.missing)
            )
        if self.extra:
            parts.append(
                "extra: " + ", ".join(c.format() for c in self.extra)
            )
        return " | ".join(parts)


def parse_expected_calls(raw: object) -> list[ExpectedCall]:
    """Parse a golden-set ``tools`` cell into expected calls.

    Accepts a list of dicts (JSONL / JSON) or a JSON string (the
    shape a CSV cell arrives as). A bare string entry is treated
    as a tool name with no pinned arguments, so a row can expect
    ``["find_product", "check_stock"]``.

    :param raw: The raw cell value.
    :return: The parsed expectations, in order.
    :raises ValueError: If *raw* is not a list of calls, or a call
        entry has no ``name``.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"`{TRAJECTORY_FIELD}` is not valid JSON: {exc}"
            ) from exc
    if not isinstance(raw, list):
        raise ValueError(
            f"`{TRAJECTORY_FIELD}` must be a list of expected calls, "
            f"got {type(raw).__name__}."
        )

    calls: list[ExpectedCall] = []
    for entry in raw:
        if isinstance(entry, str):
            calls.append(ExpectedCall(name=entry))
            continue
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError(
                "Each expected call must be a tool name or a dict "
                f"with a `name`, got: {entry!r}"
            )
        calls.append(
            ExpectedCall(
                name=str(entry["name"]),
                arguments=entry.get("arguments") or None,
                defaults=entry.get("defaults") or None,
                allow_failure=bool(entry.get("allow_failure")),
            )
        )
    return calls


def match_trajectory(
    expected: list[ExpectedCall],
    actual: list[ToolCall],
    mode: TrajectoryMode = TrajectoryMode.IN_ORDER,
) -> TrajectoryResult:
    """Match an actual trajectory against the expected one.

    Each expectation is matched against the earliest still-unused
    actual call that satisfies it, which keeps the longest possible
    suffix available for the remaining expectations — so a greedy
    left-to-right scan finds a match whenever one exists.

    :param expected: The expected calls, in order.
    :param actual: The calls the agent made, in order.
    :param mode: `TrajectoryMode.IN_ORDER` tolerates extra calls;
        `TrajectoryMode.EXACT` does not. An empty *expected* is
        scored under `TrajectoryMode.EXACT` in either mode, so a row
        expecting no calls fails when the agent made one.
    :return: The populated `TrajectoryResult`.
    """
    matched: list[int | None] = []
    cursor = 0
    for exp in expected:
        found: int | None = None
        for i in range(cursor, len(actual)):
            if exp.matches(actual[i]):
                found = i
                break
        matched.append(found)
        if found is not None:
            cursor = found + 1

    complete = all(idx is not None for idx in matched)
    no_extras = len(actual) == len(expected)
    # A row expecting no calls is asserting exactly that. With
    # nothing to miss, `in_order`'s subsequence semantics degenerate
    # to "anything passes", so an empty expectation is scored under
    # `exact` whatever mode the run is in.
    strict = mode == TrajectoryMode.EXACT or not expected
    passed = complete and (not strict or no_extras)

    return TrajectoryResult(
        expected=expected,
        actual=actual,
        matched=matched,
        mode=mode,
        passed=passed,
    )


def collect_trajectory(
    agent: ToolAgent,
    user_input: str,
) -> tuple[list[ToolCall], AgentResponse | None]:
    """Run *agent* once, recording the tools it calls.

    Drives the agent's own streaming loop, so the trajectory is
    what the agent really did rather than a reconstruction.

    :param agent: The tool agent to run.
    :param user_input: The user message to answer.
    :return: ``(calls, response)`` — the calls made in order, each
        carrying the outcome its tool returned, and the final
        `AgentResponse`. The response is None when the loop raised
        before answering, in which case the calls made up to that
        point are still returned.
    :raises Exception: Whatever the agent's loop raises (e.g.
        `ToolLoopError`) is propagated after the calls are
        recorded on the stream.
    """
    stream = agent.act_stream(user_input)
    calls: list[ToolCall] = []
    # The agent emits one result event per call event, in the same
    # order, so a queue of the calls still waiting for a result
    # pairs them — across iterations as well as within one.
    awaiting: list[int] = []
    for event in stream:
        if event.kind == ToolStepKind.TOOL_CALL:
            awaiting.append(len(calls))
            calls.append(
                ToolCall(
                    name=event.name or "",
                    arguments=dict(event.arguments or {}),
                )
            )
        elif event.kind == ToolStepKind.TOOL_RESULT and awaiting:
            index = awaiting.pop(0)
            calls[index] = replace(
                calls[index], result=event.result, error=event.error
            )
    return calls, stream.agent_response


def run_trajectory_evaluation(
    agent: ToolAgent,
    eval_data: list[dict],
    mode: TrajectoryMode = TrajectoryMode.IN_ORDER,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[EvalResult]:
    """Score *agent*'s trajectory on every row in *eval_data*.

    Runs sequentially: each row is a multi-turn tool loop, so there
    is no batch call to amortise, and interleaving loops would make
    the recorded trajectories ambiguous.

    :param agent: The tool agent to evaluate.
    :param eval_data: Rows carrying an ``input`` (the user message)
        and a ``tools`` list of expected calls.
    :param mode: Trajectory match strictness.
    :param progress_callback: Optional ``(current, total)`` callable
        invoked after each row.
    :return: One `EvalResult` per row, each carrying its
        `TrajectoryResult`.
    """
    total = len(eval_data)
    results: list[EvalResult] = []

    for index, row in enumerate(eval_data, start=1):
        results.append(_evaluate_row(agent, row, mode))
        if progress_callback is not None:
            progress_callback(index, total)

    return results


def _evaluate_row(
    agent: ToolAgent,
    row: dict,
    mode: TrajectoryMode,
) -> EvalResult:
    """Run one row and score the trajectory it produced.

    :param agent: The tool agent to run.
    :param row: An eval row with ``input`` and ``tools``.
    :param mode: Trajectory match strictness.
    :return: The populated `EvalResult`.
    """
    user_input = row.get("input", "")
    expected_raw = row.get(TRAJECTORY_FIELD, [])

    try:
        expected = parse_expected_calls(expected_raw)
        text = _as_user_message(user_input)
    except ValueError as exc:
        return EvalResult(
            input_data=user_input,
            expected={TRAJECTORY_FIELD: expected_raw},
            error=str(exc),
        )

    result = EvalResult(
        input_data=user_input,
        expected={TRAJECTORY_FIELD: [e.format() for e in expected]},
    )
    try:
        calls, response = collect_trajectory(agent, text)
    except Exception as exc:
        result.error = str(exc)
        return result

    trajectory = match_trajectory(expected, calls, mode)
    result.response = response
    result.trajectory = trajectory
    result.actual = {TRAJECTORY_FIELD: [c.format() for c in calls]}
    result.field_results = {TRAJECTORY_FIELD: trajectory.passed}
    result.passed = trajectory.passed
    return result


def _as_user_message(value: object) -> str:
    """Coerce a row's input into the string a tool agent takes.

    :param value: The row's ``input`` value.
    :return: The user message.
    :raises ValueError: If *value* is not a string — a tool agent
        answers a message, not a field dict, so a structured input
        is a golden-set authoring error worth surfacing.
    """
    if isinstance(value, str):
        return value
    raise ValueError(
        "A tool agent's `input` must be a string (the user "
        f"message), got {type(value).__name__}."
    )


def compute_trajectory_metrics(results: list[EvalResult]) -> dict:
    """Aggregate trajectory-specific metrics.

    Complements `compute_metrics`, which already reports the
    pass rate: this adds the call-level detail that says *why* a
    trajectory failed — which tools were missed and how many extra
    calls were made.

    :param results: Results from `run_trajectory_evaluation`.
    :return: Dict with ``mode``, ``expected_calls``,
        ``matched_calls``, ``extra_calls``, ``failed_calls`` (every
        call the tool rejected, whether or not a row expecting a
        failure matched it), and ``per_tool`` (tool name →
        ``{"expected": n, "matched": n}``).
    """
    scored = [r for r in results if r.trajectory is not None]
    per_tool: dict[str, dict[str, int]] = {}
    expected_calls = matched_calls = extra_calls = failed_calls = 0

    for result in scored:
        trajectory = result.trajectory
        extra_calls += len(trajectory.extra)
        failed_calls += sum(1 for c in trajectory.actual if c.failed)
        for exp, idx in zip(trajectory.expected, trajectory.matched):
            stats = per_tool.setdefault(
                exp.name, {"expected": 0, "matched": 0}
            )
            stats["expected"] += 1
            expected_calls += 1
            if idx is not None:
                stats["matched"] += 1
                matched_calls += 1

    mode = scored[0].trajectory.mode if scored else TrajectoryMode.IN_ORDER
    return {
        "mode": str(mode),
        "expected_calls": expected_calls,
        "matched_calls": matched_calls,
        "extra_calls": extra_calls,
        "failed_calls": failed_calls,
        "per_tool": dict(sorted(per_tool.items())),
    }
