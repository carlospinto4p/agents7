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
        {"name": "order_total", "arguments": {"quantity": 3}}
     ]}

Matching semantics:

- **Order matters.** Expected calls must appear in the expected
  relative order.
- **`in_order` (default)** treats the expected calls as an ordered
  *subsequence*: extra calls the agent made are tolerated, but
  counted and reported. **`exact`** additionally requires that no
  extra call was made.
- **Arguments match on the stated keys only.** A row pins the
  arguments it cares about; unstated keys are ignored, and omitting
  ``arguments`` scores the call on its name alone.

Values are compared with the same case-insensitive rule the
field scorer uses, so a golden set loaded from CSV (all strings)
scores the same as one loaded from JSONL.
"""

import json
from dataclasses import dataclass, field
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
    """

    name: str
    arguments: dict = field(default_factory=dict)

    def format(self) -> str:
        """Render the call as ``name(k=v, ...)``.

        :return: A compact single-line rendering.
        """
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"{self.name}({args})"


@dataclass(frozen=True)
class ExpectedCall:
    """A tool call a golden-set row expects.

    :param name: The tool name that must be called.
    :param arguments: Arguments that must match. Only these keys
        are compared; ``None`` or empty scores the call on its
        name alone.
    """

    name: str
    arguments: dict | None = None

    def matches(self, call: ToolCall) -> bool:
        """Return whether *call* satisfies this expectation.

        :param call: An actual tool call.
        :return: True when the name matches and every stated
            argument matches.
        """
        if not _compare_field(call.name, self.name):
            return False
        if not self.arguments:
            return True
        return all(
            key in call.arguments
            and _compare_field(call.arguments[key], value)
            for key, value in self.arguments.items()
        )

    def format(self) -> str:
        """Render the expectation as ``name(k=v, ...)``.

        :return: A compact single-line rendering, with ``(…)`` when
            no arguments are pinned.
        """
        if not self.arguments:
            return f"{self.name}(…)"
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"{self.name}({args})"


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
            `TrajectoryMode.EXACT`, no extra call was made.
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
            order, with any missing or extra calls flagged.
        """
        actual = " -> ".join(c.name for c in self.actual) or "<none>"
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
        `TrajectoryMode.EXACT` does not.
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
    passed = complete and (mode != TrajectoryMode.EXACT or no_extras)

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
    :return: ``(calls, response)`` — the calls made in order, and
        the final `AgentResponse`. The response is None when the
        loop raised before answering, in which case the calls made
        up to that point are still returned.
    :raises Exception: Whatever the agent's loop raises (e.g.
        `ToolLoopError`) is propagated after the calls are
        recorded on the stream.
    """
    stream = agent.act_stream(user_input)
    calls: list[ToolCall] = []
    for event in stream:
        if event.kind == ToolStepKind.TOOL_CALL:
            calls.append(
                ToolCall(
                    name=event.name or "",
                    arguments=dict(event.arguments or {}),
                )
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
        ``matched_calls``, ``extra_calls``, and ``per_tool``
        (tool name → ``{"expected": n, "matched": n}``).
    """
    scored = [r for r in results if r.trajectory is not None]
    per_tool: dict[str, dict[str, int]] = {}
    expected_calls = matched_calls = extra_calls = 0

    for result in scored:
        trajectory = result.trajectory
        extra_calls += len(trajectory.extra)
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
        "per_tool": dict(sorted(per_tool.items())),
    }
