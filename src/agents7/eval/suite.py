"""The composed evaluation route: data -> run -> metrics -> report.

`run_eval_suite` is the single entry point that ties together the
pieces that previously had to be wired by hand: eval-data loading
and normalisation (`agents7.eval.data`), agent construction, the
evaluation run and its metrics (`agents7.eval.metrics`), the
Markdown report (`agents7.eval.export`), and the call-log summary
(`agents7.call_log`).

Usage::

    from agents7.eval import run_eval_suite

    run = run_eval_suite(
        "configs/sentiment.yml",
        "data/golden_set_40.jsonl",
        report="reports/sentiment.md",
        call_log="reports/sentiment_calls.jsonl",
    )
    print(run.format())
    print(run.metrics["accuracy"])

The individual primitives remain public for callers that need
finer control; this module is the path to follow when you just
want an agent scored against a golden set.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agents7.agents.base import BaseAgent
from agents7.agents.tool_agent import ToolAgent
from agents7.call_log import (
    CallLogSummary,
    CallRecord,
    JsonCallLogger,
    generate_trace_id,
    summarize_call_log,
)
from agents7.eval.data import (
    build_agent,
    load_config_dict,
    load_eval_rows,
    model_output_fields,
    normalise_rows,
)
from agents7.eval.export import export_eval_report
from agents7.eval.metrics import (
    EvalResult,
    compute_metrics,
    compute_per_category,
    resolve_category_field,
    run_evaluation,
)
from agents7.eval.trajectory import (
    TRAJECTORY_FIELD,
    TrajectoryMode,
    compute_trajectory_metrics,
    run_trajectory_evaluation,
)


@dataclass
class EvalRun:
    """Everything one evaluation run produced.

    Attributes:
        agent_name: Name of the evaluated agent.
        model_id: Model identifier the agent ran against.
        data_path: Path to the golden set that was scored.
        fields: Expected output fields that were compared.
        results: Per-row `EvalResult` list.
        metrics: Aggregate metrics from `compute_metrics`.
        call_log: Summary of the calls the run made. None only
            when the agent made no LLM calls at all — a
            rule-based agent has nothing to summarize.
        report_path: Path of the written Markdown report, when a
            *report* was requested. None otherwise.
    """

    agent_name: str
    model_id: str
    data_path: Path
    fields: list[str]
    results: list[EvalResult] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    call_log: CallLogSummary | None = None
    report_path: Path | None = None

    def format(self) -> str:
        """Render a human-readable summary of the run.

        :return: A multi-line summary block covering the agent,
            the golden set, overall accuracy, and per-field
            accuracy.
        """
        lines = [
            f"Agent:    {self.agent_name}",
            f"Model:    {self.model_id}",
            f"Data:     {self.data_path.name}",
            f"Fields:   {', '.join(self.fields)}",
            f"Accuracy: {self.metrics.get('accuracy', 0.0):.1%} "
            f"({self.metrics.get('passed', 0)}/"
            f"{self.metrics.get('total', 0)})",
        ]
        trajectory = self.metrics.get("trajectory")
        if trajectory:
            lines.extend(self._trajectory_lines(trajectory))
        elif per_field := self.metrics.get("per_field", {}):
            lines.append("Per-field accuracy:")
            lines.extend(
                f"  {name}: {acc:.1%}" for name, acc in per_field.items()
            )
        if self.call_log is not None:
            lines.append("")
            lines.append(self.call_log.format())
        if self.report_path is not None:
            lines.append("")
            lines.append(f"Report written to {self.report_path}")
        return "\n".join(lines)

    @staticmethod
    def _trajectory_lines(trajectory: dict) -> list[str]:
        """Render the trajectory block of a tool run's summary.

        :param trajectory: The run's ``trajectory`` metrics.
        :return: Summary lines covering the match mode, the
            call-level totals, and per-tool recall.
        """
        expected = trajectory.get("expected_calls", 0)
        matched = trajectory.get("matched_calls", 0)
        rate = matched / expected if expected else 0.0
        lines = [
            f"Match:    {trajectory.get('mode', '')}",
            f"Calls:    {matched}/{expected} expected calls made "
            f"({rate:.1%}), {trajectory.get('extra_calls', 0)} extra, "
            f"{trajectory.get('failed_calls', 0)} rejected",
        ]
        per_tool = trajectory.get("per_tool", {})
        if per_tool:
            lines.append("Per-tool recall:")
            lines.extend(
                f"  {name}: {stats['matched']}/{stats['expected']}"
                for name, stats in per_tool.items()
            )
        return lines


def resolve_eval_fields(
    config_path: str | Path,
    rows: list[dict],
    fields: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Normalise raw rows and resolve the fields to score.

    An explicit *fields* list overrides the fields derived from
    the golden set and the agent's declared ``output_model``.

    :param config_path: Path to the agent's YAML config.
    :param rows: Raw rows from `load_eval_rows`.
    :param fields: Optional explicit field names to score.
    :return: ``(eval_rows, expected_fields)``.
    :raises ValueError: If no expected fields can be determined.
    """
    config = load_config_dict(config_path)
    eval_rows, derived = normalise_rows(rows, model_output_fields(config))
    resolved = list(fields) if fields else derived
    if not resolved:
        raise ValueError(
            "Could not determine expected output fields — pass "
            "`fields` explicitly."
        )
    return eval_rows, resolved


def run_eval_suite(
    config: str | Path,
    data: str | Path,
    *,
    agent_type: str | None = None,
    fields: list[str] | None = None,
    report: str | Path | None = None,
    confusion_field: str | None = None,
    call_log: str | Path | None = None,
    notes: list[str] | None = None,
    trajectory_mode: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> EvalRun:
    """Evaluate a YAML-configured agent against a golden set.

    Runs the whole route in one call: load and normalise the eval
    data, build the agent, score every row, aggregate metrics,
    summarize the calls made, and optionally write a Markdown
    report.

    A **tool agent** is scored on its *trajectory* rather than on
    output fields: its rows carry a ``tools`` list of expected
    calls, and the run reports whether it called the right tools,
    in the right order, with the right arguments (see
    `agents7.eval.trajectory`). Everything else about the route —
    the call log, the metrics, the report — is identical.

    :param config: Path to the agent's YAML config file.
    :param data: Path to the golden set (``.jsonl``, ``.json``,
        or ``.csv``).
    :param agent_type: Optional explicit agent type, overriding
        the config's ``agent_type`` field.
    :param fields: Optional explicit expected-field names. When
        omitted, they are derived from the golden set and the
        config's ``output_model``. Ignored for a tool agent, which
        is always scored on its trajectory.
    :param report: When set, write a Markdown report to this path.
    :param confusion_field: Categorical field for the report's
        confusion matrix. Defaults to ``category`` when scored,
        otherwise the first scored field.
    :param call_log: When set, also write every call made during
        the run as JSON lines to this path. The run's
        `CallLogSummary` is attached either way — a caller does
        not have to write a file to get the run's tokens,
        duration and cost.
    :param notes: Optional extra note lines for the report, appended
        after the standard eval-set and accuracy notes. Use them for
        domain caveats a reader needs to interpret the numbers.
    :param trajectory_mode: For a tool agent, ``in_order`` (extra
        calls tolerated) or ``exact`` (they are not). Defaults to
        the config's ``trajectory_mode``, then to ``in_order``.
    :param progress_callback: Optional ``(current, total)``
        callable forwarded to the run.
    :return: The populated `EvalRun`.
    :raises ValueError: If the golden set is empty, the expected
        fields cannot be determined, or a tool agent's golden set
        has no ``tools`` column.
    """
    data_path = Path(data)
    rows = load_eval_rows(data_path)
    if not rows:
        raise ValueError(f"Golden set is empty: {data_path}")

    agent = build_agent(config, agent_type)
    is_tool = isinstance(agent, ToolAgent)

    if is_tool:
        eval_rows = _trajectory_rows(rows, data_path)
        resolved_fields = [TRAJECTORY_FIELD]
    else:
        eval_rows, resolved_fields = resolve_eval_fields(config, rows, fields)

    # Every run collects its own calls, whether or not a *call_log*
    # path was asked for: the tokens, duration and cost of a run are
    # measurements it produces, and a caller that wants them should
    # not have to write a file to see them. The path only decides
    # whether they are also persisted.
    logger: JsonCallLogger | None = None
    logged: list[CallRecord] = []
    if call_log is not None:
        logger = JsonCallLogger(call_log)
    write = logger
    # A config may declare its own sink (`call_log:` or
    # `call_logger:`). Chain it rather than replace it — collecting
    # for the summary must not silence a log the config asked for.
    configured = getattr(agent, "call_logger", None)

    def _record(record: CallRecord) -> None:
        """Keep a call for this run's summary, and pass it on."""
        logged.append(record)
        if write is not None:
            write(record)
        if configured is not None:
            configured(record)

    # One eval run is one run: every row's calls share an id
    # so a log holding several runs can be split back into
    # them. A ToolAgent takes the same field.
    agent = agent.with_overrides(
        call_logger=_record,
        run_id=getattr(agent, "run_id", None) or generate_trace_id(),
    )

    try:
        if is_tool:
            results = run_trajectory_evaluation(
                agent,
                eval_rows,
                _resolve_trajectory_mode(config, trajectory_mode),
                progress_callback,
            )
        else:
            results = run_evaluation(
                agent, eval_rows, resolved_fields, progress_callback
            )
    finally:
        if logger is not None:
            logger.close()

    metrics = compute_metrics(results, resolved_fields)
    if is_tool:
        # A trajectory is a sequence of calls, not a label, so
        # there is nothing to break it down by.
        metrics["trajectory"] = compute_trajectory_metrics(results)
        metrics["per_category"] = {}
    else:
        category_field = resolve_category_field(resolved_fields)
        metrics["per_category"] = (
            compute_per_category(results, category_field)
            if category_field
            else {}
        )
    # Summarize the in-memory records rather than re-reading the
    # log file: ``JsonCallLogger`` appends, so a path reused across
    # runs holds earlier runs' records too and re-reading it would
    # report cumulative totals as if they were this run's. An agent
    # that makes no LLM calls at all (a rule-based one) logs
    # nothing, and gets no summary.
    summary = summarize_call_log(logged) if logged else None

    run = EvalRun(
        agent_name=agent.name,
        model_id=agent.engine.model_id,
        data_path=data_path,
        fields=resolved_fields,
        results=results,
        metrics=metrics,
        call_log=summary,
    )

    if report is not None:
        run.report_path = _write_report(
            run, agent, report, confusion_field, notes
        )
    return run


def _trajectory_rows(rows: list[dict], data_path: Path) -> list[dict]:
    """Shape raw rows for a trajectory run.

    Trajectory rows need no output-model reconciliation — the
    input and the expected calls are both named explicitly — so
    they skip `normalise_rows` and are only validated here.

    :param rows: Raw rows from `load_eval_rows`.
    :param data_path: The golden set's path, for error messages.
    :return: Rows carrying ``input`` and ``tools``.
    :raises ValueError: If no row declares a ``tools`` column.
    """
    if not any(TRAJECTORY_FIELD in row for row in rows):
        raise ValueError(
            f"`{data_path.name}` has no `{TRAJECTORY_FIELD}` column — a "
            "tool agent is scored on the calls it makes, so each row "
            "needs the trajectory it is expected to take."
        )
    return [
        {
            "input": row.get("input", ""),
            TRAJECTORY_FIELD: row.get(TRAJECTORY_FIELD, []),
        }
        for row in rows
    ]


def _resolve_trajectory_mode(
    config: str | Path,
    override: str | None,
) -> TrajectoryMode:
    """Resolve the trajectory match mode for a run.

    Precedence: an explicit *override*, then the config's
    ``trajectory_mode``, then `TrajectoryMode.IN_ORDER`.

    :param config: Path to the agent's YAML config.
    :param override: Explicit mode passed by the caller.
    :return: The resolved mode.
    :raises ValueError: If the resolved mode is not a known mode.
    """
    raw = override or load_config_dict(config).get("trajectory_mode")
    if not raw:
        return TrajectoryMode.IN_ORDER
    try:
        return TrajectoryMode(str(raw).lower())
    except ValueError as exc:
        modes = ", ".join(m.value for m in TrajectoryMode)
        raise ValueError(
            f"Unknown trajectory_mode '{raw}' — must be one of: {modes}"
        ) from exc


def _write_report(
    run: EvalRun,
    agent: BaseAgent,
    report: str | Path,
    confusion_field: str | None,
    notes: list[str] | None = None,
) -> Path:
    """Write the run's Markdown report.

    :param run: The populated run (minus its report path).
    :param agent: The agent that was evaluated.
    :param report: Output path for the report.
    :param confusion_field: Explicit confusion-matrix field, or
        None to pick ``category`` when scored.
    :param notes: Extra note lines appended after the standard ones.
    :return: The path the report was written to.
    """
    if run.metrics.get("trajectory"):
        # A trajectory is a sequence of calls, not a label — there
        # is no categorical field to cross-tabulate.
        confusion = None
    else:
        confusion = confusion_field or resolve_category_field(run.fields)
    export_eval_report(
        run.results,
        run.metrics,
        report,
        agent_name=agent.name,
        model_id=agent.engine.model_id,
        confusion_field=confusion,
        call_log=run.call_log,
        notes=[
            f"Eval set: `{run.data_path.name}` ({run.metrics['total']} rows)",
            f"Accuracy: {run.metrics['accuracy']:.0%} "
            f"({run.metrics['passed']}/{run.metrics['total']})",
            *(notes or []),
        ],
    )
    return Path(report)
