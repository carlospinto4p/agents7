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
from agents7.call_log import (
    CallLogSummary,
    CallRecord,
    JsonCallLogger,
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
    run_evaluation,
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
        call_log: Summary of the calls the run made, when a
            *call_log* path was requested. None otherwise.
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
        per_field = self.metrics.get("per_field", {})
        if per_field:
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
    progress_callback: Callable[[int, int], None] | None = None,
) -> EvalRun:
    """Evaluate a YAML-configured agent against a golden set.

    Runs the whole route in one call: load and normalise the eval
    data, build the agent, score every row, aggregate metrics,
    summarize the calls made, and optionally write a Markdown
    report.

    :param config: Path to the agent's YAML config file.
    :param data: Path to the golden set (``.jsonl``, ``.json``,
        or ``.csv``).
    :param agent_type: Optional explicit agent type, overriding
        the config's ``agent_type`` field.
    :param fields: Optional explicit expected-field names. When
        omitted, they are derived from the golden set and the
        config's ``output_model``.
    :param report: When set, write a Markdown report to this path.
    :param confusion_field: Categorical field for the report's
        confusion matrix. Defaults to ``category`` when scored,
        otherwise the first scored field.
    :param call_log: When set, log every call made during the run
        as JSON lines to this path and attach the resulting
        `CallLogSummary` to the returned run.
    :param notes: Optional extra note lines for the report, appended
        after the standard eval-set and accuracy notes. Use them for
        domain caveats a reader needs to interpret the numbers.
    :param progress_callback: Optional ``(current, total)``
        callable forwarded to `run_evaluation`.
    :return: The populated `EvalRun`.
    :raises ValueError: If the golden set is empty or the expected
        fields cannot be determined.
    """
    data_path = Path(data)
    rows = load_eval_rows(data_path)
    if not rows:
        raise ValueError(f"Golden set is empty: {data_path}")

    eval_rows, resolved_fields = resolve_eval_fields(config, rows, fields)
    agent = build_agent(config, agent_type)

    logger: JsonCallLogger | None = None
    logged: list[CallRecord] = []
    if call_log is not None:
        logger = JsonCallLogger(call_log)
        write = logger

        def _record(record: CallRecord) -> None:
            """Persist a call and keep it for this run's summary."""
            logged.append(record)
            write(record)

        agent = agent.with_overrides(call_logger=_record)

    try:
        results = run_evaluation(
            agent, eval_rows, resolved_fields, progress_callback
        )
    finally:
        if logger is not None:
            logger.close()

    metrics = compute_metrics(results, resolved_fields)
    summary = _summarize_calls(call_log, logged)

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


def _summarize_calls(
    call_log: str | Path | None,
    logged: list[CallRecord],
) -> CallLogSummary | None:
    """Summarize the calls *this run* made.

    Summarizes the in-memory records rather than re-reading the
    log file: ``JsonCallLogger`` appends, so a path reused across
    runs holds earlier runs' records too and re-reading it would
    report cumulative totals as if they were this run's.

    :param call_log: Path the run logged to, or None.
    :param logged: Records captured during this run.
    :return: The summary, or None when no log was requested or
        the run made no logged calls.
    """
    if call_log is None or not logged:
        return None
    return summarize_call_log(logged)


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
    confusion = confusion_field or (
        "category" if "category" in run.fields else run.fields[0]
    )
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
