"""Export agent responses to JSONL, CSV, DataFrame, and
evaluation reports."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from agents7.call_log import CallLogSummary
from agents7.eval.metrics import _resolve_enum
from agents7.models.base import AgentResponse


def export_jsonl(
    responses: list[AgentResponse],
    path: str | Path,
) -> None:
    """Write agent responses to a JSONL file.

    Each line is a JSON object with keys ``response``,
    ``model`` (dict or null), ``thinking``, and ``metrics``.

    :param responses: List of `AgentResponse` from
        ``act()`` or ``act_in_batch()``.
    :param path: Output file path.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for r in responses:
            obj = {
                "response": r.response,
                "model": (r.model.model_dump() if r.model else None),
                "thinking": r.thinking,
                "metrics": r.metrics.model_dump(),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def export_csv(
    responses: list[AgentResponse],
    path: str | Path,
    fields: list[str] | None = None,
) -> None:
    """Write agent responses to a CSV file.

    Columns: ``response``, plus one column per model field
    (flattened from ``model.model_dump()``). If ``fields``
    is provided, only those model fields are included.

    :param responses: List of `AgentResponse` from
        ``act()`` or ``act_in_batch()``.
    :param path: Output file path.
    :param fields: Optional list of model field names to
        include. Defaults to all fields from the first
        non-None model.
    """
    path = Path(path)
    if fields is None:
        fields = _detect_fields(responses)

    fieldnames = ["response"] + fields
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in responses:
            row: dict[str, str] = {"response": r.response}
            if r.model is not None:
                dump = r.model.model_dump()
                for field in fields:
                    val = dump.get(field)
                    row[field] = _format_value(val)
            else:
                for field in fields:
                    row[field] = ""
            writer.writerow(row)


def _detect_fields(
    responses: list[AgentResponse],
) -> list[str]:
    """Detect model field names from the first non-None model."""
    for r in responses:
        if r.model is not None:
            return list(r.model.model_dump().keys())
    return []


def _format_value(val: object) -> str:
    """Format a model field value for CSV output."""
    if val is None:
        return ""
    return str(_resolve_enum(val))


def export_dataframe(
    responses: list[AgentResponse],
    fields: list[str] | None = None,
):
    """Convert agent responses to a Polars DataFrame.

    Requires the ``polars`` package (installed with
    ``agents7[ui]`` or ``pip install polars``).

    Columns: ``response``, plus one column per model field
    (flattened from ``model.model_dump()``). If ``fields``
    is provided, only those model fields are included.

    :param responses: List of `AgentResponse` from
        ``act()`` or ``act_in_batch()``.
    :param fields: Optional list of model field names to
        include. Defaults to all fields from the first
        non-None model.
    :return: A ``polars.DataFrame``.
    :raises ImportError: If ``polars`` is not installed.
    """
    try:
        import polars as pl
    except ImportError as e:
        raise ImportError(
            "The `polars` package is required for "
            "`export_dataframe`. Install it with: "
            "pip install polars"
        ) from e

    if fields is None:
        fields = _detect_fields(responses)

    data: dict[str, list] = {"response": []}
    for f in fields:
        data[f] = []

    for r in responses:
        data["response"].append(r.response)
        if r.model is not None:
            dump = r.model.model_dump()
            for f in fields:
                data[f].append(_format_value(dump.get(f)))
        else:
            for f in fields:
                data[f].append(None)

    return pl.DataFrame(data)


def export_eval_report(
    results: list,
    metrics: dict,
    path: str | Path,
    agent_name: str | None = None,
    model_id: str | None = None,
    confusion_field: str | None = None,
    notes: list[str] | None = None,
    call_log: CallLogSummary | None = None,
) -> None:
    """Write an evaluation report as a Markdown file.

    :param results: List of `EvalResult` from
        ``run_evaluation()``.
    :param metrics: Dict from ``compute_metrics()``.
    :param path: Output file path.
    :param agent_name: Agent name for the header.
    :param model_id: Model identifier for the header.
    :param confusion_field: If set, include a confusion
        matrix for this categorical field (e.g.
        ``"category"``). Requires
        ``agents7.eval.metrics.build_confusion_matrix``.
    :param notes: Optional list of note strings to include
        in a "Notes" section after the header.
    :param call_log: Optional `CallLogSummary` from
        ``summarize_call_log()``. When given, a "Call Log"
        section with token and duration totals is included.
    """
    path = Path(path)
    lines: list[str] = []

    # --- Header ---
    lines.append("# Evaluation Report")
    lines.append("")
    if agent_name:
        lines.append(f"- **Agent**: {agent_name}")
    if model_id:
        lines.append(f"- **Model**: {model_id}")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"- **Date**: {now}")
    lines.append("")

    # --- Notes ---
    if notes:
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    # --- Overall metrics ---
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total | {metrics.get('total', 0)} |")
    lines.append(f"| Passed | {metrics.get('passed', 0)} |")
    lines.append(f"| Failed | {metrics.get('failed', 0)} |")
    accuracy = metrics.get("accuracy", 0.0)
    lines.append(f"| Accuracy | {accuracy:.1%} |")
    lines.append("")

    # --- Per-field accuracy ---
    per_field = metrics.get("per_field", {})
    if per_field:
        lines.append("## Per-Field Accuracy")
        lines.append("")
        lines.append("| Field | Accuracy |")
        lines.append("|-------|----------|")
        for fname, acc in per_field.items():
            lines.append(f"| {fname} | {acc:.1%} |")
        lines.append("")

    # --- Timing ---
    timing = metrics.get("timing", {})
    if timing and timing.get("total_s", 0) > 0:
        lines.append("## Response Times")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total | {timing['total_s']:.2f}s |")
        lines.append(f"| Average | {timing['avg_s']:.2f}s |")
        lines.append(f"| Min | {timing['min_s']:.2f}s |")
        lines.append(f"| Max | {timing['max_s']:.2f}s |")
        lines.append("")

    # --- Trajectory (tool agents) ---
    trajectory = metrics.get("trajectory")
    if trajectory:
        _append_trajectory(lines, trajectory)

    # --- Call log ---
    if call_log is not None:
        _append_call_log(lines, call_log)

    # --- Confusion matrix ---
    if confusion_field:
        _append_confusion_matrix(lines, results, confusion_field)

    # --- Per-case detail ---
    lines.append("## Per-Case Results")
    lines.append("")
    per_case_times = timing.get("per_case", [])
    for i, r in enumerate(results):
        icon = "PASS" if r.passed else "FAIL"
        duration = (
            f" ({per_case_times[i]:.2f}s)" if i < len(per_case_times) else ""
        )
        lines.append(f"### Case {i + 1} — {icon}{duration}")
        lines.append("")
        lines.append(f"- **Input**: {r.input_data}")
        if getattr(r, "trajectory", None) is not None:
            _append_case_trajectory(lines, r)
        else:
            lines.append(f"- **Expected**: {r.expected}")
            lines.append(f"- **Actual**: {r.actual}")
        if r.error:
            lines.append(f"- **Error**: {r.error}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def export_benchmark_report(
    result,
    path: str | Path,
    yaml_path: str | None = None,
) -> None:
    """Write a benchmark report as a Markdown file.

    :param result: `BenchmarkResult` from ``find_best_model()``.
    :param path: Output file path.
    :param yaml_path: Optional agent config path for the header.
    """
    path = Path(path)
    lines: list[str] = []

    lines.append("# Benchmark Report")
    lines.append("")
    if yaml_path:
        lines.append(f"- **Config**: {yaml_path}")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"- **Date**: {now}")
    if result.best_model:
        size_gb = (result.best_model_size or 0) / 1_000_000_000
        lines.append(
            f"- **Best model**: {result.best_model} ({size_gb:.1f} GB)"
        )
    else:
        lines.append("- **Best model**: none passed")
    lines.append("")

    lines.append("## Model Scores")
    lines.append("")
    lines.append("| Model | Size | Passed | Pass rate | Avg time | Error |")
    lines.append("|---|---|---|---|---|---|")
    for s in result.scores:
        size_gb = s.model_size / 1_000_000_000
        status = "yes" if s.passed else "no"
        err = s.error or ""
        lines.append(
            f"| {s.model_name} | {size_gb:.1f} GB | {status} | "
            f"{s.pass_rate:.0%} | {s.avg_duration:.2f}s | {err} |"
        )
    lines.append("")

    for s in result.scores:
        if not s.case_results:
            continue
        lines.append(f"## {s.model_name} — Per-case Results")
        lines.append("")
        lines.append("| # | Run | Passed | Expected | Actual |")
        lines.append("|---|---|---|---|---|")
        for i, cr in enumerate(s.case_results, 1):
            status = "PASS" if cr.passed else "FAIL"
            expected = cr.test_case.expected
            actual = cr.actual if cr.actual is not None else "<none>"
            lines.append(
                f"| {i} | {cr.run} | {status} | {expected} | {actual} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _append_call_log(
    lines: list[str],
    summary: CallLogSummary,
) -> None:
    """Append a call-log section to the report.

    :param lines: Report lines to append to.
    :param summary: Summary from ``summarize_call_log()``.
    """
    o = summary.overall
    lines.append("## Call Log")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Calls | {o.calls} |")
    lines.append(f"| Succeeded | {o.successes} ({o.success_rate:.1%}) |")
    lines.append(f"| Failed | {o.failures} |")
    lines.append(f"| Completion tokens | {o.tokens} |")
    lines.append(f"| Prompt tokens | {o.prompt_tokens} |")
    lines.append(f"| Total duration | {o.total_duration:.2f}s |")
    lines.append(f"| Average duration | {o.avg_duration:.2f}s |")
    # "unpriced" rather than 0.0000c: no record carried a cost,
    # which is not the same as the run having been free.
    cost = f"{o.cost_cents:.4f}c" if o.cost_cents is not None else "unpriced"
    lines.append(f"| Cost | {cost} |")
    if o.cached_calls:
        lines.append(
            f"| Served from cache | {o.cached_calls} of {o.calls} "
            f"(tokens counted above, not billed) |"
        )
    lines.append("")

    if summary.errors:
        lines.append("### Errors")
        lines.append("")
        lines.append("| Count | Error |")
        lines.append("|-------|-------|")
        lines.extend(
            f"| {count} | {msg} |"
            for msg, count in sorted(
                summary.errors.items(), key=lambda kv: -kv[1]
            )
        )
        lines.append("")


def _append_trajectory(lines: list[str], trajectory: dict) -> None:
    """Append the trajectory section for a tool-agent run.

    :param lines: Report lines to append to.
    :param trajectory: The ``trajectory`` metrics dict.
    """
    expected = trajectory.get("expected_calls", 0)
    matched = trajectory.get("matched_calls", 0)
    rate = matched / expected if expected else 0.0

    lines.append("## Trajectory")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Match mode | {trajectory.get('mode', '')} |")
    lines.append(f"| Expected calls | {expected} |")
    lines.append(f"| Matched calls | {matched} ({rate:.1%}) |")
    lines.append(f"| Extra calls | {trajectory.get('extra_calls', 0)} |")
    lines.append(f"| Rejected calls | {trajectory.get('failed_calls', 0)} |")
    lines.append("")

    per_tool = trajectory.get("per_tool", {})
    if per_tool:
        lines.append("### Per-Tool Recall")
        lines.append("")
        lines.append("| Tool | Matched | Expected | Recall |")
        lines.append("|------|---------|----------|--------|")
        for name, stats in per_tool.items():
            want = stats["expected"]
            got = stats["matched"]
            recall = got / want if want else 0.0
            lines.append(f"| {name} | {got} | {want} | {recall:.1%} |")
        lines.append("")


def _append_case_trajectory(lines: list[str], result) -> None:
    """Append one case's trajectory detail.

    :param lines: Report lines to append to.
    :param result: An `EvalResult` carrying a `TrajectoryResult`.
    """
    t = result.trajectory
    expected = " -> ".join(e.format() for e in t.expected) or "<none>"
    actual = " -> ".join(c.format() for c in t.actual) or "<none>"
    lines.append(f"- **Expected calls**: {expected}")
    lines.append(f"- **Actual calls**: {actual}")
    if t.missing:
        lines.append(
            "- **Missing**: " + ", ".join(e.format() for e in t.missing)
        )
    if t.extra:
        lines.append("- **Extra**: " + ", ".join(c.format() for c in t.extra))
    if result.response is not None:
        lines.append(f"- **Answer**: {result.response.response}")


def _append_confusion_matrix(
    lines: list[str],
    results: list,
    field_name: str,
) -> None:
    """Append a confusion matrix section to the report.

    :param lines: Report lines to append to.
    :param results: List of `EvalResult`.
    :param field_name: The categorical field name.
    """
    from agents7.eval.metrics import build_confusion_matrix

    cm = build_confusion_matrix(results, field_name)
    if not cm["labels"]:
        return
    labels = cm["labels"]
    matrix = cm["matrix"]

    lines.append(f"## Confusion Matrix ({field_name})")
    lines.append("")
    header = "| Expected \\\\ Actual | " + " | ".join(labels) + " |"
    lines.append(header)
    sep = "|" + "|".join(["---"] * (len(labels) + 1)) + "|"
    lines.append(sep)
    lines.extend(
        f"| {label} | {' | '.join(str(v) for v in row)} |"
        for label, row in zip(labels, matrix)
    )
    lines.append("")
