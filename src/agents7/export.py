"""Export agent responses to JSONL, CSV, DataFrame, and
evaluation reports."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

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
    if hasattr(val, "name"):
        return val.name
    return str(val)


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
        ``agents7.metrics.build_confusion_matrix``.
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

    # --- Confusion matrix ---
    if confusion_field:
        _append_confusion_matrix(lines, results, confusion_field)

    # --- Per-case detail ---
    lines.append("## Per-Case Results")
    lines.append("")
    for i, r in enumerate(results):
        icon = "PASS" if r.passed else "FAIL"
        lines.append(f"### Case {i + 1} — {icon}")
        lines.append("")
        lines.append(f"- **Input**: {r.input_data}")
        lines.append(f"- **Expected**: {r.expected}")
        lines.append(f"- **Actual**: {r.actual}")
        if r.error:
            lines.append(f"- **Error**: {r.error}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


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
    from agents7.metrics import build_confusion_matrix

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
    for label, row in zip(labels, matrix):
        vals = " | ".join(str(v) for v in row)
        lines.append(f"| {label} | {vals} |")
    lines.append("")
