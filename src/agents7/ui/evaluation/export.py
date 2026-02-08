"""Export evaluation results to CSV and JSONL formats."""

from __future__ import annotations

import csv
import io
import json

from agents7.ui.evaluation.runner import EvalResult


def format_results_csv(
    results: list[EvalResult],
    expected_fields: list[str],
) -> str:
    """Format evaluation results as a CSV string.

    Produces flat columns:
    ``input | {field}_expected | {field}_actual | {field}_passed |
    ... | passed | error``

    Args:
        results: List of `EvalResult` from `run_evaluation`.
        expected_fields: Field names that were evaluated
            (e.g. ``["category", "reason"]``).

    Returns:
        CSV-formatted string (including header row).
    """
    fieldnames = ["input"]
    for f in expected_fields:
        fieldnames.extend([f"{f}_expected", f"{f}_actual", f"{f}_passed"])
    fieldnames.extend(["passed", "error"])

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    for r in results:
        row: dict[str, str] = {
            "input": _input_to_str(r.input_data),
            "passed": str(r.passed),
            "error": r.error or "",
        }
        for f in expected_fields:
            row[f"{f}_expected"] = str(r.expected.get(f, ""))
            actual_val = r.actual.get(f) if r.actual is not None else None
            row[f"{f}_actual"] = (
                str(actual_val) if actual_val is not None else ""
            )
            row[f"{f}_passed"] = str(r.field_results.get(f, False))
        writer.writerow(row)

    return buf.getvalue()


def format_results_jsonl(results: list[EvalResult]) -> str:
    """Format evaluation results as a JSONL string.

    One JSON object per line with keys: ``input``, ``expected``,
    ``actual``, ``field_results``, ``passed``, ``error``.
    The heavy ``response`` object is excluded.

    Args:
        results: List of `EvalResult` from `run_evaluation`.

    Returns:
        JSONL-formatted string (one JSON object per line).
    """
    lines: list[str] = []
    for r in results:
        obj = {
            "input": r.input_data,
            "expected": r.expected,
            "actual": r.actual,
            "field_results": r.field_results,
            "passed": r.passed,
            "error": r.error,
        }
        lines.append(json.dumps(obj, ensure_ascii=False))
    return "\n".join(lines)


def _input_to_str(input_data: dict) -> str:
    """Stringify input data for CSV display.

    If the dict has a single ``"text"`` key, return its value
    directly; otherwise return the string representation.
    """
    if isinstance(input_data, dict) and list(input_data.keys()) == ["text"]:
        return str(input_data["text"])
    return str(input_data)
