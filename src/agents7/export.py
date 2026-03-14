"""Export agent responses to JSONL and CSV files."""

from __future__ import annotations

import csv
import json
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
