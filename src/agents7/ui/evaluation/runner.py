"""Evaluation runner — execute an agent against labeled data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel

from agents7.agents.labeler import Labeler
from agents7.models.base import AgentResponse


@dataclass
class EvalResult:
    """Result of evaluating a single row."""

    input_data: dict
    expected: dict
    actual: dict | None = None
    passed: bool = False
    field_results: dict[str, bool] = field(default_factory=dict)
    response: AgentResponse | None = None
    error: str | None = None


def run_evaluation(
    agent: Labeler,
    eval_data: list[dict],
    expected_fields: list[str],
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[EvalResult]:
    """Run *agent* against every row in *eval_data*.

    Each row must have an `"input"` key (the agent input) and
    keys matching *expected_fields* holding the expected outputs.

    Args:
        agent: Any Labeler-subclass agent instance.
        eval_data: List of dicts with `input` + expected fields.
        expected_fields: Field names to compare (e.g.
            `["category", "reason"]`).
        progress_callback: Optional `(current, total)` callable
            invoked after each row (for `st.progress`).

    Returns:
        List of :class:`EvalResult`, one per row.
    """
    results: list[EvalResult] = []
    total = len(eval_data)

    for idx, row in enumerate(eval_data):
        input_data = row.get("input", row)
        expected = {k: row[k] for k in expected_fields if k in row}

        result = EvalResult(input_data=input_data, expected=expected)

        try:
            response = agent.act(input_data)
            result.response = response

            if response.model is not None:
                actual = _extract_fields(response.model, expected_fields)
                result.actual = actual
                result.field_results = {
                    k: _compare_field(actual.get(k), expected.get(k))
                    for k in expected_fields
                }
                result.passed = all(result.field_results.values())
            else:
                result.error = "Model parsing failed"
        except Exception as e:
            result.error = str(e)

        results.append(result)

        if progress_callback is not None:
            progress_callback(idx + 1, total)

    return results


def _extract_fields(model: BaseModel, fields: list[str]) -> dict:
    """Pull field values from a model, using enum .name."""
    out: dict = {}
    for f in fields:
        val = getattr(model, f, None)
        if hasattr(val, "name"):
            out[f] = val.name
        else:
            out[f] = val
    return out


def _compare_field(actual, expected) -> bool:
    """Case-insensitive string comparison."""
    if actual is None or expected is None:
        return actual == expected
    return str(actual).upper() == str(expected).upper()
