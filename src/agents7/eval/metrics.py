"""Agent evaluation metrics.

Provides evaluation primitives usable without Streamlit:
`EvalResult`, `run_evaluation`, `run_evaluation_async`,
`compute_metrics`, and `build_confusion_matrix`.
"""

from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel

from agents7.agents.labeler import Labeler
from agents7.logs import logger
from agents7.models.base import AgentResponse


def _row_input(row: dict) -> dict:
    """Return the agent input from an eval row.

    Rows may embed the input under an ``"input"`` key (so the
    same dict can carry both input and expected fields) or be
    the input dict directly.
    """
    return row.get("input", row)


@dataclass
class EvalResult:
    """Result of evaluating a single row.

    Attributes:
        input_data: The input sent to the agent.
        expected: Dict of expected field values.
        actual: Dict of actual field values extracted
            from the agent response, or None on failure.
        passed: True when all expected fields matched.
        field_results: Per-field match results.
        response: The full `AgentResponse`, or None.
        error: Error message string, or None.
    """

    input_data: dict
    expected: dict
    actual: dict | None = None
    passed: bool = False
    field_results: dict[str, bool] = field(default_factory=dict)
    response: AgentResponse | None = None
    error: str | None = None


def _build_eval_result(
    row: dict,
    expected_fields: list[str],
    response: AgentResponse | None,
    error: str | None = None,
) -> EvalResult:
    """Build an `EvalResult` from a row and agent response.

    :param row: Original eval data row.
    :param expected_fields: Fields to compare.
    :param response: Agent response, or None on error.
    :param error: Error message, or None.
    :return: Populated `EvalResult`.
    """
    input_data = _row_input(row)
    expected = {k: row[k] for k in expected_fields if k in row}
    result = EvalResult(input_data=input_data, expected=expected)

    if error is not None:
        result.error = error
        return result

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

    return result


def run_evaluation(
    agent: Labeler,
    eval_data: list[dict],
    expected_fields: list[str],
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[EvalResult]:
    """Run *agent* against every row in *eval_data* using
    batch processing.

    Each row must have an ``"input"`` key (the agent input)
    and keys matching *expected_fields* holding the expected
    outputs.

    Uses ``act_in_batch()`` for efficient processing. Rows
    that cause exceptions are individually retried with
    ``act()`` so a single failure doesn't break the batch.

    :param agent: Any `Labeler`-subclass agent instance.
    :param eval_data: List of dicts with ``input`` + expected
        fields.
    :param expected_fields: Field names to compare (e.g.
        ``["category", "reason"]``).
    :param progress_callback: Optional ``(current, total)``
        callable invoked after the batch completes.
    :return: List of `EvalResult`, one per row.
    """
    inputs = [_row_input(row) for row in eval_data]
    total = len(eval_data)

    try:
        responses = agent.act_in_batch(inputs, iter_func=None)
        results = [
            _build_eval_result(row, expected_fields, resp)
            for row, resp in zip(eval_data, responses)
        ]
    except Exception as e:
        logger.warning(
            "Batch eval failed, falling back to sequential: %s",
            e,
            exc_info=True,
        )
        results = _fallback_sequential(agent, eval_data, expected_fields)

    if progress_callback is not None:
        progress_callback(total, total)

    return results


async def run_evaluation_async(
    agent: Labeler,
    eval_data: list[dict],
    expected_fields: list[str],
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[EvalResult]:
    """Async version of `run_evaluation`.

    Uses ``act_in_batch_async()`` for concurrent processing.

    :param agent: Any `Labeler`-subclass agent instance.
    :param eval_data: List of dicts with ``input`` + expected
        fields.
    :param expected_fields: Field names to compare (e.g.
        ``["category", "reason"]``).
    :param progress_callback: Optional ``(current, total)``
        callable invoked after the batch completes.
    :return: List of `EvalResult`, one per row.
    """
    inputs = [_row_input(row) for row in eval_data]
    total = len(eval_data)

    try:
        responses = await agent.act_in_batch_async(inputs, with_tqdm=False)
        results = [
            _build_eval_result(row, expected_fields, resp)
            for row, resp in zip(eval_data, responses)
        ]
    except Exception as e:
        logger.warning(
            "Async batch eval failed, falling back to sequential: %s",
            e,
            exc_info=True,
        )
        results = _fallback_sequential(agent, eval_data, expected_fields)

    if progress_callback is not None:
        progress_callback(total, total)

    return results


def _fallback_sequential(
    agent: Labeler,
    eval_data: list[dict],
    expected_fields: list[str],
) -> list[EvalResult]:
    """Fall back to sequential ``act()`` calls when batch
    fails.

    :param agent: Agent instance.
    :param eval_data: Original eval data rows.
    :param expected_fields: Fields to compare.
    :return: List of `EvalResult`, one per row.
    """
    results: list[EvalResult] = []
    for row in eval_data:
        input_data = _row_input(row)
        try:
            response = agent.act(input_data)
            results.append(_build_eval_result(row, expected_fields, response))
        except Exception as e:
            results.append(
                _build_eval_result(row, expected_fields, None, error=str(e))
            )
    return results


def compute_metrics(
    results: list[EvalResult],
    expected_fields: list[str],
) -> dict:
    """Aggregate metrics from evaluation results.

    :param results: List of `EvalResult` from
        `run_evaluation`.
    :param expected_fields: Field names that were evaluated.
    :return: Dict with keys ``total``, ``passed``, ``failed``,
        ``accuracy``, and ``per_field`` (field → accuracy).
    """
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "accuracy": 0.0,
            "per_field": {},
            "timing": {
                "total_s": 0.0,
                "avg_s": 0.0,
                "min_s": 0.0,
                "max_s": 0.0,
                "per_case": [],
            },
        }

    passed = sum(1 for r in results if r.passed)

    per_field: dict[str, float] = {}
    for field_name in expected_fields:
        correct = sum(
            1 for r in results if r.field_results.get(field_name, False)
        )
        per_field[field_name] = correct / total

    timing = _compute_timing(results)

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": passed / total,
        "per_field": per_field,
        "timing": timing,
    }


def _compute_timing(results: list[EvalResult]) -> dict:
    """Extract timing statistics from evaluation results.

    :param results: List of `EvalResult`.
    :return: Dict with ``total_s``, ``avg_s``, ``min_s``,
        ``max_s``, and ``per_case`` (list of durations).
    """
    durations: list[float] = []
    for r in results:
        if r.response is not None and hasattr(r.response, "metrics"):
            durations.append(r.response.metrics.total_duration)
        else:
            durations.append(0.0)

    total_s = sum(durations)
    return {
        "total_s": total_s,
        "avg_s": total_s / len(durations) if durations else 0.0,
        "min_s": min(durations) if durations else 0.0,
        "max_s": max(durations) if durations else 0.0,
        "per_case": durations,
    }


def build_confusion_matrix(
    results: list[EvalResult],
    field_name: str = "category",
) -> dict:
    """Build a confusion matrix for a categorical field.

    :param results: List of `EvalResult` from
        `run_evaluation`.
    :param field_name: The categorical field to build the
        matrix for. Defaults to ``"category"``.
    :return: Dict with keys ``labels`` (sorted list of unique
        label strings) and ``matrix`` (2-D list
        ``[expected_idx][actual_idx]`` of counts).
    """
    pairs: list[tuple[str, str]] = []
    label_set: set[str] = set()

    for r in results:
        expected_val = r.expected.get(field_name)
        actual_val = (
            r.actual.get(field_name) if r.actual is not None else None
        )
        if expected_val is None:
            continue
        exp = str(expected_val).upper()
        act = str(actual_val).upper() if actual_val is not None else "_NONE_"
        label_set.add(exp)
        label_set.add(act)
        pairs.append((exp, act))

    labels = sorted(label_set)
    label_idx = {lbl: i for i, lbl in enumerate(labels)}
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]

    for exp, act in pairs:
        matrix[label_idx[exp]][label_idx[act]] += 1

    return {"labels": labels, "matrix": matrix}


def _resolve_enum(val: object) -> object:
    """Return ``val.name`` for enum members, otherwise ``val``."""
    return val.name if hasattr(val, "name") else val


def _extract_fields(model: BaseModel, fields: list[str]) -> dict:
    """Pull field values from a model, using enum ``.name``."""
    return {f: _resolve_enum(getattr(model, f, None)) for f in fields}


def _compare_field(actual: object, expected: object) -> bool:
    """Case-insensitive string comparison."""
    if actual is None or expected is None:
        return actual == expected
    return str(actual).upper() == str(expected).upper()
