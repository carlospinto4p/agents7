"""Compute evaluation metrics from EvalResult lists."""

from __future__ import annotations

from agents7.ui.evaluation.runner import EvalResult


def compute_metrics(
    results: list[EvalResult],
    expected_fields: list[str],
) -> dict:
    """Aggregate metrics from evaluation results.

    Returns:
        Dict with keys:
        - `total`: number of rows
        - `passed`: rows where all fields matched
        - `failed`: rows where at least one field did not
        - `accuracy`: passed / total
        - `per_field`: dict mapping field name to its accuracy
    """
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "accuracy": 0.0,
            "per_field": {},
        }

    passed = sum(1 for r in results if r.passed)

    per_field: dict[str, float] = {}
    for field_name in expected_fields:
        correct = sum(
            1 for r in results if r.field_results.get(field_name, False)
        )
        per_field[field_name] = correct / total

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": passed / total,
        "per_field": per_field,
    }


def build_confusion_matrix(
    results: list[EvalResult],
    field_name: str = "category",
) -> dict:
    """Build a confusion matrix for a categorical field.

    Returns:
        Dict with keys:
        - `labels`: sorted list of unique label strings
        - `matrix`: 2-D list `[actual_idx][predicted_idx]`
          of counts
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
