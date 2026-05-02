"""Backward-compatible re-export. Canonical: `agents7.eval.metrics`."""

from agents7.eval.metrics import (
    EvalResult,
    _compare_field,
    _extract_fields,
    _resolve_enum,
    build_confusion_matrix,
    compute_metrics,
    run_evaluation,
    run_evaluation_async,
)

__all__ = [
    "EvalResult",
    "_compare_field",
    "_extract_fields",
    "_resolve_enum",
    "build_confusion_matrix",
    "compute_metrics",
    "run_evaluation",
    "run_evaluation_async",
]
