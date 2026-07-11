"""Evaluation subpackage: metrics, export, and reporting."""

from agents7.eval.ab_test import (
    ABTestResult,
    run_ab_test,
    run_ab_test_async,
)
from agents7.eval.data import (
    AGENT_TYPE_MAP,
    build_agent,
    load_config_dict,
    load_eval_rows,
    model_output_fields,
    normalise_rows,
    resolve_agent_class,
)
from agents7.eval.export import (
    export_benchmark_report,
    export_csv,
    export_dataframe,
    export_eval_report,
    export_jsonl,
)
from agents7.eval.metrics import (
    EvalResult,
    build_confusion_matrix,
    compute_metrics,
    run_evaluation,
    run_evaluation_async,
)

__all__ = [
    "AGENT_TYPE_MAP",
    "ABTestResult",
    "EvalResult",
    "build_agent",
    "build_confusion_matrix",
    "compute_metrics",
    "export_benchmark_report",
    "export_csv",
    "export_dataframe",
    "export_eval_report",
    "export_jsonl",
    "load_config_dict",
    "load_eval_rows",
    "model_output_fields",
    "normalise_rows",
    "resolve_agent_class",
    "run_ab_test",
    "run_ab_test_async",
    "run_evaluation",
    "run_evaluation_async",
]
