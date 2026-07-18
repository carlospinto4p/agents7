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
from agents7.eval.suite import (
    EvalRun,
    resolve_eval_fields,
    run_eval_suite,
)
from agents7.eval.trajectory import (
    TRAJECTORY_FIELD,
    ExpectedCall,
    ToolCall,
    TrajectoryMode,
    TrajectoryResult,
    collect_trajectory,
    compute_trajectory_metrics,
    match_trajectory,
    parse_expected_calls,
    run_trajectory_evaluation,
)

__all__ = [
    "AGENT_TYPE_MAP",
    "TRAJECTORY_FIELD",
    "ABTestResult",
    "EvalResult",
    "EvalRun",
    "ExpectedCall",
    "ToolCall",
    "TrajectoryMode",
    "TrajectoryResult",
    "build_agent",
    "build_confusion_matrix",
    "collect_trajectory",
    "compute_metrics",
    "compute_trajectory_metrics",
    "export_benchmark_report",
    "export_csv",
    "export_dataframe",
    "export_eval_report",
    "export_jsonl",
    "load_config_dict",
    "load_eval_rows",
    "match_trajectory",
    "model_output_fields",
    "normalise_rows",
    "parse_expected_calls",
    "resolve_agent_class",
    "resolve_eval_fields",
    "run_ab_test",
    "run_ab_test_async",
    "run_eval_suite",
    "run_evaluation",
    "run_evaluation_async",
    "run_trajectory_evaluation",
]
