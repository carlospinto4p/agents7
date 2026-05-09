"""Backward-compatible re-export. Canonical: `agents7.eval.export`."""

import warnings

warnings.warn(
    "agents7.export is deprecated; import from agents7.eval.export instead.",
    DeprecationWarning,
    stacklevel=2,
)

from agents7.eval.export import (  # noqa: E402
    export_benchmark_report,
    export_csv,
    export_dataframe,
    export_eval_report,
    export_jsonl,
)

__all__ = [
    "export_benchmark_report",
    "export_csv",
    "export_dataframe",
    "export_eval_report",
    "export_jsonl",
]
