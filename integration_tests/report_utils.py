"""Helpers for generating eval reports from test responses."""

from pathlib import Path

from agents7.export import export_eval_report
from agents7.eval.metrics import (
    EvalResult,
    _compare_field,
    _extract_fields,
    compute_metrics,
)


def write_eval_report_from_responses(
    agent,
    responses,
    test_data: list[dict],
    expected_field: str,
    expected_key: str,
    reports_dir: Path,
    filename: str,
    confusion_field: str | None = None,
    notes: list[str] | None = None,
) -> None:
    """Build an eval report from already-collected responses.

    Reuses LLM responses from batch tests — no extra LLM
    calls.

    :param agent: Agent instance (for name/model metadata).
    :param responses: List of AgentResponse from act_in_batch.
    :param test_data: Original test data list.
    :param expected_field: Model field to compare (e.g.
        ``"reason"``).
    :param expected_key: Key in test_data holding expected
        value (e.g. ``"expected_reason"``).
    :param reports_dir: Directory for the report file.
    :param filename: Report filename.
    :param confusion_field: Optional field for confusion
        matrix.
    :param notes: Optional list of note strings for the
        report.
    """
    results: list[EvalResult] = []
    for case, resp in zip(test_data, responses):
        expected = {expected_field: case[expected_key]}
        result = EvalResult(
            input_data={"text": case["text"]},
            expected=expected,
        )
        result.response = resp
        if resp.model is not None:
            actual = _extract_fields(resp.model, [expected_field])
            result.actual = actual
            result.field_results = {
                expected_field: _compare_field(
                    actual.get(expected_field),
                    expected.get(expected_field),
                )
            }
            result.passed = all(result.field_results.values())
        else:
            result.error = "Model parsing failed"
        results.append(result)

    metrics = compute_metrics(results, [expected_field])
    export_eval_report(
        results,
        metrics,
        reports_dir / filename,
        agent_name=agent.name,
        model_id=agent.engine.model_id,
        confusion_field=confusion_field,
        notes=notes,
    )
