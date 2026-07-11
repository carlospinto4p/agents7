"""A/B testing harness for comparing two agents.

Runs two agents over the **same** test cases and returns a
side-by-side comparison: accuracy deltas, per-field deltas,
timing deltas, pass/fail agreement, and a winner. Built on the
existing `run_evaluation` / `compute_metrics` primitives so both
agents are scored identically.
"""

from dataclasses import dataclass, field

from agents7.agents.labeler import Labeler
from agents7.eval.data import normalise_rows
from agents7.eval.metrics import (
    EvalResult,
    _compare_field,
    compute_metrics,
    run_evaluation,
    run_evaluation_async,
)


@dataclass
class ABTestResult:
    """Comparative result of running two agents on shared cases.

    Deltas are computed as *A minus B*, so a positive
    `accuracy_delta` means agent A scored higher.

    Attributes:
        label_a: Display label for agent A.
        label_b: Display label for agent B.
        expected_fields: Fields that were scored.
        metrics_a: `compute_metrics` dict for agent A.
        metrics_b: `compute_metrics` dict for agent B.
        results_a: Per-case `EvalResult` list for agent A.
        results_b: Per-case `EvalResult` list for agent B.
        accuracy_delta: ``accuracy_a - accuracy_b``.
        per_field_delta: Per-field ``accuracy_a - accuracy_b``.
        avg_time_delta: ``avg_s_a - avg_s_b`` in seconds.
        agreement: Pass/fail agreement counts and output
            agreement fraction (see `_build_agreement`).
        winner: ``label_a``, ``label_b``, or ``"tie"`` by
            overall accuracy.
    """

    label_a: str
    label_b: str
    expected_fields: list[str]
    metrics_a: dict
    metrics_b: dict
    results_a: list[EvalResult]
    results_b: list[EvalResult]
    accuracy_delta: float
    per_field_delta: dict[str, float] = field(default_factory=dict)
    avg_time_delta: float = 0.0
    agreement: dict = field(default_factory=dict)
    winner: str = "tie"


def _prepare(
    test_cases: list[dict],
    expected_fields: list[str] | None,
) -> tuple[list[dict], list[str]]:
    """Resolve the eval rows and fields to score.

    When *expected_fields* is None, the rows are normalised via
    `normalise_rows` (which also derives the fields); otherwise
    the rows are used as-is.

    :param test_cases: Raw or canonical eval rows.
    :param expected_fields: Explicit fields to score, or None to
        infer them.
    :return: ``(eval_rows, expected_fields)``.
    """
    if expected_fields is None:
        return normalise_rows(test_cases)
    return test_cases, expected_fields


def _build_agreement(
    results_a: list[EvalResult],
    results_b: list[EvalResult],
    expected_fields: list[str],
) -> dict:
    """Compute per-case pass/fail and output agreement.

    Cases are aligned by index (both agents ran the same rows in
    the same order). ``output_agreement`` is the fraction of
    cases where both agents produced the same value on every
    expected field, regardless of correctness.

    :param results_a: Agent A results.
    :param results_b: Agent B results.
    :param expected_fields: Fields to compare outputs on.
    :return: Dict with ``both_passed``, ``both_failed``,
        ``a_only``, ``b_only``, and ``output_agreement``.
    """
    both_passed = both_failed = a_only = b_only = 0
    same_output = 0
    total = len(results_a)

    for ra, rb in zip(results_a, results_b):
        if ra.passed and rb.passed:
            both_passed += 1
        elif ra.passed:
            a_only += 1
        elif rb.passed:
            b_only += 1
        else:
            both_failed += 1

        actual_a = ra.actual or {}
        actual_b = rb.actual or {}
        if all(
            _compare_field(actual_a.get(f), actual_b.get(f))
            for f in expected_fields
        ):
            same_output += 1

    return {
        "both_passed": both_passed,
        "both_failed": both_failed,
        "a_only": a_only,
        "b_only": b_only,
        "output_agreement": (same_output / total) if total else 0.0,
    }


def _build_ab_result(
    label_a: str,
    label_b: str,
    expected_fields: list[str],
    results_a: list[EvalResult],
    results_b: list[EvalResult],
) -> ABTestResult:
    """Assemble an `ABTestResult` from both agents' results.

    :param label_a: Label for agent A.
    :param label_b: Label for agent B.
    :param expected_fields: Fields that were scored.
    :param results_a: Agent A results.
    :param results_b: Agent B results.
    :return: Populated `ABTestResult`.
    """
    metrics_a = compute_metrics(results_a, expected_fields)
    metrics_b = compute_metrics(results_b, expected_fields)

    per_field_a = metrics_a["per_field"]
    per_field_b = metrics_b["per_field"]
    per_field_delta = {
        f: per_field_a.get(f, 0.0) - per_field_b.get(f, 0.0)
        for f in expected_fields
    }

    accuracy_delta = metrics_a["accuracy"] - metrics_b["accuracy"]
    if accuracy_delta > 0:
        winner = label_a
    elif accuracy_delta < 0:
        winner = label_b
    else:
        winner = "tie"

    return ABTestResult(
        label_a=label_a,
        label_b=label_b,
        expected_fields=expected_fields,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        results_a=results_a,
        results_b=results_b,
        accuracy_delta=accuracy_delta,
        per_field_delta=per_field_delta,
        avg_time_delta=(
            metrics_a["timing"]["avg_s"] - metrics_b["timing"]["avg_s"]
        ),
        agreement=_build_agreement(results_a, results_b, expected_fields),
        winner=winner,
    )


def run_ab_test(
    agent_a: Labeler,
    agent_b: Labeler,
    test_cases: list[dict],
    expected_fields: list[str] | None = None,
    label_a: str = "A",
    label_b: str = "B",
) -> ABTestResult:
    """Run two agents on the same *test_cases* and compare them.

    Both agents are evaluated with `run_evaluation`, so scoring
    is identical. Deltas are reported as *A minus B*.

    :param agent_a: First agent (any `Labeler` subclass).
    :param agent_b: Second agent to compare against.
    :param test_cases: Eval rows — each a dict with an
        ``"input"`` key plus expected-field keys. Flat rows are
        accepted and normalised when *expected_fields* is None.
    :param expected_fields: Fields to score. When None, they are
        inferred (and the rows normalised) via `normalise_rows`.
    :param label_a: Display label for agent A.
    :param label_b: Display label for agent B.
    :return: An `ABTestResult` with comparative metrics.
    """
    rows, fields = _prepare(test_cases, expected_fields)
    results_a = run_evaluation(agent_a, rows, fields)
    results_b = run_evaluation(agent_b, rows, fields)
    return _build_ab_result(label_a, label_b, fields, results_a, results_b)


async def run_ab_test_async(
    agent_a: Labeler,
    agent_b: Labeler,
    test_cases: list[dict],
    expected_fields: list[str] | None = None,
    label_a: str = "A",
    label_b: str = "B",
) -> ABTestResult:
    """Async version of `run_ab_test`.

    Uses `run_evaluation_async` for each agent. The two agents
    are still evaluated sequentially (A then B) so their timing
    metrics are not skewed by shared concurrency.

    :param agent_a: First agent (any `Labeler` subclass).
    :param agent_b: Second agent to compare against.
    :param test_cases: Eval rows (see `run_ab_test`).
    :param expected_fields: Fields to score, or None to infer.
    :param label_a: Display label for agent A.
    :param label_b: Display label for agent B.
    :return: An `ABTestResult` with comparative metrics.
    """
    rows, fields = _prepare(test_cases, expected_fields)
    results_a = await run_evaluation_async(agent_a, rows, fields)
    results_b = await run_evaluation_async(agent_b, rows, fields)
    return _build_ab_result(label_a, label_b, fields, results_a, results_b)
