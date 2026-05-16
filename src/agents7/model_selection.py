"""Automatic model selection for structured extraction tasks.

Given an agent configuration and a set of test cases with expected
outputs, find the smallest Ollama model that passes all cases
without losing quality.

Usage::

    from agents7.model_selection import find_best_model, TestCase

    cases = [
        TestCase(
            input={"text": "It is -10 degrees."},
            expected={"reason": "TOO_COLD"},
        ),
        TestCase(
            input={"text": "She bought a book."},
            expected={"reason": "NOT_APPLICABLE"},
        ),
    ]
    result = find_best_model(
        "examples/configs/demo_temperature_detector.yml",
        test_cases=cases,
    )
    print(result.best_model)        # e.g. "qwen2.5:3b"
    print(result.best_model_size)   # e.g. 1910000000
"""

import types
from dataclasses import dataclass, field
from enum import StrEnum

import ollama as ollama_client
from pydantic import BaseModel

from agents7.agents import Classifier, Detector, Labeler, Regressor
from agents7.agents.config_loader import load_yaml_settings
from agents7.exceptions import Agents7Exception
from agents7.llm import LLMEngine
from agents7.logs import logger
from agents7.models.base import ClassificationProperty, Property


# --- Data classes ---


@dataclass(frozen=True)
class TestCase:
    """A single test case for model benchmarking.

    Attributes:
        input (dict): Input data passed to the agent's
            ``act()`` method.
        expected (dict): Expected output fields. Only the
            keys present in this dict are checked against
            the agent's output. Enum fields are compared
            by name (string), regular fields by value.
    """

    __test__ = False  # prevent pytest collection

    input: dict
    expected: dict


@dataclass(frozen=True)
class CaseResult:
    """Result of running a single test case.

    Attributes:
        test_case (TestCase): The test case that was run.
        passed (bool): Whether the output matched expected.
        actual (dict | None): The actual output fields, or
            None if the agent returned no model.
        run (int): The run number (1-based).
    """

    test_case: TestCase
    passed: bool
    actual: dict | None
    run: int


@dataclass(frozen=True)
class ModelScore:
    """Benchmark result for a single model.

    Attributes:
        model_name (str): The Ollama model name.
        model_size (int): Model size in bytes (from Ollama).
        passed (bool): Whether all test cases passed on
            every run.
        pass_rate (float): Fraction of case-runs that
            passed (0.0 to 1.0).
        case_results (list[CaseResult]): Detailed results
            for each test case and run.
        avg_duration (float): Average response time in
            seconds across all calls.
        error (str | None): Error message if the model
            could not be tested (e.g. Ollama error).
    """

    model_name: str
    model_size: int
    passed: bool
    pass_rate: float
    case_results: list[CaseResult] = field(default_factory=list)
    avg_duration: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    """Full benchmark result across all tested models.

    Attributes:
        best_model (str | None): Name of the smallest model
            that passed all test cases, or None if no model
            passed.
        best_model_size (int | None): Size of the best model
            in bytes, or None.
        scores (list[ModelScore]): Results for every tested
            model, sorted by size ascending.
    """

    best_model: str | None
    best_model_size: int | None
    scores: list[ModelScore]


class ModelSelectionError(Agents7Exception):
    """Raised when model selection fails."""


# --- Helpers ---


def _compare_field(actual_val: object, expected_val: object) -> bool:
    """Compare an actual output field to the expected value.

    Enum fields are compared by name; None/null fields are
    compared directly; other fields use equality.
    """
    if isinstance(actual_val, StrEnum):
        return actual_val.name == str(expected_val)
    if actual_val is None and expected_val is None:
        return True
    if actual_val is None or expected_val is None:
        return False
    return actual_val == expected_val


def _compare_output(
    model: BaseModel | None,
    expected: dict,
) -> tuple[bool, dict | None]:
    """Compare an agent output model against expected fields.

    Returns:
        Tuple of (all_matched, actual_fields_dict).
    """
    if model is None:
        return (not expected, None)

    actual = {}
    all_match = True
    for key, expected_val in expected.items():
        if not hasattr(model, key):
            actual[key] = "<missing>"
            all_match = False
            continue
        actual_val = getattr(model, key)
        actual[key] = (
            actual_val.name if isinstance(actual_val, StrEnum) else actual_val
        )
        if not _compare_field(actual_val, expected_val):
            all_match = False

    return all_match, actual


def _list_ollama_models() -> list[dict]:
    """List locally available Ollama models with sizes.

    Returns a list of dicts with keys ``name`` and ``size``.
    Sorted by size ascending.
    """
    response = ollama_client.list()
    models = []
    for m in response.models:
        models.append(
            {
                "name": m.model,
                "size": m.size,
            }
        )
    models.sort(key=lambda m: m["size"])
    return models


# --- Main function ---


def find_best_model(
    yaml_path: str,
    test_cases: list[TestCase],
    *,
    models: list[str] | None = None,
    runs: int = 1,
    include_batch: bool = False,
    stop_on_first: bool = True,
    report_path: str | None = None,
) -> BenchmarkResult:
    """Find the smallest model that passes all test cases.

    Loads the agent configuration from a YAML file, then
    iterates through candidate models (smallest first) and
    runs the test cases against each. Returns as soon as a
    model passes all cases (unless ``stop_on_first=False``).

    :param yaml_path: Path to the agent's YAML config file.
    :param test_cases: List of ``TestCase`` instances. Each
        must include ``input`` (dict passed to ``act()``)
        and ``expected`` (dict of fields to check).
    :param models: Optional list of model names to test. If
        None, all locally available Ollama models are used.
    :param runs: Number of times to run each test case per
        model (for reliability). Defaults to 1.
    :param include_batch: If True, also test ``act_in_batch``
        with all test cases as a batch. A model must pass
        both single and batch modes. Defaults to False.
    :param stop_on_first: If True, stop after the first
        model that passes all cases. Defaults to True.
    :param report_path: If set, write a Markdown benchmark
        report to this path once benchmarking completes.
    :return: A ``BenchmarkResult`` with the best model and
        scores for all tested models.
    :raises ModelSelectionError: If no test cases are
        provided or YAML config is invalid.
    """
    if not test_cases:
        raise ModelSelectionError("At least one test case is required.")

    config = load_yaml_settings(yaml_path)
    candidates = _resolve_candidates(models)

    if not candidates:
        raise ModelSelectionError(
            "No Ollama models available for benchmarking."
        )

    logger.info(
        f"Model selection: testing {len(candidates)} "
        f"model(s) with {len(test_cases)} case(s), "
        f"{runs} run(s) each."
    )

    scores: list[ModelScore] = []
    best_model = None
    best_model_size = None

    for candidate in candidates:
        model_name = candidate["name"]
        model_size = candidate["size"]
        logger.info(
            f"  Testing model: {model_name} ({_format_size(model_size)})"
        )

        score = _benchmark_model(
            config,
            model_name,
            model_size,
            test_cases,
            runs=runs,
            include_batch=include_batch,
        )
        scores.append(score)

        if score.passed:
            logger.info(
                f"  -> PASSED ({score.pass_rate:.0%}, "
                f"avg {score.avg_duration:.2f}s)"
            )
            if best_model is None:
                best_model = model_name
                best_model_size = model_size
                if stop_on_first:
                    break
        else:
            logger.info(
                f"  -> FAILED ({score.pass_rate:.0%}"
                f"{', ' + score.error if score.error else ''})"
            )

    result = BenchmarkResult(
        best_model=best_model,
        best_model_size=best_model_size,
        scores=scores,
    )

    if report_path is not None:
        from agents7.eval.export import export_benchmark_report

        export_benchmark_report(result, report_path, yaml_path=yaml_path)
        logger.info(f"Benchmark report written to {report_path}")

    return result


def _resolve_candidates(
    models: list[str] | None,
) -> list[dict]:
    """Resolve candidate models with sizes.

    If ``models`` is given, filter the local list to only
    those names (preserving size-based ordering). Otherwise
    return all local models sorted by size.
    """
    available = _list_ollama_models()
    if models is None:
        return available

    available_by_name = {m["name"]: m for m in available}
    candidates = []
    for name in models:
        matched = available_by_name.get(name)
        if matched is None:
            # Try with :latest suffix
            matched = available_by_name.get(f"{name}:latest")
        if matched is not None:
            candidates.append(matched)
        else:
            logger.warning(f"Model '{name}' not found locally, skipping.")
    candidates.sort(key=lambda m: m["size"])
    return candidates


def _benchmark_model(
    config: dict,
    model_name: str,
    model_size: int,
    test_cases: list[TestCase],
    *,
    runs: int,
    include_batch: bool,
) -> ModelScore:
    """Benchmark a single model against all test cases."""
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")
    try:
        engine = LLMEngine(provider="ollama", model_name=model_name)
    except Exception as e:
        logger.debug(f"  Engine init failed for {model_name}: {e}")
        return ModelScore(
            model_name=model_name,
            model_size=model_size,
            passed=False,
            pass_rate=0.0,
            error=f"Engine init failed: {e}",
        )

    # Replace the engine in config
    agent_config = {**config, "engine": engine}

    try:
        agent = _create_agent(agent_config)
    except Exception as e:
        logger.debug(f"  Agent creation failed for {model_name}: {e}")
        return ModelScore(
            model_name=model_name,
            model_size=model_size,
            passed=False,
            pass_rate=0.0,
            error=f"Agent creation failed: {e}",
        )

    case_results: list[CaseResult] = []
    durations: list[float] = []

    # Single-mode tests
    for run_num in range(1, runs + 1):
        for tc in test_cases:
            result, duration = _run_single_case(agent, tc, run_num)
            case_results.append(result)
            durations.append(duration)

    # Batch-mode tests
    if include_batch:
        batch_results, batch_duration = _run_batch_cases(
            agent, test_cases, runs
        )
        case_results.extend(batch_results)
        durations.append(batch_duration)

    total = len(case_results)
    passed_count = sum(1 for r in case_results if r.passed)
    pass_rate = passed_count / total if total > 0 else 0.0
    avg_duration = sum(durations) / len(durations) if durations else 0.0

    return ModelScore(
        model_name=model_name,
        model_size=model_size,
        passed=(passed_count == total),
        pass_rate=pass_rate,
        case_results=case_results,
        avg_duration=avg_duration,
    )


def _create_agent(config: dict):
    """Create the appropriate agent type from config."""
    output_model = config["output_model"]

    if issubclass(output_model, ClassificationProperty):
        return Classifier(**config)
    if issubclass(output_model, Property):
        value_type = output_model.model_fields.get("value")
        if value_type and hasattr(value_type, "annotation"):
            ann = value_type.annotation
            origin = getattr(ann, "__origin__", None)
            if origin is types.UnionType:
                args = [a for a in ann.__args__ if a is not type(None)]
                if args and args[0] in (float, int):
                    return Regressor(**config)
        return Detector(**config)
    return Labeler(**config)


def _run_single_case(
    agent,
    tc: TestCase,
    run_num: int,
) -> tuple[CaseResult, float]:
    """Run a single test case and return result + duration."""
    try:
        response = agent.act(tc.input)
        duration = response.metrics.total_duration
        passed, actual = _compare_output(response.model, tc.expected)
        return (
            CaseResult(
                test_case=tc,
                passed=passed,
                actual=actual,
                run=run_num,
            ),
            duration,
        )
    except Exception as e:
        logger.debug(f"    Case failed with error: {e}")
        return (
            CaseResult(
                test_case=tc,
                passed=False,
                actual=None,
                run=run_num,
            ),
            0.0,
        )


def _run_batch_cases(
    agent,
    test_cases: list[TestCase],
    runs: int,
) -> tuple[list[CaseResult], float]:
    """Run all test cases as a batch and return results."""
    results: list[CaseResult] = []
    total_duration = 0.0

    for run_num in range(1, runs + 1):
        inputs = [tc.input for tc in test_cases]
        try:
            responses = agent.act_in_batch(inputs, iter_func=None)
            for i, (response, tc) in enumerate(zip(responses, test_cases)):
                total_duration += response.metrics.total_duration
                passed, actual = _compare_output(response.model, tc.expected)
                results.append(
                    CaseResult(
                        test_case=tc,
                        passed=passed,
                        actual=actual,
                        run=run_num,
                    )
                )
        except Exception as e:
            logger.debug(f"    Batch failed: {e}")
            for tc in test_cases:
                results.append(
                    CaseResult(
                        test_case=tc,
                        passed=False,
                        actual=None,
                        run=run_num,
                    )
                )

    return results, total_duration


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.1f} GB"
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    return f"{size_bytes:,} B"
