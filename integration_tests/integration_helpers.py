"""Shared helpers for integration tests."""

import subprocess
import time

import ollama
from report_utils import write_eval_report_from_responses


def is_model_available(model_name: str) -> bool:
    """Check if an Ollama model is available locally.

    :param model_name: Model tag to look for (e.g.
        ``"qwen2.5:3b"``).
    :return: ``True`` if `ollama list` reports the model.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return model_name in result.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


_WARMUP_TIMEOUT = 300  # seconds, per model
_warmup_seconds: dict[str, float] = {}


def warm_model(model_name: str) -> None:
    """Load an Ollama model into memory, once per session.

    The first call into a model that is not loaded pays for
    reading it from disk, which can take longer than a whole
    test is allowed. Fixtures call this so the load happens in
    setup, which the per-test timeout does not cover.

    An empty prompt loads the model without generating:
    https://github.com/ollama/ollama/blob/main/docs/api.md#load-a-model
    (checked 2nd October 2026).

    :param model_name: Model tag to load (e.g. ``"qwen2.5:3b"``).
    """
    if model_name in _warmup_seconds:
        return
    start = time.monotonic()
    ollama.Client(timeout=_WARMUP_TIMEOUT).generate(
        model=model_name, prompt=""
    )
    _warmup_seconds[model_name] = time.monotonic() - start


def warmup_seconds() -> float:
    """Total time this session has spent loading models.

    :return: Seconds spent inside `warm_model`.
    """
    return sum(_warmup_seconds.values())


def run_single_case(
    agent,
    test_case: dict,
    field: str,
    expected_key: str,
) -> None:
    """Run a single act() call and assert one output field.

    :param agent: Any structured-output agent (Detector,
        Classifier, Regressor).
    :param test_case: Dict with ``"text"`` and the expected
        key (e.g. ``"expected_reason"``).
    :param field: Model attribute to check (e.g.
        ``"reason"`` or ``"category"``).
    :param expected_key: Key in *test_case* holding the
        expected enum name.
    """
    response = agent.act({"text": test_case["text"]})
    actual = getattr(response.model, field).name
    assert actual == test_case[expected_key], (
        f"Expected {field} '{test_case[expected_key]}' "
        f"but got '{actual}' for text: "
        f"'{test_case['text']}'"
    )


def run_batch_and_report(
    agent,
    test_data: list[dict],
    field: str,
    expected_key: str,
    reports_dir,
    report_name: str,
    **report_kwargs,
) -> None:
    """Run batch act, assert all items, and write a report.

    :param agent: Any structured-output agent.
    :param test_data: List of test-case dicts.
    :param field: Model attribute to check.
    :param expected_key: Key in each test case holding the
        expected enum name.
    :param reports_dir: Directory for the eval report.
    :param report_name: Report filename (e.g.
        ``"detector.md"``).
    :param report_kwargs: Extra kwargs forwarded to
        `write_eval_report_from_responses`.
    """
    input_list = [{"text": case["text"]} for case in test_data]
    responses = agent.act_in_batch(input_list)

    assert len(responses) == len(test_data), (
        f"Expected {len(test_data)} responses but got {len(responses)}"
    )

    for i, (case, output) in enumerate(
        zip(test_data, responses),
    ):
        actual = getattr(output.model, field).name
        assert case[expected_key] == actual, (
            f"Expected {field} '{case[expected_key]}' "
            f"but got '{actual}' for case {i}: "
            f"'{case['text']}'"
        )

    write_eval_report_from_responses(
        agent,
        responses,
        test_data,
        field,
        expected_key,
        reports_dir,
        report_name,
        **report_kwargs,
    )


def assert_not_applicable(agent, inputs: list[str]) -> None:
    """Assert that the agent classifies all inputs as
    NOT_APPLICABLE.

    Runs `agent.act()` on each input and checks that the
    resulting model has `reason.name == "NOT_APPLICABLE"`.

    :param agent: A Detector, Classifier, or Regressor agent.
    :param inputs: List of input texts that should be
        irrelevant to the agent's domain.
    """
    for text in inputs:
        result = agent.act({"text": text})
        assert result.model is not None, f"Model is None for input: {text}"
        assert result.model.reason.name == "NOT_APPLICABLE", (
            f"Expected NOT_APPLICABLE for '{text}', "
            f"got {result.model.reason.name}"
        )
