"""Testing utilities for agents7.

Provides helpers for unit-testing agents without LLM calls
and for asserting evaluation accuracy thresholds.

Usage::

    from agents7.testing import (
        create_mock_engine,
        assert_accuracy_above,
    )

    # Unit test without LLM
    engine = create_mock_engine({"text": "hello"}: '{"value": "hi"}')
    agent = Labeler(..., engine=engine)
    response = agent.act({"text": "hello"})

    # Evaluate and assert accuracy
    assert_accuracy_above(agent, eval_data, ["category"], 0.9)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from agents7.llm.base import LLMEngine
from agents7.metrics import compute_metrics, run_evaluation


_DEFAULT_METRICS = {
    "tokens": 10,
    "prompt_tokens": 5,
    "total_duration": 0.1,
}


def create_mock_engine(
    responses: dict[str, str] | str | None = None,
    *,
    model_name: str = "mock",
) -> LLMEngine:
    """Create a mock LLM engine for unit testing.

    Returns an ``LLMEngine`` whose ``chat()`` and
    ``chat_async()`` return predefined responses without
    calling any LLM provider.

    :param responses: Controls what the mock returns:
        - ``dict``: Maps input text to JSON response
          strings. The mock inspects the user message
          content and returns the matching response.
        - ``str``: A single JSON string returned for
          every input.
        - ``None``: Returns ``'{"value": "mock"}'``
          for every input.
    :param model_name: Model name for the engine.
        Defaults to ``"mock"``.
    :return: A configured ``LLMEngine`` with a mocked
        provider.

    Usage::

        engine = create_mock_engine({
            "hello": '{"sentiment": "POSITIVE"}',
            "bad day": '{"sentiment": "NEGATIVE"}',
        })

        engine = create_mock_engine('{"value": "ok"}')
    """
    default_response = '{"value": "mock"}'
    if isinstance(responses, str):
        default_response = responses
        responses = None

    def _make_response(response_str: str) -> dict:
        return {
            "role": "assistant",
            "response": response_str,
            "thinking": None,
            "metrics": dict(_DEFAULT_METRICS),
        }

    def _chat_side_effect(**kwargs: Any) -> dict:
        messages = kwargs.get("messages", [])
        user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        if responses is not None:
            for key, val in responses.items():
                if key in user_content:
                    return _make_response(val)

        return _make_response(default_response)

    async def _chat_async_side_effect(**kwargs: Any) -> dict:
        return _chat_side_effect(**kwargs)

    engine = object.__new__(LLMEngine)
    engine.provider = "mock"
    engine.model_name = model_name
    engine.model_id = f"mock::{model_name}"
    engine.retry_config = None
    engine.cost_per_1m_tokens = None
    engine._cache = None
    engine._fallback_provider = None

    provider = Mock()
    provider.chat.side_effect = _chat_side_effect
    provider.chat_async = _chat_async_side_effect
    engine.llm_provider = provider

    return engine


def assert_accuracy_above(
    agent,
    eval_data: list[dict],
    expected_fields: list[str],
    min_accuracy: float = 0.9,
) -> dict:
    """Run evaluation and assert accuracy meets threshold.

    Runs ``run_evaluation()`` on the given agent and data,
    then asserts that overall accuracy is at or above
    ``min_accuracy``.

    :param agent: Any ``Labeler``-subclass agent instance.
    :param eval_data: List of dicts with ``input`` +
        expected fields (same format as
        ``run_evaluation``).
    :param expected_fields: Field names to compare (e.g.
        ``["category"]``).
    :param min_accuracy: Minimum acceptable accuracy
        (0.0 to 1.0). Defaults to 0.9.
    :return: The metrics dict from ``compute_metrics()``.
    :raises AssertionError: If accuracy is below threshold.

    Usage::

        eval_data = [
            {"input": {"text": "hello"}, "category": "GREETING"},
            {"input": {"text": "bye"}, "category": "FAREWELL"},
        ]
        metrics = assert_accuracy_above(
            agent, eval_data, ["category"], 0.9,
        )
    """
    results = run_evaluation(agent, eval_data, expected_fields)
    metrics = compute_metrics(results, expected_fields)

    accuracy = metrics["accuracy"]
    passed = metrics["passed"]
    total = metrics["total"]

    assert accuracy >= min_accuracy, (
        f"Accuracy {accuracy:.1%} ({passed}/{total}) "
        f"is below threshold {min_accuracy:.1%}"
    )

    return metrics
