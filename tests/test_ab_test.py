"""Unit tests for agents7.eval.ab_test (A/B testing harness)."""

import pytest

from agents7.eval.ab_test import (
    ABTestResult,
    run_ab_test,
    run_ab_test_async,
)

from helpers import make_agent_mock, make_response

# ── run_ab_test ────────────────────────────────────────────


def _cases():
    """Two golden rows scored on ``category``."""
    return [
        {"input": {"text": "x"}, "category": "POSITIVE"},
        {"input": {"text": "y"}, "category": "NEGATIVE"},
    ]


def test_ab_test_a_wins():
    """Agent A scores higher → A is the winner, positive delta."""
    agent_a = make_agent_mock(
        [make_response("POSITIVE"), make_response("NEGATIVE")]
    )
    agent_b = make_agent_mock(
        [make_response("POSITIVE"), make_response("POSITIVE")]
    )
    result = run_ab_test(agent_a, agent_b, _cases(), ["category"])

    assert isinstance(result, ABTestResult)
    assert result.metrics_a["accuracy"] == 1.0
    assert result.metrics_b["accuracy"] == 0.5
    assert result.accuracy_delta == pytest.approx(0.5)
    assert result.winner == "A"


def test_ab_test_b_wins():
    """Agent B scores higher → B wins, negative delta."""
    agent_a = make_agent_mock(
        [make_response("POSITIVE"), make_response("POSITIVE")]
    )
    agent_b = make_agent_mock(
        [make_response("POSITIVE"), make_response("NEGATIVE")]
    )
    result = run_ab_test(agent_a, agent_b, _cases(), ["category"])

    assert result.accuracy_delta == pytest.approx(-0.5)
    assert result.winner == "B"


def test_ab_test_tie():
    """Equal accuracy → tie."""
    agent_a = make_agent_mock(
        [make_response("POSITIVE"), make_response("NEGATIVE")]
    )
    agent_b = make_agent_mock(
        [make_response("POSITIVE"), make_response("NEGATIVE")]
    )
    result = run_ab_test(agent_a, agent_b, _cases(), ["category"])

    assert result.accuracy_delta == pytest.approx(0.0)
    assert result.winner == "tie"


def test_ab_test_custom_labels_used_as_winner():
    """Custom labels flow through to the winner field."""
    agent_a = make_agent_mock(
        [make_response("POSITIVE"), make_response("NEGATIVE")]
    )
    agent_b = make_agent_mock(
        [make_response("POSITIVE"), make_response("POSITIVE")]
    )
    result = run_ab_test(
        agent_a,
        agent_b,
        _cases(),
        ["category"],
        label_a="qwen",
        label_b="llama",
    )

    assert result.label_a == "qwen"
    assert result.label_b == "llama"
    assert result.winner == "qwen"


def test_ab_test_per_field_delta():
    """Per-field deltas isolate the field that differs."""
    cases = [
        {"input": {"text": "x"}, "category": "POSITIVE", "reason": "HAPPY"},
        {"input": {"text": "y"}, "category": "NEGATIVE", "reason": "SAD"},
    ]
    # Both agents nail category; A nails both reasons, B misses one.
    agent_a = make_agent_mock(
        [make_response("POSITIVE", "HAPPY"), make_response("NEGATIVE", "SAD")]
    )
    agent_b = make_agent_mock(
        [make_response("POSITIVE", "SAD"), make_response("NEGATIVE", "SAD")]
    )
    result = run_ab_test(agent_a, agent_b, cases, ["category", "reason"])

    assert result.per_field_delta["category"] == pytest.approx(0.0)
    assert result.per_field_delta["reason"] == pytest.approx(0.5)


def test_ab_test_agreement_counts():
    """Agreement buckets are computed per aligned case."""
    # Case 1: both pass. Case 2: A passes, B fails.
    agent_a = make_agent_mock(
        [make_response("POSITIVE"), make_response("NEGATIVE")]
    )
    agent_b = make_agent_mock(
        [make_response("POSITIVE"), make_response("POSITIVE")]
    )
    result = run_ab_test(agent_a, agent_b, _cases(), ["category"])

    agree = result.agreement
    assert agree["both_passed"] == 1
    assert agree["a_only"] == 1
    assert agree["b_only"] == 0
    assert agree["both_failed"] == 0
    # Outputs match on case 1 only → 0.5 agreement.
    assert agree["output_agreement"] == pytest.approx(0.5)


def test_ab_test_infers_fields_when_omitted():
    """expected_fields=None infers fields from the rows."""
    agent_a = make_agent_mock(
        [make_response("POSITIVE"), make_response("NEGATIVE")]
    )
    agent_b = make_agent_mock(
        [make_response("POSITIVE"), make_response("NEGATIVE")]
    )
    result = run_ab_test(agent_a, agent_b, _cases())

    assert "category" in result.expected_fields
    assert result.metrics_a["accuracy"] == 1.0


def test_ab_test_empty_cases():
    """No test cases → zeroed metrics and a tie."""
    agent_a = make_agent_mock([])
    agent_b = make_agent_mock([])
    result = run_ab_test(agent_a, agent_b, [], ["category"])

    assert result.metrics_a["total"] == 0
    assert result.accuracy_delta == 0.0
    assert result.winner == "tie"
    assert result.agreement["output_agreement"] == 0.0


# ── run_ab_test_async ──────────────────────────────────────


@pytest.mark.asyncio
async def test_ab_test_async_matches_sync():
    """Async harness produces the same winner as the sync one."""
    agent_a = make_agent_mock(
        [make_response("POSITIVE"), make_response("NEGATIVE")]
    )
    agent_b = make_agent_mock(
        [make_response("POSITIVE"), make_response("POSITIVE")]
    )
    agent_a.act_in_batch_async = _async_return(
        [make_response("POSITIVE"), make_response("NEGATIVE")]
    )
    agent_b.act_in_batch_async = _async_return(
        [make_response("POSITIVE"), make_response("POSITIVE")]
    )
    result = await run_ab_test_async(agent_a, agent_b, _cases(), ["category"])

    assert result.winner == "A"
    assert result.accuracy_delta == pytest.approx(0.5)


def _async_return(responses):
    """Build an AsyncMock returning *responses* from act_in_batch_async."""
    from unittest.mock import AsyncMock

    return AsyncMock(return_value=list(responses))
