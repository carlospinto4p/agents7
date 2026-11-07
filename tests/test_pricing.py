"""Tests for token pricing and cost estimation."""

import pytest

from agents7.llm.pricing import Pricing


# ── Construction ───────────────────────────────────────────────────────────


def test_cached_input_defaults_to_input():
    """A provider with no prompt discount charges one input rate."""
    p = Pricing(input=0.44, output=1.32)

    assert p.cached_input == 0.44


def test_cached_input_is_kept_when_given():
    p = Pricing(input=0.44, output=1.32, cached_input=0.014)

    assert p.cached_input == 0.014


def test_flat_charges_one_rate_for_everything():
    p = Pricing.flat(2.0)

    assert (p.input, p.output, p.cached_input) == (2.0, 2.0, 2.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input": -1.0, "output": 1.0},
        {"input": 1.0, "output": -1.0},
        {"input": 1.0, "output": 1.0, "cached_input": -1.0},
    ],
)
def test_negative_rate_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Pricing(**kwargs)


# ── Cost ───────────────────────────────────────────────────────────────────


def test_cost_is_in_cents_from_dollar_rates():
    """Rates are dollars per 1M; the result is cents."""
    p = Pricing(input=1.0, output=0.0)

    assert p.cost_cents(prompt_tokens=1_000_000, tokens=0) == (
        pytest.approx(100.0)
    )


def test_prompt_and_completion_bill_separately():
    p = Pricing(input=0.44, output=1.32)

    cost = p.cost_cents(prompt_tokens=1_000_000, tokens=1_000_000)

    assert cost == pytest.approx((0.44 + 1.32) * 100)


def test_cached_tokens_are_billed_at_the_cached_rate():
    p = Pricing(input=0.44, output=1.32, cached_input=0.014)

    cost = p.cost_cents(
        prompt_tokens=1_000_000,
        tokens=0,
        cached_prompt_tokens=1_000_000,
    )

    assert cost == pytest.approx(0.014 * 100)


def test_cached_tokens_are_a_subset_of_prompt_tokens():
    """The cached count comes out of the prompt total, not on top."""
    p = Pricing(input=1.0, output=0.0, cached_input=0.0)

    cost = p.cost_cents(
        prompt_tokens=1_000_000,
        tokens=0,
        cached_prompt_tokens=250_000,
    )

    # 750k billed fresh, 250k free.
    assert cost == pytest.approx(75.0)


def test_cached_over_prompt_total_is_clamped():
    """A provider reporting more cached than prompt cannot go negative."""
    p = Pricing(input=1.0, output=0.0, cached_input=0.0)

    cost = p.cost_cents(
        prompt_tokens=1000,
        tokens=0,
        cached_prompt_tokens=99_999,
    )

    assert cost == pytest.approx(0.0)


def test_zero_tokens_costs_nothing():
    p = Pricing(input=0.44, output=1.32)

    assert p.cost_cents(prompt_tokens=0, tokens=0) == 0.0


def test_a_long_system_prompt_is_dominated_by_cached_input():
    """The case this exists for: a persona resent every turn.

    A chat agent resends its whole persona on each turn, so after
    the first turn nearly every prompt token is a cache hit. At
    DeepSeek's published rates that is ~31x cheaper than paying
    the fresh rate, which is the difference between a real
    estimate and a useless one.
    """
    p = Pricing(input=0.44, output=1.32, cached_input=0.014)

    fresh = p.cost_cents(prompt_tokens=6239, tokens=9)
    cached = p.cost_cents(
        prompt_tokens=6239, tokens=9, cached_prompt_tokens=6239
    )

    assert fresh > cached * 10
