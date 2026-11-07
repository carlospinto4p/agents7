"""Unit tests for comparing two sets of style-contract invocations."""

import math

import pytest

from agents7.eval.style import RunScore
from agents7.eval.style_arms import (
    LIVE_RATE,
    MIN_ARM_RUNS,
    ArmComparison,
    _excess_variance,
    _signed_sd,
    compare_arms,
)
from agents7.eval.style_history import RuleRecord, StyleRunRecord
from agents7.exceptions import AgentConfigurationError


# --- Helpers ---


def runs(script, counts, applicable=20):
    """RunScores for one script: one per (violations) in `counts`."""
    return tuple(
        RunScore(
            label=f"{script}#{index}",
            violations=violations,
            applicable=applicable,
        )
        for index, violations in enumerate(counts, start=1)
    )


def record(rules, when="2026-09-05T00:00:00Z", fingerprint="fp"):
    """One invocation carrying the given `{rule_id: RunScore tuple}`."""
    return StyleRunRecord(
        recorded_at=when,
        subject="test",
        fingerprint=fingerprint,
        agent="a",
        agent_version=1,
        model_id="p::m",
        repeats=len(next(iter(rules.values()))),
        replies=100,
        calls=100,
        cost_cents=1.0,
        rules=tuple(
            RuleRecord(
                id=rule_id,
                budget=None,
                violations=sum(run.violations for run in scores),
                applicable=sum(run.applicable for run in scores),
                runs=scores,
            )
            for rule_id, scores in rules.items()
        ),
    )


def arm(per_invocation, rule_id="r"):
    """One arm: a record per list of per-run violation counts."""
    return [
        record({rule_id: runs("cold", counts)}, when=f"2026-09-0{i}")
        for i, counts in enumerate(per_invocation, start=1)
    ]


# Two arms at the same level whose *invocation* scores move by
# different amounts — which is the quantity a band is built from, and
# not the same thing as how much the runs inside one invocation
# disagree. WIDE scores 33%/50%/67%, TIGHT 47%/50%/53%.
WIDE = [[5, 5, 10], [10, 10, 10], [15, 15, 10]]
TIGHT = [[9, 9, 10], [10, 10, 10], [11, 11, 10]]


# --- Basic shape ---


def test_compare_arms_returns_a_typed_result():
    result = compare_arms(arm(WIDE), arm(TIGHT), "left", "right")
    assert isinstance(result, ArmComparison)
    assert result.left_count == 3
    assert result.right_count == 3
    assert [rule.rule_id for rule in result.rules] == ["r"]


def test_level_is_the_mean_rate_each_way():
    result = compare_arms(arm([[10, 10, 10]] * 3), arm([[5, 5, 5]] * 3))
    rule = result.rules[0]
    assert rule.left_rate == pytest.approx(0.5)
    assert rule.right_rate == pytest.approx(0.25)
    assert rule.level_diff == pytest.approx(0.25)


def test_a_rule_missing_from_one_arm_is_dropped():
    left = arm(WIDE)
    right = [record({"other": runs("cold", [1, 2, 3])}) for _ in range(3)]
    assert compare_arms(left, right).rules == ()


def test_too_few_invocations_raises():
    with pytest.raises(AgentConfigurationError, match="at least 2"):
        compare_arms(arm(WIDE)[:1], arm(TIGHT))


def test_no_rules_raises():
    empty = [
        StyleRunRecord(
            recorded_at="2026-09-05T00:00:00Z",
            subject="s",
            fingerprint="fp",
            agent="a",
            agent_version=1,
            model_id="p::m",
            repeats=1,
            replies=0,
            calls=0,
            cost_cents=None,
            rules=(),
        )
        for _ in range(2)
    ]
    with pytest.raises(AgentConfigurationError, match="no rules"):
        compare_arms(empty, empty)


# --- Permutation is over whole invocations ---


def test_permutes_whole_invocations_never_conversations():
    """Two conversations inside one invocation are not exchangeable
    with two inside another, so the enumeration is over invocations."""
    left, right = arm(WIDE), arm(TIGHT + [[9, 10, 11]])
    result = compare_arms(left, right, trim=False)
    assert result.permutations == math.comb(3 + 4, 3)
    # And emphatically not over the 21 conversations those carry.
    assert sum(len(r.rules[0].runs) for r in left + right) == 21


def test_p_values_are_multiples_of_one_over_the_permutations():
    result = compare_arms(arm(WIDE), arm(TIGHT))
    rule = result.rules[0]
    for p in (rule.level_p, rule.spread_p):
        assert p * result.permutations == pytest.approx(
            round(p * result.permutations)
        )


# --- (1) The size a test could have seen, beside every p-value ---


def test_every_live_rule_reports_what_it_could_have_seen():
    """A null result is unreadable without the size of difference the
    test had the power to register, so it sits beside the p-value."""
    result = compare_arms(arm(WIDE), arm(TIGHT))
    assert result.live
    for rule in result.live:
        assert rule.spread_threshold > 1.0


def test_the_panel_reports_what_it_could_have_seen():
    result = compare_arms(arm(WIDE), arm(TIGHT))
    assert result.panel_threshold is not None
    assert result.panel_threshold > 1.0
    assert "would have registered" in result.summary()


# --- (2) Rules that never fire stay out of the panel ---


def test_rules_that_never_fire_are_excluded_from_the_panel():
    """A rule with no violations either way has a spread of zero on
    both sides — a ratio of exactly one carrying no information. Enough
    of them manufacture a dead heat out of whatever the live rules did.
    """
    live_only = compare_arms(arm(WIDE), arm(TIGHT))

    def with_dead(counts):
        return [
            record(
                {
                    "r": runs("cold", per),
                    "dead_a": runs("cold", [0, 0, 0]),
                    "dead_b": runs("cold", [0, 0, 0]),
                    "dead_c": runs("cold", [0, 0, 0]),
                },
                when=f"2026-09-0{i}",
            )
            for i, per in enumerate(counts, start=1)
        ]

    padded = compare_arms(with_dead(WIDE), with_dead(TIGHT))

    assert len(padded.rules) == 4
    assert [rule.rule_id for rule in padded.live] == ["r"]
    assert padded.median_spread_ratio == pytest.approx(
        live_only.median_spread_ratio
    )
    assert padded.panel_p == pytest.approx(live_only.panel_p)


def test_a_rule_below_the_live_rate_does_not_fire():
    quiet = [[0, 0, 0], [0, 0, 1], [0, 0, 0]]
    result = compare_arms(arm(quiet), arm(quiet))
    rule = result.rules[0]
    assert rule.left_rate < LIVE_RATE
    assert not rule.fires
    assert not rule.live
    assert result.median_spread_ratio is None


# --- (3) Binomial noise is subtracted before comparing conversations ---


def test_excess_variance_subtracts_the_binomial_part():
    """Exact arithmetic, so the correction cannot silently stop.

    Three runs of 20 replies at rates .40/.50/.60: the observed
    variance is .01 and the binomial part mu(1-mu)/m is .0125, over a
    (1 - 1/m) denominator of .95.
    """
    observed = _excess_variance(runs("cold", [8, 10, 12]))
    assert observed == pytest.approx((0.01 - 0.0125) / 0.95)


def test_two_arms_at_different_levels_are_not_called_different_wobble():
    """Two arms with different mean rates carry different amounts of
    binomial noise, so comparing raw run spreads would report a
    difference in *level* as a difference in wobble."""
    high = runs("cold", [8, 10, 12])  # rates .40 .50 .60, sd 10%
    low = runs("cold", [1, 2, 3])  # rates .05 .10 .15, sd  5%

    raw = _raw_sd(high) / _raw_sd(low)
    assert raw == pytest.approx(2.0)

    corrected = _signed_sd(_excess_variance(high)) / _signed_sd(
        _excess_variance(low)
    )
    assert corrected == pytest.approx(1.11, abs=0.05)


def _raw_sd(scores):
    """Uncorrected standard deviation of a group's run rates."""
    rates = [run.violations / run.applicable for run in scores]
    mean = sum(rates) / len(rates)
    return (sum((r - mean) ** 2 for r in rates) / (len(rates) - 1)) ** 0.5


def test_excess_variance_keeps_scripts_apart():
    """Two scripts are two stimuli with two different true rates, so
    the gap between them is structure and not wobble.

    The same four runs — two at 50%, two at 10% — read as one script
    put that whole gap down to conversations disagreeing.
    """
    apart = runs("cold", [10, 10]) + runs("warm", [2, 2])
    together = runs("cold", [10, 10, 2, 2])
    assert _excess_variance(apart) < 0
    assert _excess_variance(together) > 0.04


def test_excess_variance_needs_a_script_with_two_runs():
    assert _excess_variance(()) is None
    assert _excess_variance(runs("cold", [5])) is None


def test_tau_ratio_is_none_when_a_side_is_underdispersed():
    """A negative estimate means those conversations agreed *more*
    closely than binomial noise predicts — small in a different sense,
    and a ratio over it means nothing."""
    result = compare_arms(
        arm([[8, 10, 10], [10, 10, 10], [10, 10, 12]]),
        arm(WIDE),
    )
    rule = result.rules[0]
    assert rule.live
    assert rule.left_tau is not None and rule.left_tau < 0
    assert rule.right_tau is not None and rule.right_tau > 0
    assert rule.tau_ratio is None


# --- Trimming, and the drift it has to declare ---


def test_trim_keeps_the_newest_and_reports_what_it_held_back():
    left, right = arm(WIDE), arm(TIGHT + [[9, 10, 11], [10, 10, 10]])
    result = compare_arms(left, right, "L", "R")
    assert result.left_count == result.right_count == 3
    assert result.held_back_right == 2
    assert result.held_back_left == 0
    assert [row.rule_id for row in result.drift_right] == ["r"]
    assert not result.drift_left
    assert "2 earlier R held back" in result.summary()


def test_trim_can_hold_back_from_the_left_arm_too():
    result = compare_arms(arm(WIDE + [[1, 10, 19]]), arm(TIGHT), "L", "R")
    assert result.held_back_left == 1
    assert [row.rule_id for row in result.drift_left] == ["r"]


def test_drift_compares_the_held_back_against_the_fresh():
    older = [[20, 20, 20], [20, 20, 20]]
    fresh = [[0, 0, 0], [0, 10, 20], [10, 10, 10]]
    result = compare_arms(arm(TIGHT), arm(older + fresh), "L", "R")
    row = result.drift_right[0]
    assert row.earlier == pytest.approx(1.0)
    assert row.fresh == pytest.approx(1 / 3)
    assert row.diff == pytest.approx(2 / 3)


def test_no_trim_uses_every_invocation_and_reports_no_drift():
    result = compare_arms(arm(WIDE), arm(TIGHT + [[9, 10, 11]]), trim=False)
    assert result.left_count == 3
    assert result.right_count == 4
    assert result.held_back_left == result.held_back_right == 0
    assert result.drift_left == result.drift_right == ()
    assert "held back" not in result.summary()


# --- The rendered summary ---


def test_summary_carries_the_three_tables_and_the_panel():
    text = compare_arms(arm(WIDE), arm(TIGHT), "guarded", "unguarded")
    rendered = text.summary()
    assert "3 guarded vs 3 unguarded" in rendered
    assert "LEVEL — mean rate each way" in rendered
    assert "INVOCATION SPREAD — what a band is built from" in rendered
    assert "CONVERSATION SPREAD — beyond binomial noise" in rendered
    assert "PANEL over the 1 rules that fire" in rendered


def test_summary_omits_the_panel_when_nothing_is_live():
    quiet = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    rendered = compare_arms(arm(quiet), arm(quiet)).summary()
    assert "PANEL" not in rendered


def test_left_moved_more_counts_the_live_rules():
    result = compare_arms(arm(WIDE), arm(TIGHT))
    assert result.left_moved_more == 1
    assert result.rules[0].spread_ratio > 1.0


def test_min_arm_runs_is_the_floor_a_spread_needs():
    assert MIN_ARM_RUNS == 2
