"""Unit tests for locating where inside a conversation a rule breaks."""

import pytest

from agents7.eval.style import ReplyContext, StyleRule
from agents7.eval.style_rhythm import (
    RhythmDecomposition,
    decompose_rhythm,
    split_conversations,
    turn_matrix,
)
from agents7.exceptions import AgentConfigurationError


# --- Helpers ---


def reply(index, text, prompt="ya"):
    """One reply, split into messages on its lines."""
    return ReplyContext(
        index=index,
        prompt=prompt,
        text=text,
        messages=tuple(line for line in text.split("\n") if line),
    )


def two_messages(ctx):
    """Violation on two messages, not applicable to an opening."""
    if ctx.prompt is None:
        return None
    return len(ctx.messages) > 1


RULE = StyleRule(id="one_message", description="d", check=two_messages)


# ── split_conversations Tests ──────────────────────────────────────


def test_split_conversations_breaks_where_index_restarts():
    replies = [reply(0, "a"), reply(1, "b"), reply(0, "c"), reply(1, "d")]
    assert [[r.text for r in c] for c in split_conversations(replies)] == [
        ["a", "b"],
        ["c", "d"],
    ]


def test_split_conversations_empty():
    assert split_conversations([]) == []


# ── turn_matrix Tests ──────────────────────────────────────────────


def test_turn_matrix_skips_turns_the_rule_does_not_apply_to():
    replies = [
        reply(0, "hola", prompt=None),
        reply(1, "a\nb"),
        reply(2, "a"),
        reply(0, "hola", prompt=None),
        reply(1, "a"),
    ]
    assert turn_matrix(replies, RULE) == [[1, 0], [0]]


def test_turn_matrix_drops_conversations_never_applicable():
    replies = [reply(0, "hola", prompt=None), reply(0, "a\nb")]
    assert turn_matrix(replies, RULE) == [[1]]


# ── decompose_rhythm Tests ─────────────────────────────────────────


def test_decompose_rhythm_reports_position_drift():
    rows = [[0, 0, 1, 1]] * 3 + [[0, 1, 1, 1]]
    result = decompose_rhythm(rows, trials=50)
    assert result.first_half < result.second_half


def test_decompose_rhythm_cuts_rows_to_the_shortest():
    rows = [[0, 1, 0, 1, 1, 1], [1, 0, 1, 0], [0, 0, 1, 1, 0]]
    assert decompose_rhythm(rows, trials=50).turns == 4


def test_decompose_rhythm_sees_conversations_sitting_apart():
    rows = [[1] * 8, [0] * 8, [1] * 8, [0] * 8]
    result = decompose_rhythm(rows, trials=50)
    assert result.dispersion_ratio > 2


def test_decompose_rhythm_dispersion_none_without_coin_flip_spread():
    rows = [[1, 1, 0, 0]] * 3
    assert decompose_rhythm(rows, trials=50).dispersion_ratio is None


def test_decompose_rhythm_momentum_found_in_runs():
    # Each conversation breaks the rule in one unbroken run, at a
    # different place, so neither position nor level explains it.
    rows = [
        [1 if start <= i < start + 6 else 0 for i in range(12)]
        for start in (0, 3, 6, 1, 4, 5, 2, 0)
    ]
    assert decompose_rhythm(rows, trials=2000).momentum_p < 0.01


def test_decompose_rhythm_no_momentum_when_turns_alternate():
    rows = [[i % 2 for i in range(12)], [(i + 1) % 2 for i in range(12)]] * 3
    assert decompose_rhythm(rows, trials=2000).momentum_p > 0.5


def test_decompose_rhythm_same_seed_same_p():
    rows = [[0, 1, 1, 0, 1], [1, 1, 0, 0, 0], [0, 0, 1, 1, 1]]
    first = decompose_rhythm(rows, trials=300, seed=7)
    second = decompose_rhythm(rows, trials=300, seed=7)
    assert first.momentum_p == second.momentum_p


def test_decompose_rhythm_uncontrolled_contrast():
    rows = [[1, 1, 0, 0]] * 3
    result = decompose_rhythm(rows, trials=10)
    assert (result.after_violation, result.after_violation_n) == (0.5, 6)


def test_decompose_rhythm_contrast_none_without_violations():
    result = decompose_rhythm([[0, 0, 0, 0]] * 3, trials=10)
    assert result.after_violation is None


@pytest.mark.parametrize(
    "rows",
    [
        [[0, 1, 0, 1]] * 2,
        [[0, 1, 0]] * 5,
        [],
    ],
)
def test_decompose_rhythm_refuses_too_little(rows):
    with pytest.raises(AgentConfigurationError):
        decompose_rhythm(rows)


def test_decompose_rhythm_refuses_zero_trials():
    with pytest.raises(AgentConfigurationError):
        decompose_rhythm([[0, 1, 0, 1]] * 3, trials=0)


# ── RhythmDecomposition.summary Tests ──────────────────────────────


def test_summary_names_all_three_effects():
    result = decompose_rhythm([[0, 1, 1, 0], [1, 0, 1, 1]] * 2, trials=20)
    text = result.summary()
    assert all(
        head in text for head in ("POSITION", "CONVERSATION", "MOMENTUM")
    )


def test_summary_without_contrast_or_ratio():
    result = RhythmDecomposition(
        conversations=3,
        turns=4,
        pooled=0.0,
        position=(0.0, 0.0, 0.0, 0.0),
        observed_sd=0.0,
        independent_sd=0.0,
        conversation_rates=(0.0, 0.0, 0.0),
        lag1_covariance=0.0,
        momentum_p=1.0,
        trials=10,
        after_violation=None,
        after_violation_n=0,
        after_clean=0.0,
        after_clean_n=9,
    )
    assert "uncontrolled" not in result.summary()
