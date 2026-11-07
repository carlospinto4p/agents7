"""Where inside a conversation does a style rule break?

A style contract scores a rule per invocation and per run. Both are
too coarse to say whether a rule holds at the start of a conversation
and decays, whether whole conversations simply sit high or low, or
whether one broken turn makes the next more likely. Those are three
different mechanisms, they point at three different fixes, and a raw
"does a violation follow a violation" figure runs all three together.

This separates them, from the replies a `StyleReport` already keeps
in order:

1. **Position.** The rate drifts over a conversation, so two adjacent
   turns are both more likely to break the rule late in it. That alone
   manufactures apparent memory.
2. **Conversation.** Whole conversations sit high or low — the
   over-dispersion that makes a rule expensive to resolve. Measured
   with the position drift removed first, so a rule that simply decays
   does not look like conversations disagreeing.
3. **Momentum.** Whether a turn still predicts the next once both of
   the above are removed. Tested by permuting the turns *within* each
   conversation, which keeps every position rate and every
   conversation's mean fixed by construction.

Built to find out why a style rule held on a conversation's first
turns and broke on later ones. Nothing in the method is specific to
one agent or one rule.

Usage::

    report = run_style_contract(factory, one_script_contract)
    rows = turn_matrix(report.replies, rule)
    print(decompose_rhythm(rows).summary())

Keep *rows*: at any temperature above zero a re-run is a different set
of conversations, so the matrix cannot be recovered afterwards, while
re-analysing it costs nothing.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, stdev

from agents7.eval.style import ReplyContext, StyleRule
from agents7.exceptions import AgentConfigurationError

# The conversation spread is a standard deviation of conversation
# means, which needs a few to mean anything; the position row and the
# halves need a few turns. Below either, the decomposition is refused
# rather than printed as though it said something.
MIN_RHYTHM_CONVERSATIONS = 3
MIN_RHYTHM_TURNS = 4

# Enough shuffles that the p-value's own error (about 0.002 at
# p = 0.05) is smaller than any difference worth reading.
RHYTHM_TRIALS = 10000


def split_conversations(
    replies: Sequence[ReplyContext],
) -> list[list[ReplyContext]]:
    """Split a flat reply list back into conversations.

    `StyleReport.replies` holds every reply of every run in order, and
    `ReplyContext.index` restarts at zero on each run — which is the
    only boundary there is.

    :param replies: Replies in run order, as a report keeps them.
    :return: One list per conversation, in the same order.
    """
    out: list[list[ReplyContext]] = []
    current: list[ReplyContext] = []
    for ctx in replies:
        if ctx.index == 0 and current:
            out.append(current)
            current = []
        current.append(ctx)
    if current:
        out.append(current)
    return out


def turn_matrix(
    replies: Sequence[ReplyContext], rule: StyleRule
) -> list[list[int]]:
    """One 0/1 row per conversation, one column per applicable turn.

    Turns the rule does not apply to are skipped rather than scored
    as compliant, so column *n* is the rule's *n*-th chance in that
    conversation, not its *n*-th reply. Conversations where the rule
    never applied are dropped.

    **Score replies from one script.** Two scripts are two stimuli
    with two different true rates, so rows from both would put that
    gap into the conversation spread.

    :param replies: Replies in run order, as a report keeps them.
    :param rule: The contract rule to score.
    :return: The matrix, rows in run order. Rows may differ in length
        when conversations did.
    """
    rows = []
    for conversation in split_conversations(replies):
        scored = (rule.check(ctx) for ctx in conversation)
        row = [int(verdict) for verdict in scored if verdict is not None]
        if row:
            rows.append(row)
    return rows


def _lag1(rows: Sequence[Sequence[float]]) -> float:
    """Mean product of adjacent values, pooled over conversations."""
    return fmean(
        row[i] * row[i + 1] for row in rows for i in range(len(row) - 1)
    )


def _mean_or_none(values: Sequence[int]) -> float | None:
    """The mean, or None for nothing to average."""
    return fmean(values) if values else None


@dataclass(frozen=True)
class RhythmDecomposition:
    """Where a rule breaks, split into position, conversation and
    momentum.

    :param conversations: How many conversations were decomposed.
    :param turns: Applicable turns per conversation used. Longer
        conversations are cut to the shortest one's length.
    :param pooled: Violation rate over the whole matrix.
    :param position: Violation rate at each applicable-turn ordinal.
    :param observed_sd: Spread of conversation means once the position
        drift is removed.
    :param independent_sd: The spread independence alone would give —
        every turn a coin at its own position's rate.
    :param conversation_rates: Each conversation's raw rate, sorted.
    :param lag1_covariance: Adjacent-turn covariance with position and
        conversation removed.
    :param momentum_p: Share of within-conversation shuffles whose
        covariance was at least as large, add-one smoothed.
    :param trials: How many shuffles `momentum_p` is over.
    :param after_violation: The uncontrolled rate on the turn after a
        violation, or None if there was no such turn.
    :param after_violation_n: How many turns that is over.
    :param after_clean: The same after a compliant turn.
    :param after_clean_n: How many turns that is over.
    """

    conversations: int
    turns: int
    pooled: float
    position: tuple[float, ...]
    observed_sd: float
    independent_sd: float
    conversation_rates: tuple[float, ...]
    lag1_covariance: float
    momentum_p: float
    trials: int
    after_violation: float | None
    after_violation_n: int
    after_clean: float | None
    after_clean_n: int

    @property
    def first_half(self) -> float:
        """Mean position rate over the first half of the turns."""
        return fmean(self.position[: self.turns // 2])

    @property
    def second_half(self) -> float:
        """Mean position rate over the second half of the turns."""
        return fmean(self.position[self.turns // 2 :])

    @property
    def dispersion_ratio(self) -> float | None:
        """Observed conversation spread over what independence gives.

        None when independence gives zero — every position rate was 0
        or 1, so there is no coin-flip spread to compare against.
        """
        if self.independent_sd == 0:
            return None
        return self.observed_sd / self.independent_sd

    def summary(self) -> str:
        """The three effects, then the uncontrolled figure for contrast.

        :return: Multi-line string.
        """
        ratio = self.dispersion_ratio
        lines = [
            f"{self.conversations} conversations x {self.turns} turns, "
            f"pooled {self.pooled:.0%}",
            "",
            "POSITION — violation rate by applicable-turn ordinal",
            "  " + " ".join(f"{p:.0%}" for p in self.position),
            f"  first half {self.first_half:.0%}, "
            f"second half {self.second_half:.0%}",
            "",
            "CONVERSATION — spread of conversation means, position removed",
            f"  observed sd {self.observed_sd:.1%}, independence alone "
            f"{self.independent_sd:.1%}, ratio "
            f"{'-' if ratio is None else f'{ratio:.2f}x'}",
            "  conversation rates: "
            + " ".join(f"{r:.0%}" for r in self.conversation_rates),
            "",
            "MOMENTUM — does a turn still predict the next one?",
            f"  lag-1 covariance {self.lag1_covariance:+.4f}",
            f"  turns permuted within each conversation: "
            f"p = {self.momentum_p:.3f}",
        ]
        if self.after_violation is not None and self.after_clean is not None:
            lines += [
                "",
                f"  For contrast, the uncontrolled figure: "
                f"{self.after_violation:.0%} after a violation "
                f"(n={self.after_violation_n}), {self.after_clean:.0%} "
                f"after a clean turn (n={self.after_clean_n}). Most of "
                f"that gap is position and conversation, not memory — "
                f"which is why it must not be quoted on its own.",
            ]
        return "\n".join(lines)


def decompose_rhythm(
    rows: Sequence[Sequence[int]],
    *,
    trials: int = RHYTHM_TRIALS,
    seed: int = 0,
) -> RhythmDecomposition:
    """Split a rule's per-turn violations into three effects.

    :param rows: One 0/1 row per conversation — see `turn_matrix`.
        Rows are cut to the shortest one's length.
    :param trials: How many within-conversation shuffles the momentum
        p-value is over.
    :param seed: Seed for the shuffles, so a saved matrix re-analyses
        to the same figures.
    :return: The decomposition.
    :raises AgentConfigurationError: If there are fewer than
        `MIN_RHYTHM_CONVERSATIONS` rows, fewer than `MIN_RHYTHM_TURNS`
        turns in the shortest one, or *trials* is below 1.
    """
    if trials < 1:
        raise AgentConfigurationError(
            f"decompose_rhythm: trials must be at least 1. Got {trials!r}."
        )
    width = min((len(row) for row in rows), default=0)
    if len(rows) < MIN_RHYTHM_CONVERSATIONS or width < MIN_RHYTHM_TURNS:
        raise AgentConfigurationError(
            f"decompose_rhythm: needs at least {MIN_RHYTHM_CONVERSATIONS} "
            f"conversations of {MIN_RHYTHM_TURNS} applicable turns, got "
            f"{len(rows)} x {width}."
        )
    matrix = [list(row[:width]) for row in rows]
    position = [fmean(row[i] for row in matrix) for i in range(width)]

    # Conversation, with the position drift removed first.
    resid = [[row[i] - position[i] for i in range(width)] for row in matrix]
    means = [fmean(row) for row in resid]
    independent = (sum(p * (1 - p) for p in position) / width**2) ** 0.5

    # Momentum, with the conversation removed as well. Shuffling a
    # conversation's own turns keeps its mean and every position rate,
    # so whatever the shuffle destroys is order.
    centred = [[v - m for v in row] for row, m in zip(resid, means)]
    actual = _lag1(centred)
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        shuffled = []
        for row in centred:
            copy = list(row)
            rng.shuffle(copy)
            shuffled.append(copy)
        if _lag1(shuffled) >= actual:
            hits += 1

    pairs = [(row[i], row[i + 1]) for row in matrix for i in range(width - 1)]
    after_bad = [nxt for cur, nxt in pairs if cur]
    after_ok = [nxt for cur, nxt in pairs if not cur]
    return RhythmDecomposition(
        conversations=len(matrix),
        turns=width,
        pooled=fmean(v for row in matrix for v in row),
        position=tuple(position),
        observed_sd=stdev(means),
        independent_sd=independent,
        conversation_rates=tuple(sorted(fmean(row) for row in matrix)),
        lag1_covariance=actual,
        momentum_p=(hits + 1) / (trials + 1),
        trials=trials,
        after_violation=_mean_or_none(after_bad),
        after_violation_n=len(after_bad),
        after_clean=_mean_or_none(after_ok),
        after_clean_n=len(after_ok),
    )


__all__ = [
    "MIN_RHYTHM_CONVERSATIONS",
    "MIN_RHYTHM_TURNS",
    "RHYTHM_TRIALS",
    "RhythmDecomposition",
    "decompose_rhythm",
    "split_conversations",
    "turn_matrix",
]
