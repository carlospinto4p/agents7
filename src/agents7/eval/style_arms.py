"""Comparing two sets of style-contract invocations against each other.

`compare_to_history` answers *"did this invocation leave its band"* —
one configuration judged against its own past. It has no answer for
*"do these two configurations wobble by different amounts"*, because a
band is a property of one configuration and this is a comparison of
two. Hence a separate instrument.

Built on 5th September 2026 to answer whether one agent's guard is
itself a source of variance, and promoted here because nothing in the
method is specific to that agent or to guards. The question recurs
whenever the apparatus around an agent changes — a guard added or
dropped, a provider swapped, a cache turned on.

Three things are reported, and they answer different halves:

* **Level** — the mean rate each way. An apparatus that repairs faults
  shows up here, not in the spread.
* **Invocation spread** — how far whole-invocation scores sit from one
  another. This is the quantity `compare_to_history` builds a band
  from, so it is what decides whether an apparatus change affects a
  band.
* **Conversation spread** — the wobble between individual
  conversations, beyond the coin-flip noise a run of that length
  carries anyway. There are more conversations than invocations, so it
  sees a smaller difference; and it is the level a guard would act at,
  since a guard runs inside a conversation.

Significance is by exact permutation of **whole invocations** between
the arms, never of conversations: two conversations inside one
invocation are not exchangeable with two inside another, for the same
reason `sampling_margin` takes the run rather than the reply as the
independent unit. For the spread tests each rule's values are centred
on their own arm's mean first, so a difference in level cannot
masquerade as a difference in spread.

**A "no difference" result is unreadable without the size of thing the
test could have seen**, so every rule carries the ratio that would have
registered and so does the panel. That is the same discipline
`ShiftEstimate.source` and `RuleComparison.margin_source` apply to a
figure's provenance, applied here to its power.

Usage::

    left = load_style_history(path, fingerprint="30d996ff9270")
    right = load_style_history(other, fingerprint="526788c30730")
    print(compare_arms(left, right, "guarded", "unguarded").summary())
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from statistics import fmean, stdev

from agents7.eval.style import RunScore
from agents7.eval.style_history import StyleRunRecord
from agents7.exceptions import AgentConfigurationError

# Two invocations per arm is the floor, for the same reason
# `MIN_BASELINE_RUNS` is: it is the smallest number that can express
# any spread at all.
MIN_ARM_RUNS = 2

# Finer than the instrument resolves — one violation in the 48 replies
# a rule gets across three short conversations is 2.1 points, and an
# invocation's hundred-odd replies put the floor near 0.9 — so this
# never masks a real difference. It stops an arm that happened to
# agree exactly from having a spread of zero and blowing a ratio up to
# infinity.
SPREAD_FLOOR = 0.005

# A rule has to fire somewhere to say anything about variance. A rule
# with no violations either way has a spread of zero both ways, which
# is a ratio of exactly one carrying no information — and a dozen of
# those drag any panel median to a dead heat regardless of what the
# live rules did. Found the hard way: the first version of this
# comparison reported 1.00x at p=1.000 over fifteen of one agent's
# rules, seven of which never fired.
LIVE_RATE = 0.01


def _rate(record: StyleRunRecord, rule_id: str) -> float | None:
    """One invocation's rate for one rule, or None if unexercised."""
    rule = record.rule(rule_id)
    return None if rule is None else rule.rate


def _runs(record: StyleRunRecord, rule_id: str) -> tuple[RunScore, ...]:
    """One invocation's runs for one rule, the unexercised dropped."""
    rule = record.rule(rule_id)
    if rule is None:
        return ()
    return tuple(run for run in rule.runs if run.applicable > 0)


def _excess_variance(runs: Sequence[RunScore]) -> float | None:
    """Conversation-level variance beyond what binomial noise explains.

    For a run of ``m`` replies, ``Var(v/m) = tau^2 (1 - 1/m) +
    mu(1-mu)/m``. So the observed spread is corrected for the coin-flip
    part and what is left is the real conversation-to-conversation
    wobble. Estimated within each script and pooled by degrees of
    freedom.

    **The correction is not optional here.** Two arms with different
    mean rates carry different amounts of binomial noise, so comparing
    their raw run spreads would report a difference in *level* as a
    difference in wobble — which is precisely the confusion the three
    tables exist to keep apart.

    Related to `_cluster_error`, and not the same thing. That one
    answers how far a pooled rate sits from its own truth, so it
    returns a standard error of a mean and keeps the binomial part,
    which belongs in a margin. This one answers how much two
    conversations disagree with each other, so it returns a variance
    component with the binomial part removed. Both stratify by script,
    for the same reason: two scripts are two stimuli with two different
    true rates, and that gap is structure rather than noise.

    :param runs: The runs of one arm, across every invocation in it.
    :return: The estimate, or None if no script had two runs. May be
        negative, which means those conversations agreed *more* closely
        than binomial noise alone predicts.
    """
    by_script: dict[str, list[RunScore]] = {}
    for run in runs:
        by_script.setdefault(run.script, []).append(run)
    parts, weights = [], []
    for group in by_script.values():
        if len(group) < 2:
            continue
        rates = [run.violations / run.applicable for run in group]
        mean = fmean(rates)
        inverse = fmean([1.0 / run.applicable for run in group])
        denominator = 1.0 - inverse
        if denominator <= 0:
            continue
        binomial = mean * (1.0 - mean) * inverse
        parts.append((stdev(rates) ** 2 - binomial) / denominator)
        weights.append(len(group) - 1)
    if not parts:
        return None
    total = sum(part * weight for part, weight in zip(parts, weights))
    return total / sum(weights)


def _signed_sd(variance: float) -> float:
    """A variance as an sd, keeping the sign of an underdispersed one."""
    return math.copysign(abs(variance) ** 0.5, variance)


def _log_ratio(left: float, right: float) -> float:
    """``log(left/right)``, both floored at `SPREAD_FLOOR`."""
    return math.log(max(left, SPREAD_FLOOR) / max(right, SPREAD_FLOOR))


def _splits(
    left: int, right: int
) -> tuple[list[tuple[int, ...]], list[list[int]]]:
    """Every way to deal the invocations back into two groups.

    Unequal groups are fine — the arms of a before/after are whatever
    sizes they happen to be, and the null is still "the label carries
    no information", so the same enumeration applies.

    :param left: How many invocations the left arm has.
    :param right: How many the right arm has.
    :return: The picks, and each pick's complement.
    """
    total = left + right
    picks = list(combinations(range(total), left))
    rest = [[i for i in range(total) if i not in set(p)] for p in picks]
    return picks, rest


def _p_value(observed: float, nulls: Sequence[float]) -> float:
    """Two-sided permutation p: how often the null was as extreme."""
    extreme = sum(abs(value) >= abs(observed) - 1e-12 for value in nulls)
    return extreme / len(nulls)


def _threshold(nulls: Sequence[float]) -> float:
    """The 95th percentile of a null, as a ratio rather than a log."""
    ordered = sorted(abs(value) for value in nulls)
    return math.exp(ordered[int(0.95 * len(ordered))])


@dataclass(frozen=True)
class RuleArms:
    """One rule measured in both arms, three ways.

    :param rule_id: The rule.
    :param left_rate: Its mean rate across the left arm's
        invocations.
    :param right_rate: The same for the right arm.
    :param level_p: Permutation p for the difference in those means.
    :param left_spread: Standard deviation of the left arm's
        whole-invocation rates — the quantity a band is built from.
    :param right_spread: The same for the right arm.
    :param spread_p: Permutation p for the ratio of those two.
    :param spread_threshold: The ratio that would have registered at
        p < 0.05, read off this rule's own permutation null. **A "not
        significant" is unreadable without it**, since it is the
        difference between having ruled something out and having been
        unable to see it.
    :param left_tau: Conversation-level spread in the left arm, as a
        signed standard deviation with binomial noise removed. None
        when no script had two runs to disagree.
    :param right_tau: The same for the right arm.
    :param tau_p: Permutation p for the difference in those, or None
        when either arm has no estimate.
    :param fires: Whether the rule was violated at all, either way —
        the bar for its level being worth reading.
    :param live: Whether it also varies enough in both arms for a
        spread ratio to carry information. Only live rules enter the
        panel.
    """

    rule_id: str
    left_rate: float
    right_rate: float
    level_p: float
    left_spread: float
    right_spread: float
    spread_p: float
    spread_threshold: float
    left_tau: float | None
    right_tau: float | None
    tau_p: float | None
    fires: bool
    live: bool

    @property
    def level_diff(self) -> float:
        """Left's mean rate minus right's."""
        return self.left_rate - self.right_rate

    @property
    def spread_ratio(self) -> float:
        """Left's invocation spread over right's, both floored."""
        return math.exp(_log_ratio(self.left_spread, self.right_spread))

    @property
    def tau_ratio(self) -> float | None:
        """Left's conversation spread over right's.

        None when either side is missing or came out negative, which
        makes it underdispersed rather than small — a ratio of two
        such figures means nothing.
        """
        if self.left_tau is None or self.right_tau is None:
            return None
        left, right = _signed_sd(self.left_tau), _signed_sd(self.right_tau)
        if left <= 0 or right <= 0:
            return None
        return left / right


@dataclass(frozen=True)
class DriftRow:
    """One rule's rate in held-back invocations against fresh ones.

    A difference here is the hour changing — a provider serving a
    different model, a different load — rather than anything the
    comparison is about.

    :param rule_id: The rule.
    :param earlier: Its mean rate across the held-back invocations.
    :param fresh: Its mean rate across the ones that were compared.
    """

    rule_id: str
    earlier: float
    fresh: float

    @property
    def diff(self) -> float:
        """Earlier minus fresh."""
        return self.earlier - self.fresh


@dataclass(frozen=True)
class ArmComparison:
    """Two sets of invocations read against each other.

    :param left_name: What the left arm is, in the operator's words.
    :param right_name: What the right arm is.
    :param left_count: Invocations compared on the left, after any
        trim.
    :param right_count: The same on the right.
    :param permutations: How many exact splits the p-values are over.
    :param rules: One `RuleArms` per rule scored in both arms.
    :param median_spread_ratio: The median invocation-spread ratio
        across the live rules, or None when none are live.
    :param panel_p: Permutation p for that median.
    :param panel_threshold: The consistent ratio that would have
        registered at p < 0.05.
    :param left_moved_more: How many live rules had the larger
        invocation spread on the left.
    :param held_back_left: Invocations trimmed off the left arm.
    :param held_back_right: The same on the right.
    :param drift_left: The held-back left invocations against the
        compared ones, per rule. Empty when nothing was held back.
    :param drift_right: The same on the right.
    """

    left_name: str
    right_name: str
    left_count: int
    right_count: int
    permutations: int
    rules: tuple[RuleArms, ...]
    median_spread_ratio: float | None
    panel_p: float | None
    panel_threshold: float | None
    left_moved_more: int
    held_back_left: int = 0
    held_back_right: int = 0
    drift_left: tuple[DriftRow, ...] = ()
    drift_right: tuple[DriftRow, ...] = ()

    @property
    def firing(self) -> tuple[RuleArms, ...]:
        """Rules violated at all, either way."""
        return tuple(rule for rule in self.rules if rule.fires)

    @property
    def live(self) -> tuple[RuleArms, ...]:
        """Rules that also vary enough for a spread ratio to mean
        anything — the panel's members."""
        return tuple(rule for rule in self.rules if rule.live)

    def summary(self) -> str:
        """The three tables, the panel, and any drift.

        :return: Multi-line string.
        """
        lines = [
            f"{self.left_count} {self.left_name} vs "
            f"{self.right_count} {self.right_name}, "
            f"{self.permutations} exact permutations."
        ]
        for count, name in (
            (self.held_back_left, self.left_name),
            (self.held_back_right, self.right_name),
        ):
            if count:
                lines.append(f"{count} earlier {name} held back — see DRIFT.")
        lines += self._level_table()
        lines += self._spread_table()
        lines += self._tau_table()
        lines += self._panel()
        lines += self._drift_table(self.drift_left, self.left_name)
        lines += self._drift_table(self.drift_right, self.right_name)
        return "\n".join(lines)

    def _level_table(self) -> list[str]:
        """The mean rate each way."""
        head = f"{'rule':<31}{'left':>9}{'right':>10}{'diff':>8}{'p':>7}"
        lines = ["", "LEVEL — mean rate each way", head, "-" * len(head)]
        for rule in self.firing:
            lines.append(
                f"{rule.rule_id:<31}{rule.left_rate:>8.1%}"
                f"{rule.right_rate:>10.1%}{rule.level_diff:>+8.1%}"
                f"{rule.level_p:>7.3f}"
            )
        return lines

    def _spread_table(self) -> list[str]:
        """The invocation spread each way, with what it could see."""
        head = (
            f"{'rule':<31}{'sd L':>9}{'sd R':>10}{'ratio':>8}"
            f"{'p':>7}{'needs':>8}"
        )
        lines = [
            "",
            "INVOCATION SPREAD — what a band is built from",
            head,
            "-" * len(head),
        ]
        for rule in self.live:
            lines.append(
                f"{rule.rule_id:<31}{rule.left_spread:>8.1%}"
                f"{rule.right_spread:>10.1%}{rule.spread_ratio:>8.2f}"
                f"{rule.spread_p:>7.3f}{rule.spread_threshold:>8.2f}"
            )
        return lines

    def _tau_table(self) -> list[str]:
        """The conversation spread each way."""
        head = f"{'rule':<31}{'tau L':>9}{'tau R':>10}{'ratio':>8}{'p':>7}"
        lines = [
            "",
            "CONVERSATION SPREAD — beyond binomial noise, as an sd",
            head,
            "-" * len(head),
        ]
        for rule in self.live:
            if rule.tau_p is None:
                continue
            ratio = rule.tau_ratio
            lines.append(
                f"{rule.rule_id:<31}"
                f"{_signed_sd(rule.left_tau):>8.1%}"
                f"{_signed_sd(rule.right_tau):>10.1%}"
                f"{'-' if ratio is None else f'{ratio:.2f}':>8}"
                f"{rule.tau_p:>7.3f}"
            )
        return lines

    def _panel(self) -> list[str]:
        """The median across the live rules, and its power."""
        if self.median_spread_ratio is None:
            return []
        live = len(self.live)
        return [
            "",
            f"PANEL over the {live} rules that fire — the per-rule "
            f"figures share invocations, so a sign test across them "
            f"would overstate its confidence; permuting whole "
            f"invocations and recomputing the median does not.",
            f"  Left moved more on {self.left_moved_more} of {live}.",
            f"  Median spread ratio {self.median_spread_ratio:.2f}x, "
            f"p = {self.panel_p:.3f}.",
            f"  A consistent difference above "
            f"{self.panel_threshold:.2f}x would have registered at "
            f"p<0.05.",
        ]

    @staticmethod
    def _drift_table(rows: Sequence[DriftRow], name: str) -> list[str]:
        """Held-back invocations against the compared ones."""
        if not rows:
            return []
        head = f"{'rule':<31}{'earlier':>9}{'fresh':>10}{'diff':>8}"
        lines = [
            "",
            f"DRIFT — earlier {name} vs the interleaved ones",
            head,
            "-" * len(head),
        ]
        for row in rows:
            lines.append(
                f"{row.rule_id:<31}{row.earlier:>8.1%}"
                f"{row.fresh:>10.1%}{row.diff:>+8.1%}"
            )
        return lines


def _drift_rows(
    earlier: Sequence[StyleRunRecord],
    fresh: Sequence[StyleRunRecord],
) -> tuple[DriftRow, ...]:
    """Per-rule rates in the held-back invocations against the fresh.

    :param earlier: The invocations the trim removed.
    :param fresh: The ones that were compared.
    :return: One row per firing rule scored in both, or nothing when
        no invocations were held back.
    """
    if not earlier or not fresh:
        return ()
    rows = []
    for rule_id in [rule.id for rule in fresh[0].rules]:
        old = [_rate(record, rule_id) for record in earlier]
        new = [_rate(record, rule_id) for record in fresh]
        if any(value is None for value in old + new):
            continue
        if max(fmean(old), fmean(new)) < LIVE_RATE:
            continue
        rows.append(
            DriftRow(rule_id=rule_id, earlier=fmean(old), fresh=fmean(new))
        )
    return tuple(rows)


def compare_arms(
    left: Sequence[StyleRunRecord],
    right: Sequence[StyleRunRecord],
    left_name: str = "left",
    right_name: str = "right",
    *,
    trim: bool = True,
) -> ArmComparison:
    """Compare two sets of invocations on level and on two spreads.

    Every p-value is an exact permutation of **whole invocations**
    between the arms. Conversations are never permuted: two inside one
    invocation are not exchangeable with two inside another, and a test
    that treats them as though they were reports a confidence the data
    cannot support.

    :param left: One arm's records, oldest first — typically
        `load_style_history(path, fingerprint=...)`.
    :param right: The other arm's records, oldest first.
    :param left_name: What the left arm is, for the rendered tables.
    :param right_name: What the right arm is.
    :param trim: Cut the longer arm down to the shorter one's size,
        keeping the newest, so two interleaved arms sit in the same
        window of time and a change in what the provider is serving
        cannot masquerade as the difference under test. Right for two
        arms that were deliberately interleaved; wrong for a
        before/after, where the arms are whatever sizes they are and
        trimming only throws away invocations. What it removes is
        reported as drift rather than discarded silently.
    :return: The comparison.
    :raises AgentConfigurationError: If either arm has fewer than
        `MIN_ARM_RUNS` invocations once trimmed, or if the left arm
        carries no rules to compare.
    """
    held_left = held_right = 0
    earlier_left: Sequence[StyleRunRecord] = ()
    earlier_right: Sequence[StyleRunRecord] = ()
    if trim:
        keep = min(len(left), len(right))
        held_left, held_right = len(left) - keep, len(right) - keep
        earlier_left, earlier_right = left[:held_left], right[:held_right]
        left, right = left[len(left) - keep :], right[len(right) - keep :]
    if len(left) < MIN_ARM_RUNS or len(right) < MIN_ARM_RUNS:
        raise AgentConfigurationError(
            f"compare_arms: each arm needs at least {MIN_ARM_RUNS} "
            f"invocations to have any spread at all, got {len(left)} "
            f"{left_name} and {len(right)} {right_name}."
        )
    if not left[0].rules:
        raise AgentConfigurationError(
            f"compare_arms: the first {left_name} invocation carries "
            f"no rules, so there is nothing to compare."
        )

    picks, rest = _splits(len(left), len(right))
    rules: list[RuleArms] = []
    spread_nulls: list[list[float]] = []
    for rule_id in [rule.id for rule in left[0].rules]:
        scored = _compare_one(rule_id, left, right, picks, rest)
        if scored is not None:
            rules.append(scored[0])
            spread_nulls.append(scored[1])
    return ArmComparison(
        left_name=left_name,
        right_name=right_name,
        left_count=len(left),
        right_count=len(right),
        permutations=len(picks),
        rules=tuple(rules),
        **_panel_stats(rules, spread_nulls, len(picks)),
        held_back_left=held_left,
        held_back_right=held_right,
        drift_left=_drift_rows(earlier_left, left),
        drift_right=_drift_rows(earlier_right, right),
    )


def _compare_one(
    rule_id: str,
    left: Sequence[StyleRunRecord],
    right: Sequence[StyleRunRecord],
    picks: Sequence[tuple[int, ...]],
    rest: Sequence[Sequence[int]],
) -> tuple[RuleArms, list[float]] | None:
    """One rule's three comparisons, or None if any arm never ran it.

    Returns the rule's invocation-spread permutation null alongside
    it. The panel recomputes its median under each of those same
    splits, and the null is thousands of floats per rule — working
    state, not something to hang off a public record.
    """
    left_rates = [_rate(record, rule_id) for record in left]
    right_rates = [_rate(record, rule_id) for record in right]
    if any(value is None for value in left_rates + right_rates):
        return None
    left_mean, right_mean = fmean(left_rates), fmean(right_rates)
    left_sd, right_sd = stdev(left_rates), stdev(right_rates)

    raw = left_rates + right_rates
    centred = [value - left_mean for value in left_rates]
    centred += [value - right_mean for value in right_rates]
    level_nulls = [
        fmean([raw[i] for i in pick]) - fmean([raw[i] for i in other])
        for pick, other in zip(picks, rest)
    ]
    spread_nulls = [
        _log_ratio(
            stdev([centred[i] for i in pick]),
            stdev([centred[i] for i in other]),
        )
        for pick, other in zip(picks, rest)
    ]

    every = [_runs(record, rule_id) for record in list(left) + list(right)]
    left_tau = _excess_variance(
        [r for group in every[: len(left)] for r in group]
    )
    right_tau = _excess_variance(
        [r for group in every[len(left) :] for r in group]
    )
    tau_nulls: list[float] = []
    if left_tau is not None and right_tau is not None:
        for pick, other in zip(picks, rest):
            a = _excess_variance([r for i in pick for r in every[i]])
            b = _excess_variance([r for i in other for r in every[i]])
            if a is not None and b is not None:
                tau_nulls.append(_signed_sd(a) - _signed_sd(b))

    fires = max(left_mean, right_mean) >= LIVE_RATE
    live = fires and min(left_sd, right_sd) > SPREAD_FLOOR
    scored = RuleArms(
        rule_id=rule_id,
        left_rate=left_mean,
        right_rate=right_mean,
        level_p=_p_value(left_mean - right_mean, level_nulls),
        left_spread=left_sd,
        right_spread=right_sd,
        spread_p=_p_value(_log_ratio(left_sd, right_sd), spread_nulls),
        spread_threshold=_threshold(spread_nulls),
        left_tau=left_tau,
        right_tau=right_tau,
        tau_p=(
            _p_value(_signed_sd(left_tau) - _signed_sd(right_tau), tau_nulls)
            if tau_nulls
            else None
        ),
        fires=fires,
        live=live,
    )
    return scored, spread_nulls


def _panel_stats(
    rules: Sequence[RuleArms],
    spread_nulls: Sequence[Sequence[float]],
    permutations: int,
) -> dict:
    """The median spread ratio across the live rules, and its power.

    The per-rule figures share invocations, so a sign test across them
    would overstate its confidence. Permuting whole invocations and
    recomputing the median does not — which is why the panel's null is
    built from the same splits the per-rule ones used, rather than from
    the per-rule p-values.

    **Only the live rules enter.** A rule with no violations either way
    has a spread of zero on both sides, which is a ratio of exactly one
    carrying no information, and enough of those drag the median to a
    dead heat whatever the live rules did.

    :param rules: Every rule compared, live and not.
    :param spread_nulls: Each rule's invocation-spread null, in the
        same order as `rules`.
    :param permutations: How many splits the nulls were built over.
    :return: The panel fields, for `ArmComparison`.
    """
    live = [
        (rule, nulls) for rule, nulls in zip(rules, spread_nulls) if rule.live
    ]
    if not live:
        return {
            "median_spread_ratio": None,
            "panel_p": None,
            "panel_threshold": None,
            "left_moved_more": 0,
        }
    observed = sorted(
        _log_ratio(rule.left_spread, rule.right_spread) for rule, _ in live
    )[len(live) // 2]
    nulls = []
    for index in range(permutations):
        across = sorted(rule_nulls[index] for _, rule_nulls in live)
        nulls.append(across[len(across) // 2])
    return {
        "median_spread_ratio": math.exp(observed),
        "panel_p": _p_value(observed, nulls),
        "panel_threshold": _threshold(nulls),
        "left_moved_more": sum(
            1 for rule, _ in live if rule.left_spread > rule.right_spread
        ),
    }


__all__ = [
    "LIVE_RATE",
    "MIN_ARM_RUNS",
    "SPREAD_FLOOR",
    "ArmComparison",
    "DriftRow",
    "RuleArms",
    "compare_arms",
]
