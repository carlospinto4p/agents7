"""Persisting style-contract runs so they can be compared.

`run_style_contract` measures one invocation. It cannot tell you
whether today's number differs from last week's, because it has
never seen last week's — and a check that cannot compare across
time is not a regression check, whatever else it is.

**The failure that produced this module.** On 30th August 2026 the
same chat agent config, the same scripted counterpart and the same
code were measured three times. One rule scored 23%, then 70%, then
41%. Ten runs inside a single invocation agreed with each other
well enough to look like a measurement; the invocations did not
agree with each other at all. A two-proportion test on the last two
gives z ≈ 3.9, so this is not binomial noise at that sample size —
something outside the config varies, and averaging more runs within
one invocation does not remove it.

So the unit that has to be persisted is the **invocation**, and the
comparison that matters is invocation against invocation.

**What that means for reading a diff.** A single earlier invocation
is not a baseline. Two invocations of an unchanged config give a
band, and only a rate outside that band is worth calling a change.
`compare_to_history` refuses to render a verdict below two prior
invocations of the same configuration, rather than reporting a
difference it cannot distinguish from the variation it has not
measured yet.

**Configuration identity is computed, not claimed.** Each record
carries a fingerprint over the agent's resolved system prompt, the
contract's rules and scripts, and the guard that took part in the
run. A baseline is built only from records sharing it, so editing a
persona silently starts a fresh baseline instead of being compared
against a band that described a different agent.

**A rule's history is its own, since v9.1.0.** Comparison keys on
`configuration_key`, which is the fingerprint without the rule
ids, and each rule's band holds only runs scored under that rule's
own `definition` — the check and its arguments. So adding a rule
leaves every other rule's band where it was, and changing what a
rule checks under an unchanged id starts that one rule afresh.
Before, the first threw away every band and the second was
invisible: one agent's 73-cent baseline, rebuilt that morning, was
one new rule away from being unspendable.

**The guard is part of that identity, and participation is read
from the run.** A guard rewrites replies before they are scored, so
a run made without one is measuring something else: dropping
one agent's guard moved `ends_on_a_question` from 50% to 56% and
roughly doubled three rarer rules. Whether it took part cannot be
read off the agent, because the way these runs are made unguarded
is to replace the guard's check with one that accepts without
calling a model — the agent still carries a fully configured guard.
So the fingerprint reads the calls the run actually made. Before it
did, five guarded and five unguarded invocations of one agent all
carried the fingerprint `cfcfa8d5d117` and were one file away from
sharing a band.

Usage::

    report = run_style_contract(build_agent, contract)
    record = record_style_run(report, contract, build_agent(),
                              subject="my_agent v0.1.19")
    history = load_style_history("data/style_history.jsonl")
    print(compare_to_history(record, history).summary())
    append_style_run(record, "data/style_history.jsonl")
"""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from functools import cache
from math import ceil, exp, lgamma, log
from pathlib import Path
from statistics import NormalDist, stdev

from agents7.agents.chatter import Chatter
from agents7.agents.context_guard import ContextGuard
from agents7.eval.style import RunScore, StyleContract, StyleReport
from agents7.exceptions import AgentConfigurationError

# Two prior invocations is the floor for a band, not a
# recommendation: it is the smallest number that can express any
# spread at all. A band drawn from two points is as likely to be
# accidentally narrow as accidentally wide, and the count is
# reported alongside every verdict so a reader can discount it.
MIN_BASELINE_RUNS = 2

# Four prior invocations is where their spread becomes usable as
# the bar. Below it the estimate fails in both directions and the
# within-invocation figure is the safer proxy: walked forward over
# one agent's six gate invocations, two priors gave a spread of
# exactly 0.0% on `no_trailing_laugh` and
# `no_greeting_after_the_opening` — the two happened to agree, so a
# single later event would read as a regression — while a Student-t
# correction for that same sample gave 91.7% on
# `one_message_when_dry` and 277.7% on `opens_in_lower_case`, which
# is not a bar at all. By four both failures are gone.
MIN_SPREAD_PRIORS = 4

# But a standard deviation from four values carries three degrees of
# freedom, so its lower tail is wide: four priors that happen to
# agree give a margin far below how much the rule really moves. On
# 15th September 2026 one agent's gate failed an unchanged config on
# `copies_an_example` at 18.4% against a band of 14.9%-16.5%, with a
# margin of 1.4 points. So the spread is read at the upper end of its
# own confidence interval rather than at its point estimate, which
# widens a short history's margin and fades as priors accumulate.
#
# Sized against that gate's whole history, replayed the same day
# (123 invocations, 328 rule checks judged on four or more priors):
# the point estimate failed the gate six times on unchanged
# configurations, an 80% bound once and a 90% bound never. The 90%
# bound took the median margin from 9.2 to 16.7 points — sensitivity
# given up on purpose, and still under the 33- and 40-point
# regressions the contract was built to catch.
SPREAD_CONFIDENCE = 0.90

# A rule whose prior invocations broke it fewer times than this, on
# average, is reported and never judged. At that count chance alone
# puts a rule outside a band built mostly from zeros: one agent's
# gate failed `no_trailing_laugh` twice at 3/111 on an unchanged
# config, with its three laughs in three and in two different
# conversations. A rule checked this rarely has no band that can
# tell a regression from bad luck, and fifteen of them checked
# every version make bad luck a regular event.
#
# The price, accepted on 13th September 2026: a quiet rule that
# jumps is not caught either. One agent's v0.2.9 gate scored
# `one_message_when_dry` at 10/48 after priors of 1, 1, 0 and 0 of
# 48, and that now passes, visible only in the summary.
RARE_VIOLATIONS_PER_INVOCATION = 1.0


class Verdict(StrEnum):
    """What comparing one rule against its history showed.

    :cvar RARE: Its prior invocations broke it fewer than
        `RARE_VIOLATIONS_PER_INVOCATION` times each on average, so
        its rate is shown and not judged. Never counted as a change.
    """

    NO_BASELINE = "no_baseline"
    WITHIN_BASELINE = "within_baseline"
    ABOVE_BASELINE = "above_baseline"
    BELOW_BASELINE = "below_baseline"
    UNEXERCISED = "unexercised"
    RARE = "rare"


class MarginSource(StrEnum):
    """Where a rule's margin was estimated from.

    Strongest first. The distinction is shown wherever a margin is,
    because the three differ by more than a rounding: measured over
    one agent's six gate invocations, the `RUNS` figure ran two to
    three times wider than the movement `PRIORS` measures on four
    of five exercised rules, and narrower than it on the fifth.

    :cvar PRIORS: The spread of this configuration's own earlier
        invocations — draws of exactly the quantity being judged.
    :cvar RUNS: How much the conversations inside this one
        invocation disagreed. A proxy, used until there are
        `MIN_SPREAD_PRIORS` priors to measure the real thing.
    :cvar REPLIES: The binomial error of the pooled replies, which
        assumes they were drawn independently. They are not — a run
        is one conversation — so this understates the uncertainty,
        and it is only reached by a record that kept no per-run
        rates at all.
    """

    PRIORS = "priors"
    RUNS = "runs"
    REPLIES = "replies"


_MARGIN_MARKS = {
    MarginSource.PRIORS: "",
    MarginSource.RUNS: "*",
    MarginSource.REPLIES: "**",
}

_MARGIN_FOOTNOTES = {
    MarginSource.RUNS: (
        f"  *  = fewer than {MIN_SPREAD_PRIORS} prior invocations, "
        f"so this margin is estimated from how much the "
        f"conversations inside this one run disagreed, not from how "
        f"much the score has actually moved between invocations. On "
        f"one agent that proxy ran two to three times too wide on "
        f"four rules and too narrow on a fifth, so read it as a "
        f"placeholder: it becomes a measurement at "
        f"{MIN_SPREAD_PRIORS} priors."
    ),
    MarginSource.REPLIES: (
        "  ** = this record kept no per-run rates, so its margin "
        "assumes the replies were drawn independently. They are "
        "not — a run is one conversation — so the real margin is "
        "wider than shown and a change flagged here may be none."
    ),
}


@dataclass(frozen=True)
class RuleRecord:
    """One rule's score in one invocation, as persisted.

    :param id: The rule's id.
    :param budget: The rate it tolerated at the time. Kept because
        a budget that was later tightened would otherwise make an
        old record look like a failure it never was.
    :param violations: Replies that broke it, across every run.
    :param applicable: Replies the rule applied to.
    :param runs: One entry per play of one script, carrying that
        run's label and its own counts. **The measurement
        surface's shape, not only its fields.** `RuleResult`
        already knew which script each run came from; persisting
        the rates alone threw that away, so once a run was in the
        history the cold conversations could not be told from the
        warm ones — and `sampling_margin` then counted the gap
        between two different scripts as if it were noise.
    :param run_rates: Each run's rate, flattened. Derived from
        `runs` when those are present, and carried on its own by a
        record written before they existed — which is the only
        reason it is still a field of its own. It is written to
        every line so a reader can compare an old line against a
        new one without knowing which shape it is in.
    :param definition: `StyleRule.definition` at the time — what
        the check measured. A band only holds runs with the same
        one. None on a rule built around a bare callable, and on
        every record written before v9.1.0.
    """

    id: str
    budget: float | None
    violations: int
    applicable: int
    # `run_rates` keeps its position so a record built positionally
    # before `runs` existed still means what it meant. The
    # docstring order above is the order that reads correctly.
    run_rates: tuple[float, ...] = ()
    runs: tuple[RunScore, ...] = ()
    definition: str | None = None

    def __post_init__(self) -> None:
        """Flatten `runs` into `run_rates` when it was not given."""
        if self.runs and not self.run_rates:
            object.__setattr__(
                self,
                "run_rates",
                tuple(r.rate for r in self.runs if r.rate is not None),
            )

    @property
    def rate(self) -> float | None:
        """Violations over applicable, or None if unexercised."""
        if self.applicable == 0:
            return None
        return self.violations / self.applicable

    @property
    def by_script(self) -> tuple[RunScore, ...]:
        """The runs added up per script, in the order scripts ran.

        Each entry's `label` is the script id alone. **A pooled rate
        can hide a move in one script**: one agent mixes a cold and a
        warm script in a fixed proportion, so on a rule the two
        disagree about, a six-point pooled move can be an
        eleven-point move in one and a two-point move in the other.
        v0.2.2's casing regression was invisible until it was split.

        :return: One `RunScore` per script. Empty for a record
            written before runs were labelled.
        """
        totals: dict[str, tuple[int, int]] = {}
        for run in self.runs:
            violations, applicable = totals.get(run.script, (0, 0))
            totals[run.script] = (
                violations + run.violations,
                applicable + run.applicable,
            )
        return tuple(
            RunScore(label=script, violations=v, applicable=a)
            for script, (v, a) in totals.items()
        )


@dataclass(frozen=True)
class StyleRunRecord:
    """One whole invocation of a contract, as persisted.

    :param recorded_at: ISO-8601 UTC timestamp.
    :param subject: What was measured, in the operator's words —
        typically the agent's package version. Free text, for
        reading; `fingerprint` is what comparison actually keys on.
    :param fingerprint: Hash over the agent's resolved system
        prompt, the contract's rules and scripts, and the guard
        that took part. Two records are comparable when, and only
        when, these match.
    :param agent: The agent's name.
    :param agent_version: Its version.
    :param model_id: The ``provider::model`` the agent was
        configured with — what was asked for, which is not always
        what answered. See `answered_by`.
    :param repeats: How many times each script was played.
    :param replies: Replies scored.
    :param calls: LLM calls made, guard checks included.
    :param cost_cents: What the invocation cost, or None when
        nothing was priced — never 0.0 for an unpriced run.
    :param rules: One `RuleRecord` per rule.
    :param guard_model_id: The ``provider::model`` the guard was
        configured with, or None when no guard took part. Recorded
        because the guard is frequently a different model at a
        different price — on one agent it is 59% of an
        invocation's bill — so `model_id` alone cannot say which
        model produced that half. Like `model_id`, this is what was
        asked for; `guard_answered_by` is what answered. A record
        written before this field existed also reads None; those
        records are told apart by their fingerprint, which no
        longer matches a guarded run's.
    :param lost_runs: Conversations that raised before finishing
        and were therefore not scored. Recorded because a partial
        invocation measured fewer runs than its `repeats` implies,
        so its rates carry more uncertainty than that number
        suggests — and because a run going missing explains a move
        in the figures as readily as a config edit does.
    :param answered_by: The models that actually answered the
        agent's own calls, as the provider reported them
        (`CallRecord.model`), sorted. Usually one; more than one
        means a fallback answered part of the run. Empty when no
        call reported a model, including every record written
        before v9.0.0. A provider can move an alias without the
        config changing — DeepSeek moved `deepseek-chat` from
        `deepseek-v4-flash` to `deepseek-flash` on 10th September
        2026, and one agent's casing rule went from 75% to 10% on
        an unchanged config — so this, not `model_id`, is what
        says two invocations measured the same model.
    :param config_key: `configuration_key` — the fingerprint
        without the rule ids. What comparison keys on when both
        records carry one, so adding a rule does not reset every
        band. None on every record written before v9.1.0, which
        is then compared on `fingerprint` as before.
    :param guard_answered_by: The models that actually answered the
        guard's calls, as the provider reported them, sorted. Empty
        when no guard took part, when none of its calls reported a
        model, and on every record written before v9.3.0. The same
        alias move `answered_by` exists for happens under a guard
        too — one agent's guard asks for `deepseek-chat` — and the
        guard rewrites replies before they are scored, so a guard
        answered by a different model measures something else.
    :param run_variants: The text each scored run was shown for
        each of the agent's `prompt_variants`, keyed by run label,
        as in `StyleReport.variants`. Kept once per run rather than
        on every rule's scores. Empty for an agent with no
        variants, and on every record written before v9.13.0 —
        which cannot say what its runs were shown.
    """

    recorded_at: str
    subject: str
    fingerprint: str
    agent: str
    agent_version: int
    model_id: str
    repeats: int
    replies: int
    calls: int
    cost_cents: float | None
    rules: tuple[RuleRecord, ...]
    guard_model_id: str | None = None
    lost_runs: int = 0
    answered_by: tuple[str, ...] = ()
    config_key: str | None = None
    guard_answered_by: tuple[str, ...] = ()
    run_variants: Mapping[str, Mapping[str, str]] = field(
        default_factory=dict
    )

    def rule(self, rule_id: str) -> RuleRecord | None:
        """The record for one rule, or None if it carries none."""
        for record in self.rules:
            if record.id == rule_id:
                return record
        return None

    def by_variant(self, rule_id: str, name: str) -> dict[str, RunScore]:
        """One rule's score added up per text shown for *name*.

        Answers whether a rate moved with what the runs were shown
        rather than with the config. Each entry's `label` is the
        text. Runs with no text recorded for *name* are left out,
        so the totals cover only runs that can be attributed.

        :param rule_id: The rule to split.
        :param name: The `prompt_variants` name to split by.
        :return: Text to its summed score, in the order the texts
            first appear. Empty when the rule is absent or no run
            recorded *name*.
        """
        record = self.rule(rule_id)
        if record is None:
            return {}
        totals: dict[str, tuple[int, int]] = {}
        for run in record.runs:
            text = self.run_variants.get(run.label, {}).get(name)
            if text is None:
                continue
            violations, applicable = totals.get(text, (0, 0))
            totals[text] = (
                violations + run.violations,
                applicable + run.applicable,
            )
        return {
            text: RunScore(label=text, violations=v, applicable=a)
            for text, (v, a) in totals.items()
        }

    def to_dict(self) -> dict:
        """Plain dictionary, for one JSON line."""
        return {
            "recorded_at": self.recorded_at,
            "subject": self.subject,
            "fingerprint": self.fingerprint,
            "agent": self.agent,
            "agent_version": self.agent_version,
            "model_id": self.model_id,
            "repeats": self.repeats,
            "replies": self.replies,
            "calls": self.calls,
            "cost_cents": self.cost_cents,
            "rules": [
                {
                    "id": r.id,
                    "budget": r.budget,
                    "violations": r.violations,
                    "applicable": r.applicable,
                    "runs": [
                        {
                            "label": s.label,
                            "violations": s.violations,
                            "applicable": s.applicable,
                        }
                        for s in r.runs
                    ],
                    "run_rates": list(r.run_rates),
                    "definition": r.definition,
                }
                for r in self.rules
            ],
            "guard_model_id": self.guard_model_id,
            "lost_runs": self.lost_runs,
            "answered_by": list(self.answered_by),
            "config_key": self.config_key,
            "guard_answered_by": list(self.guard_answered_by),
            "run_variants": {
                label: dict(drawn)
                for label, drawn in self.run_variants.items()
            },
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "StyleRunRecord":
        """Rebuild a record from one persisted line."""
        return cls(
            recorded_at=raw["recorded_at"],
            subject=raw["subject"],
            fingerprint=raw["fingerprint"],
            agent=raw["agent"],
            agent_version=int(raw["agent_version"]),
            model_id=raw["model_id"],
            repeats=int(raw["repeats"]),
            replies=int(raw["replies"]),
            calls=int(raw["calls"]),
            cost_cents=raw.get("cost_cents"),
            rules=tuple(
                RuleRecord(
                    id=r["id"],
                    budget=r.get("budget"),
                    violations=int(r["violations"]),
                    applicable=int(r["applicable"]),
                    runs=tuple(
                        RunScore(
                            label=s["label"],
                            violations=int(s["violations"]),
                            applicable=int(s["applicable"]),
                        )
                        for s in r.get("runs", ())
                    ),
                    run_rates=tuple(r.get("run_rates", ())),
                    definition=r.get("definition"),
                )
                for r in raw.get("rules", ())
            ),
            guard_model_id=raw.get("guard_model_id"),
            lost_runs=int(raw.get("lost_runs", 0)),
            answered_by=tuple(raw.get("answered_by", ())),
            config_key=raw.get("config_key"),
            guard_answered_by=tuple(raw.get("guard_answered_by", ())),
            run_variants={
                label: dict(drawn)
                for label, drawn in (raw.get("run_variants") or {}).items()
            },
        )


def _guard_that_ran(
    agent: Chatter,
    report: StyleReport | None,
) -> ContextGuard | None:
    """The guard that took part in a run, if one did.

    Read from the run rather than from the agent, because the two
    disagree in the one case that matters. An unguarded run is made
    by replacing the guard's check with one that accepts without
    calling a model, so the agent still carries a fully configured
    `guard` and nothing about it says the guard sat the run out.
    The calls do: a guarded invocation of one agent makes about 800,
    an unguarded one exactly the 370 the chatter made.

    :param agent: The agent the run built.
    :param report: What the run produced. None means there is no
        run to read yet, and an agent that has a guard is then
        assumed to use it — right for a caller computing a
        fingerprint in order to load history before running.
    :return: The guard, or None when the agent has none or none
        took part.
    """
    guard = getattr(agent, "guard", None)
    if guard is None or report is None:
        return guard
    ran = any(
        call.agent_name == guard.name and call.agent_version == guard.version
        for call in report.calls
    )
    return guard if ran else None


def fingerprint_run(
    contract: StyleContract,
    agent: Chatter,
    report: StyleReport | None = None,
) -> str:
    """Identify what was measured, from the thing itself.

    Hashes the agent's resolved system prompt together with the
    contract's rule ids and script lines, and — when a guard took
    part — that guard's resolved prompt and model. Anything that
    would change the numbers for a reason other than chance
    changes the fingerprint, so a band is never carried across an
    edit that invalidated it.

    A version label would not do: it is asserted by whoever writes
    it, and the config can change without one moving.

    **Why a rule's budget is not in the hash, since v8.0.0.** It
    was until then, and that was over-inclusive by this function's
    own standard. A budget is read in exactly one place —
    `StyleRuleResult.passed` — to compare a rate that has already
    been measured. It cannot move a number, so it cannot be a
    reason to start a fresh band.

    Keying on it had a cost that was not obvious until someone
    tried to spend the baseline. Writing down the threshold a band
    had just justified changed the key and discarded that band, so
    the one edit guaranteed to leave every measurement untouched
    was also guaranteed to throw them all away. One agent sat on
    exactly that: five invocations, 78 cents and 28 minutes of
    measurement, unspendable without losing them.

    Nothing is lost by dropping it. Each record keeps the budget
    it was scored under, so a comparison can still say a threshold
    moved; and a rule that is *added* or *removed* still changes
    the hash, because the rule ids do.

    **The rule ids stay in, and comparison no longer keys on
    them, since v9.1.0.** They still say which rule set a record
    was scored under, and every stored key depends on them. But a
    rule added cannot move another rule's number, by the same
    argument as the budget, so `compare_to_history` keys on
    `configuration_key` — this hash without the ids — and holds
    each rule to its own `definition` instead.

    **Why the guard is in the hash and the chatter's own model is
    not.** The chatter's model is the subject under test, and a key
    that survives a model swap is what lets the move be seen. It is
    recorded on the record instead (`answered_by`, read from the
    calls rather than the config), and since v9.0.0
    `compare_to_history` builds a band only from invocations
    answered by the same model — naming the others it left out, so
    the swap is still reported rather than silently reset. The
    guard is apparatus. It rewrites replies before they are
    scored, so a run made under a different guard — or none — is
    measuring a different thing and starts its own band.

    **The guard's model in the hash is the configured one.** A
    provider moving an alias under the guard leaves it unchanged,
    exactly as it does for the chatter, so the model that answered
    the guard is recorded as `guard_answered_by` and filtered on by
    `compare_to_history` (since v9.3.0). Hashing the answering name
    instead would have changed every guarded key already stored.

    An agent that never had a guard hashes exactly as it did
    before the guard was folded in, so unguarded history keeps its
    baseline.

    :param contract: The contract that was run.
    :param agent: An agent built the way the run built its agents.
    :param report: What the run produced, so guard participation
        is read from the calls that were made. Omit it only when
        there is no run yet.
    :return: Twelve hex characters.
    """
    agent_parts, script_parts = _identity(contract, agent, report)
    # The rule ids sit between the two, where they always have, so
    # every stored fingerprint still reproduces.
    return _digest(
        agent_parts + [rule.id for rule in contract.rules] + script_parts
    )


def configuration_key(
    contract: StyleContract,
    agent: Chatter,
    report: StyleReport | None = None,
) -> str:
    """`fingerprint_run` without the contract's rule ids.

    The agent, the guard that ran and the scripts: everything that
    decides what replies a run produces. Which rules score those
    replies does not change them, so this is what
    `compare_to_history` keys on, and a rule added to a contract
    leaves the other rules' bands in place.

    :param contract: The contract that was run.
    :param agent: An agent built the way the run built its agents.
    :param report: What the run produced, as in `fingerprint_run`.
    :return: Twelve hex characters.
    """
    agent_parts, script_parts = _identity(contract, agent, report)
    return _digest(agent_parts + script_parts)


def _identity(
    contract: StyleContract,
    agent: Chatter,
    report: StyleReport | None,
) -> tuple[list[str], list[str]]:
    """The agent's and the scripts' parts of a run's identity.

    :return: ``(agent parts, script parts)``, in hashing order.
    """
    # The identity rather than the resolved prompt: each run builds
    # its own agent, and one with `prompt_variants` draws its own
    # texts, which would give every run a different fingerprint.
    agent_parts = [agent.system_prompt_identity]
    guard = _guard_that_ran(agent, report)
    if guard is not None:
        agent_parts.append(f"guard|{guard.system_prompt_identity}")
        agent_parts.append(f"guard-model|{guard.engine.model_id}")
    # A script's context is appended only when it has one, so every
    # fingerprint stored before scripts could carry context still
    # reproduces.
    script_parts = [
        f"{script.id}|{script.opens}|{'|'.join(script.lines)}"
        + "".join(
            f"|context|{name}={value}"
            for name, value in sorted(script.context.items())
        )
        for script in contract.scripts
    ]
    return agent_parts, script_parts


def _digest(parts: list[str]) -> str:
    """Twelve hex characters of SHA-256 over *parts*, one per line."""
    digest = hashlib.sha256("\n".join(parts).encode("utf-8"))
    return digest.hexdigest()[:12]


def record_style_run(
    report: StyleReport,
    contract: StyleContract,
    agent: Chatter,
    subject: str,
) -> StyleRunRecord:
    """Turn a finished report into a persistable record.

    :param report: What `run_style_contract` returned.
    :param contract: The contract it ran.
    :param agent: An agent built the way the run built its agents,
        for the fingerprint and the model id.
    :param subject: What was measured, in the operator's words.
    :return: The record.
    """
    guard = _guard_that_ran(agent, report)
    return StyleRunRecord(
        recorded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        subject=subject,
        fingerprint=fingerprint_run(contract, agent, report),
        agent=agent.name,
        agent_version=agent.version,
        model_id=agent.engine.model_id,
        repeats=contract.repeats,
        replies=len(report.replies),
        calls=len(report.calls),
        cost_cents=report.cost_cents,
        rules=tuple(
            RuleRecord(
                id=result.rule.id,
                budget=result.rule.budget,
                violations=result.violations,
                applicable=result.applicable,
                runs=result.per_run,
                definition=result.rule.definition,
            )
            for result in report.results
        ),
        guard_model_id=guard.engine.model_id if guard else None,
        lost_runs=len(report.lost),
        answered_by=_answered_by(report, agent),
        config_key=configuration_key(contract, agent, report),
        guard_answered_by=_answered_by(report, guard) if guard else (),
        run_variants={
            label: dict(drawn) for label, drawn in report.variants.items()
        },
    )


def _answered_by(
    report: StyleReport, agent: Chatter | ContextGuard
) -> tuple[str, ...]:
    """The models that answered *agent*'s own calls in a run, sorted.

    :param report: What the run produced.
    :param agent: The chatter or the guard whose calls to read.
    :return: The distinct models its calls reported. Calls that
        reported none are skipped.
    """
    return tuple(
        sorted(
            {
                call.model
                for call in report.calls
                if call.model
                and call.agent_name == agent.name
                and call.agent_version == agent.version
            }
        )
    )


def append_style_run(record: StyleRunRecord, path: str | Path) -> Path:
    """Append one record to a history file, as JSON Lines.

    Append-only on purpose. The history is the only thing that can
    say how much a rule moves when nothing changed, and rewriting
    it to tidy an inconvenient invocation destroys exactly that.

    :param record: The record to store.
    :param path: History file. Created with its parents if absent.
    :return: The path written to.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), ensure_ascii=False)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return target


def load_style_history(
    path: str | Path,
    fingerprint: str | None = None,
) -> tuple[StyleRunRecord, ...]:
    """Read a history file, oldest first.

    :param path: History file. A missing file is an empty history,
        not an error — the first run of a new contract has none.
    :param fingerprint: Keep only records with this fingerprint.
    :return: The records.
    :raises AgentConfigurationError: If a line will not parse.
    """
    target = Path(path)
    if not target.is_file():
        return ()
    records: list[StyleRunRecord] = []
    for number, line in enumerate(
        target.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(StyleRunRecord.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise AgentConfigurationError(
                f"{target}: line {number} is not a style run record ({exc})."
            ) from exc
    if fingerprint is not None:
        records = [r for r in records if r.fingerprint == fingerprint]
    return tuple(records)


@dataclass(frozen=True)
class RuleComparison:
    """One rule's current rate against what it has done before.

    :param rule_id: The rule.
    :param current: This invocation's pooled rate, or None when
        nothing exercised it.
    :param baseline: Prior invocations' pooled rates, oldest first.
    :param verdict: What the comparison supports.
    :param margin: How far this rate could sit from its own truth
        by chance, as `sampling_margin` computed it. Carried so the
        summary can show it: once the margin is taken over runs
        rather than replies it is regularly wider than the band,
        and a rate visibly outside the band being called unchanged
        reads as a broken tool unless the bar is on the page.
    :param margin_source: Where `margin` was estimated from. Shown
        wherever the margin is: the three sources disagree by two
        to three times on real data, so rendering a proxy as though
        it were the measured spread would hide exactly the gap that
        made this instrument misread its own numbers.
    :param by_script: This invocation's score per script, as
        `RuleRecord.by_script`. For reading only — the verdict is
        taken on the pooled rate, because judging every script
        separately doubles the checks on a gate that already fails
        by chance on rare rules.
    """

    rule_id: str
    current: float | None
    baseline: tuple[float, ...]
    verdict: Verdict
    margin: float = 0.0
    margin_source: MarginSource = MarginSource.REPLIES
    by_script: tuple[RunScore, ...] = ()

    @property
    def script_split(self) -> str:
        """Each script's rate, as ``cold 17%  warm 8%``.

        :return: The split, or an empty string when fewer than two
            scripts exercised the rule — then the pooled rate is
            already the one script's rate — or when nothing broke
            it, since ``cold 0%  warm 0%`` says nothing the pooled
            0% did not. On one agent nine of fifteen rules score
            zero, and printing their splits buried the six that
            carry one.
        """
        exercised = [s for s in self.by_script if s.rate is not None]
        if len(exercised) < 2 or not any(s.violations for s in exercised):
            return ""
        return "  ".join(f"{s.label} {s.rate:.0%}" for s in exercised)

    @property
    def band(self) -> tuple[float, float] | None:
        """Lowest and highest prior rate, or None without a band."""
        if len(self.baseline) < MIN_BASELINE_RUNS:
            return None
        return (min(self.baseline), max(self.baseline))

    @property
    def changed(self) -> bool:
        """Whether the rate left the band it had established."""
        return self.verdict in (
            Verdict.ABOVE_BASELINE,
            Verdict.BELOW_BASELINE,
        )


@dataclass(frozen=True)
class HistoryComparison:
    """One invocation read against everything before it.

    :param record: The invocation being compared.
    :param prior: The records it was compared against.
    :param rules: One `RuleComparison` per rule.
    :param other_models: Invocations of the same configuration left
        out because a different model answered them, or because
        they recorded none while this one did.
    """

    record: StyleRunRecord
    prior: tuple[StyleRunRecord, ...]
    rules: tuple[RuleComparison, ...]
    other_models: tuple[StyleRunRecord, ...] = ()

    @property
    def worse(self) -> tuple[RuleComparison, ...]:
        """Rules that rose above the band they had established."""
        return tuple(
            r for r in self.rules if r.verdict is Verdict.ABOVE_BASELINE
        )

    @property
    def better(self) -> tuple[RuleComparison, ...]:
        """Rules that fell below the band they had established."""
        return tuple(
            r for r in self.rules if r.verdict is Verdict.BELOW_BASELINE
        )

    @property
    def unbaselined(self) -> tuple[RuleComparison, ...]:
        """Rules with too little history to say anything."""
        return tuple(
            r for r in self.rules if r.verdict is Verdict.NO_BASELINE
        )

    def summary(self) -> str:
        """Readable comparison, changes first.

        :return: Multi-line string.
        """
        n = len(self.prior)
        # Name what the priors were matched on. Since v9.1.0 that is
        # `config_key` whenever the record carries one, and the
        # fingerprint then differs from the priors' as soon as a rule
        # is added — printing it would name a key none of them has.
        identity = (
            f"key {self.record.config_key}"
            if self.record.config_key
            else self.record.fingerprint
        )
        lines = [
            f"Against {n} prior invocation(s) of the same "
            f"configuration ({identity}):"
        ]
        if self.other_models:
            others = sorted({_answered_label(r) for r in self.other_models})
            lines.append(
                f"  Answered by {_answered_label(self.record)}. "
                f"{len(self.other_models)} earlier invocation(s) of "
                f"this configuration were answered by "
                f"{'; '.join(others)} and are left out: a different "
                f"model starts a fresh baseline."
            )
        if n < MIN_BASELINE_RUNS:
            lines.append(
                f"  No baseline yet — {MIN_BASELINE_RUNS} prior "
                f"invocations are needed before a difference can be "
                f"told from the variation between invocations. "
                f"Nothing below is a verdict."
            )
        width = max((len(r.rule_id) for r in self.rules), default=4)
        order = {
            Verdict.ABOVE_BASELINE: 0,
            Verdict.BELOW_BASELINE: 1,
            Verdict.WITHIN_BASELINE: 2,
            Verdict.RARE: 3,
            Verdict.NO_BASELINE: 4,
            Verdict.UNEXERCISED: 5,
        }
        marks = {
            Verdict.ABOVE_BASELINE: "WORSE ",
            Verdict.BELOW_BASELINE: "better",
            Verdict.WITHIN_BASELINE: "  ==  ",
            Verdict.RARE: " rare ",
            Verdict.NO_BASELINE: "  ??  ",
            Verdict.UNEXERCISED: "  --  ",
        }
        for comparison in sorted(
            self.rules, key=lambda r: (order[r.verdict], r.rule_id)
        ):
            now = (
                "  n/a"
                if comparison.current is None
                else f"{comparison.current:5.0%}"
            )
            band = comparison.band
            against = (
                "no band"
                if band is None
                else f"was {band[0]:.0%}-{band[1]:.0%}, not judged"
                if comparison.verdict is Verdict.RARE
                else (
                    f"was {band[0]:.0%}-{band[1]:.0%}, "
                    f"margin {comparison.margin:.0%}"
                    f"{_MARGIN_MARKS[comparison.margin_source]}"
                )
            )
            lines.append(
                f"  {marks[comparison.verdict]} "
                f"{comparison.rule_id:<{width}} {now}  ({against})"
            )
            if comparison.script_split:
                lines.append(f"{' ' * 9}{comparison.script_split}")
        if any(r.script_split for r in self.rules):
            lines.append(
                "  Indented lines split a rate by script. Only the "
                "combined rate is judged."
            )
        if any(r.verdict is Verdict.RARE for r in self.rules):
            lines.append(
                "  rare = earlier runs broke this rule less than once "
                "each on average. At that count bad luck alone "
                "leaves the band now and then, so it is shown and "
                "never counted as a change."
            )
        banded = [
            r
            for r in self.rules
            if r.band is not None and r.verdict is not Verdict.RARE
        ]
        if banded:
            lines.append(
                "  margin = how far this rate could sit from its "
                "own truth by chance. A rate outside its band by "
                "less than this is not a change."
            )
        sources = {r.margin_source for r in banded}
        for source in (MarginSource.RUNS, MarginSource.REPLIES):
            if source in sources:
                lines.append(_MARGIN_FOOTNOTES[source])
        return "\n".join(lines)


def compare_to_history(
    record: StyleRunRecord,
    history: Iterable[StyleRunRecord],
    fingerprint: str | None = None,
) -> HistoryComparison:
    """Read one invocation against earlier ones.

    Compares only against records sharing a fingerprint, so a band
    never describes a different agent than the one being judged.

    **A rate is called changed only when it falls outside the band
    its own history established**, and only once that history has
    at least `MIN_BASELINE_RUNS` invocations behind it. That is
    deliberately conservative: the whole reason this module exists
    is that two invocations of an unchanged config differed by 29
    percentage points, which any test comparing a single before to
    a single after would have reported as a regression.

    **Only invocations answered by the same model count**, when
    the record says which model answered (`answered_by`). The
    fingerprint deliberately leaves the model out, so a provider
    moving an alias keeps the key — and a band spanning two models
    passes nearly anything. On one agent it spanned 75% and 10% on
    one rule after DeepSeek's 10th September 2026 change. The left
    out invocations are returned as `other_models` and named in the
    summary, so the switch is reported rather than silently reset.
    An invocation that recorded no model while this one did is left
    out too: it cannot be shown to be the same model. A record that
    carries no model itself is compared as before v9.0.0.

    **The same holds for the guard's model** (`guard_answered_by`,
    since v9.3.0), checked on its own: a record whose guard reports
    a model only counts invocations whose guard was answered by the
    same one.

    **Configuration means `config_key`, and each rule is held to
    its own definition** (since v9.1.0). Two records that both carry
    a `config_key` are the same configuration when those match,
    whatever rules each was scored under; otherwise the fingerprints
    must match, as before. Inside that, a rule's band only takes a
    run whose `RuleRecord.definition` equals the current one. A
    definition that either side does not carry is not held against
    it, so a record written before v9.1.0 counts exactly as it did.

    :param record: The invocation to judge.
    :param history: Earlier records. The record itself is ignored
        if present, matched on its timestamp and fingerprint.
    :param fingerprint: Build the baseline from this configuration
        instead of the record's own — for deliberately comparing a
        changed config against the previous one's band.
    :return: The comparison.
    """
    same_config = tuple(
        r
        for r in history
        if (
            r.fingerprint == fingerprint
            if fingerprint is not None
            else _same_configuration(r, record)
        )
        and not (
            r.recorded_at == record.recorded_at
            and r.fingerprint == record.fingerprint
        )
    )
    prior = tuple(r for r in same_config if _same_models(r, record))
    other_models = tuple(
        r for r in same_config if not _same_models(r, record)
    )
    comparisons = []
    for rule in record.rules:
        baseline: list[float] = []
        prior_violations: list[int] = []
        for earlier in prior:
            earlier_rule = earlier.rule(rule.id)
            if earlier_rule is None or not _same_definition(
                earlier_rule, rule
            ):
                continue
            rate = earlier_rule.rate
            if rate is not None:
                baseline.append(rate)
                prior_violations.append(earlier_rule.violations)
        comparisons.append(_compare_rule(rule, baseline, prior_violations))
    return HistoryComparison(
        record=record,
        prior=prior,
        rules=tuple(comparisons),
        other_models=other_models,
    )


def _same_configuration(
    earlier: StyleRunRecord, record: StyleRunRecord
) -> bool:
    """Whether two records measured the same agent on the same scripts.

    On `config_key` when both carry one, so the rule set is not
    part of the question; on the fingerprint otherwise.
    """
    if earlier.config_key and record.config_key:
        return earlier.config_key == record.config_key
    return earlier.fingerprint == record.fingerprint


def _same_models(earlier: StyleRunRecord, record: StyleRunRecord) -> bool:
    """Whether *earlier* was answered by the models *record* was.

    The chatter's and the guard's models are checked separately,
    each only when *record* says which model answered it.
    """
    if record.answered_by and earlier.answered_by != record.answered_by:
        return False
    return not (
        record.guard_answered_by
        and earlier.guard_answered_by != record.guard_answered_by
    )


def _answered_label(record: StyleRunRecord) -> str:
    """Which models answered a record, for the summary."""
    label = ", ".join(record.answered_by) or "no recorded model"
    if record.guard_answered_by:
        label += f" (guard {', '.join(record.guard_answered_by)})"
    return label


def _same_definition(earlier: RuleRecord, current: RuleRecord) -> bool:
    """Whether two scores of one rule id came from the same check.

    Only a known difference separates them: a record written before
    definitions existed has none, and is not held against a rule
    for that.
    """
    if earlier.definition is None or current.definition is None:
        return True
    return earlier.definition == current.definition


def _cluster_error(runs: Sequence[RunScore]) -> float | None:
    """Standard error of the pooled rate, stratified by script.

    **The runs of one invocation are not draws from one
    distribution.** One agent plays a cold script and a warm one,
    and they have genuinely different rates — 82% against 42% on
    `one_message_when_dry`. Every invocation replays both, in the
    same proportion, so that gap is fixed structure rather than
    something that varies from one invocation to the next.
    Estimating the spread around the grand mean counts it as noise
    and reports a margin far wider than the instrument's real
    precision, which makes the gate blind to regressions it could
    otherwise see.

    So the spread is measured **within** each script and pooled
    across them: the standard cluster-robust variance of a ratio
    estimator, each script's runs taken around that script's own
    pooled rate. Weighting each run by how many replies the rule
    applied to, rather than treating the rates as equal, also
    drops the assumption that every run is the same length — two
    scripts of different shapes break it, and one agent's do.

    With a single script this is the same estimator the equal-rate
    version approximates, and identical to it when the runs are
    the same length.

    :param runs: One entry per play of one script.
    :return: The standard error, or None when no script has two
        runs that could disagree.
    """
    if not runs:
        return None
    total = sum(r.applicable for r in runs)
    if total <= 0:
        return None
    by_script: dict[str, list[RunScore]] = {}
    for run in runs:
        by_script.setdefault(run.script, []).append(run)
    variance = 0.0
    measured = False
    for group in by_script.values():
        count = len(group)
        applicable = sum(r.applicable for r in group)
        if count < 2 or applicable <= 0:
            continue
        measured = True
        rate = sum(r.violations for r in group) / applicable
        variance += (
            sum((r.violations - r.applicable * rate) ** 2 for r in group)
            * count
            / (count - 1)
        )
    if not measured:
        return None
    return (variance / total**2) ** 0.5


def _lower_gamma(s: float, x: float) -> float:
    """The regularised lower incomplete gamma function P(s, x).

    By its power series, which converges for every positive `x`.
    """
    if x <= 0.0:
        return 0.0
    term = total = 1.0 / s
    n = 1
    while term > 1e-15 * total and n < 10_000:
        term *= x / (s + n)
        total += term
        n += 1
    return total * exp(s * log(x) - x - lgamma(s))


@cache
def _chi2_quantile(p: float, df: int) -> float:
    """The `p` quantile of a chi-square distribution.

    Found by bisection on `_lower_gamma` rather than read off a
    table, so there is no recalled constant to be wrong: the closed
    forms at one and two degrees of freedom are asserted against it
    in the tests.

    :param p: The probability, strictly between 0 and 1.
    :param df: Degrees of freedom, at least 1.
    :return: The quantile.
    """
    low, high = 0.0, 10.0 * df + 100.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if _lower_gamma(df / 2.0, mid / 2.0) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _spread_upper_bound(rates: Sequence[float]) -> float:
    """The upper `SPREAD_CONFIDENCE` bound on the rates' spread.

    The standard deviation scaled by ``√((n-1) / χ²)``, with the
    chi-square quantile taken at ``1 - SPREAD_CONFIDENCE`` on
    ``n - 1`` degrees of freedom. See `SPREAD_CONFIDENCE` for why.

    :param rates: At least two rates.
    :return: The bound, as a rate.
    """
    df = len(rates) - 1
    quantile = _chi2_quantile(1.0 - SPREAD_CONFIDENCE, df)
    return stdev(rates) * (df / quantile) ** 0.5


def sampling_margin(
    violations: int,
    applicable: int,
    run_rates: Sequence[float] = (),
    runs: Sequence[RunScore] = (),
    prior_rates: Sequence[float] = (),
) -> float:
    """How far a rate could sit from its own truth by chance.

    Twice the standard error of the pooled rate, floored at half an
    event so a rate of zero does not claim perfect precision.

    **Why a band alone is not enough.** A band from two prior
    invocations that happened to agree is a band of almost no
    width, and then a single extra event reads as a regression.
    Measured on one agent: two invocations saw one stray greeting
    each, giving a band of 0.6%-1.0%; a third saw four, and being
    outside that band would have called four events out of 175 a
    regression. Requiring the gap to exceed this margin says
    instead that four is not distinguishable from one at this
    sample size, which is true.

    **Why the replies are not the sample.** A contract does not
    draw its replies independently — it plays a script, so every
    reply after the first is answering an agent that has already
    read its own previous replies. The unit that varies
    independently is the **run**, not the reply, and a standard
    error computed over the pooled replies understates the
    uncertainty by however much the turns inside one conversation
    agree with each other.

    Measured on one agent's committed transcripts, the two are
    nothing alike. `ends_on_a_question` scored 47%-64% across nine
    conversations, a spread narrower than binomial noise alone
    would predict — its turns really are close to independent.
    `one_message_when_dry` scored 42%, 82% and 81% across the
    three conversations long enough to mean anything, about three
    times the variance the pooled figure assumes: the agent settles
    into a one-message or a two-message rhythm early and keeps it,
    so its sixteen dry turns carry nowhere near sixteen turns'
    worth of information.

    That understatement is not academic. It is what let three
    invocations of one unchanged config — 23%, 70%, 41% on that
    same rule — be read as a difference no sampling could produce,
    and sent a session looking for a cause outside the config.

    So when the runs are available, the standard error is
    estimated from **how much the runs disagreed** rather than
    assumed from the pooled denominator. The binomial figure stays
    as a floor: a handful of runs that happen to agree must not buy
    more confidence than the sample size can support.

    **And runs of different scripts are kept apart.** A contract
    with two scripts plays both every time, so the difference
    between them is fixed structure and not something that varies
    between invocations. Pooling them made the margin far wider
    than the instrument's real precision — see `_cluster_error`,
    which is what `runs` buys over `run_rates`.

    **But the runs are still only a proxy, and not a close one.**
    What is being judged is whether the *invocation's* score moved,
    so the honest bar is how much that score has moved before.
    Measured over one agent's six gate invocations of one unchanged
    configuration, the figure taken from inside the invocation runs
    two to three times wider than the movement between them:
    `two_questions_in_a_row` printed 20% against a real 6%,
    `ends_on_a_question` 16% against 9%, `capitalises_after_stop`
    18% against 11%, `one_message_when_dry` 37% against 19%. Four
    rules held to a bar three times too wide is four rules where a
    real regression passes silently. And it is not wrong in one
    direction only — `opens_in_lower_case` printed 12% where the
    real movement is 20%, so that rule can flag on noise. A bar
    wrong in both directions cannot be corrected by moving a
    constant.

    So once `prior_rates` carries `MIN_SPREAD_PRIORS` earlier
    invocations of the same configuration, the margin is taken from
    their spread and the proxy is dropped outright — not kept as a
    floor. A `max()` of the two would leave every too-wide rule
    exactly as wide as it is now, which is the failure this exists
    to fix.

    **That spread is read at its upper `SPREAD_CONFIDENCE` bound**,
    not its point estimate: four priors carry three degrees of
    freedom, and four that happened to agree gave one agent's gate a
    1.4-point margin and a failure on an unchanged config.

    :param violations: Replies that broke the rule.
    :param applicable: Replies the rule applied to.
    :param run_rates: Each run's own rate, without saying which
        script it came from. The fallback, and all a record
        written before the labels were persisted can supply.
        Weighted equally, which assumes runs of roughly equal
        length, and taken around the grand mean, which counts the
        gap between two scripts as noise.
    :param runs: The same runs, labelled. Preferred over
        `run_rates`: it keeps the scripts apart and weights each
        run by its own length.
    :param prior_rates: This configuration's earlier invocations'
        rates for this rule. Preferred over both of the above once
        there are `MIN_SPREAD_PRIORS` of them.
    :return: The margin, as a rate. 0.0 when nothing applied.
    """
    return _margin(violations, applicable, run_rates, runs, prior_rates)[0]


def _margin(
    violations: int,
    applicable: int,
    run_rates: Sequence[float] = (),
    runs: Sequence[RunScore] = (),
    prior_rates: Sequence[float] = (),
) -> tuple[float, MarginSource]:
    """`sampling_margin`, plus where the figure came from.

    One function rather than two so the margin and the label on it
    cannot drift apart — a rendered figure whose source line is
    wrong is worse than one with no source line, since the reader
    then trusts a proxy as a measurement.

    :return: The margin as a rate, and what it was estimated from.
    """
    if applicable <= 0:
        return 0.0, MarginSource.REPLIES
    floor = 0.5 / applicable
    if len(prior_rates) >= MIN_SPREAD_PRIORS:
        spread = _spread_upper_bound(prior_rates)
        return 2.0 * max(spread, floor), MarginSource.PRIORS
    rate = violations / applicable
    error = (rate * (1.0 - rate) / applicable) ** 0.5
    spread = _cluster_error(runs)
    if spread is None and len(run_rates) >= 2:
        mean = sum(run_rates) / len(run_rates)
        variance = sum((r - mean) ** 2 for r in run_rates) / (
            len(run_rates) - 1
        )
        spread = (variance / len(run_rates)) ** 0.5
    if spread is None:
        return 2.0 * max(error, floor), MarginSource.REPLIES
    return 2.0 * max(error, spread, floor), MarginSource.RUNS


@cache
def _expected_max(count: int) -> float:
    """Where the highest of `count` normal draws sits, on average.

    In standard deviations above the mean. Computed by integrating
    ``x · n · φ(x) · Φ(x)^(n-1)`` rather than read off a table, so
    there is no recalled constant to be wrong: the two closed forms
    that exist — ``1/√π`` at two draws and ``3/(2√π)`` at three —
    are asserted against this in the tests.

    :param count: How many draws the maximum is taken over.
    :return: The expected maximum. 0.0 below two draws.
    """
    if count < 2:
        return 0.0
    normal = NormalDist()
    low, high, steps = -8.0, 8.0, 4000
    step = (high - low) / steps
    total = 0.0
    for index in range(steps + 1):
        x = low + index * step
        weight = 1 if index in (0, steps) else (4 if index % 2 else 2)
        total += (
            weight * x * count * normal.pdf(x) * normal.cdf(x) ** (count - 1)
        )
    return total * step / 3.0


@dataclass(frozen=True)
class ShiftEstimate:
    """A move worth catching, and the invocation size that catches it.

    The same three numbers whichever way round they were asked
    for: `detectable_shift` is given `runs` and answers `rate`,
    `runs_for_shift` is given `rate` and answers `runs`. Both fill
    all three, so a caller never has to know which produced the
    object to read it.

    `source` is not decoration. Measured on one agent, the weakest
    of the three sources disagreed with the strongest by between
    0.38 and 3.76 times on the same rules — optimistic on some,
    pessimistic on others — so a sizing answer whose provenance is
    unknown is not a sizing answer. Returning it alongside the
    figure is the same discipline `RuleComparison.margin_source`
    applies to a margin, for the same reason.

    :param rate: The move, as a rate. 0.0 means the spread came
        out at zero, which makes it unmeasured rather than absent.
    :param runs: Runs per invocation the figure goes with.
    :param source: Where the spread behind it was estimated from.
    """

    rate: float
    runs: int
    source: MarginSource


def _spread(
    planned: int,
    run_rates: Sequence[float],
    runs: Sequence[RunScore],
    prior_rates: Sequence[float],
    prior_runs_each: int | None,
) -> tuple[float, MarginSource]:
    """How far an invocation's score sits from its own truth.

    The standard deviation of one whole invocation's rate, at
    `planned` runs, and where that figure came from. The same three
    sources `_margin` uses, in the same order and for the same
    reasons — see `MarginSource` — differing only in that this one
    is asked for a hypothetical run count rather than the one that
    actually happened, so each estimate is rescaled to it.

    Rescaling is the square-root law: an invocation's score is the
    mean of its runs, so its spread falls with the square root of
    how many there are. That law is an assumption, and it is the
    one this module exists to doubt — but it is a measured one
    here. Comparing the same rules at 3 and 10 unguarded runs on
    one agent (30th August 2026) it roughly held, which is what
    makes a sizing answer worth printing at a count nobody has run.

    :param planned: Runs per invocation the answer is for.
    :param run_rates: Each run's rate, unlabelled.
    :param runs: The same runs, labelled — preferred, since it
        keeps two scripts' different true rates apart.
    :param prior_rates: Earlier invocations' rates for this rule.
    :param prior_runs_each: How many runs each of those had.
    :return: The spread as a rate, and its source.
    :raises AgentConfigurationError: If nothing supplies a spread,
        or priors are usable but their run count was not given.
    """
    if len(prior_rates) >= MIN_SPREAD_PRIORS:
        if prior_runs_each is None or prior_runs_each < 1:
            raise AgentConfigurationError(
                f"_spread: {len(prior_rates)} prior invocations were "
                f"given but not how many runs each of them had, so "
                f"their spread cannot be rescaled to {planned} runs. "
                f"Pass prior_runs_each."
            )
        scale = (prior_runs_each / planned) ** 0.5
        return stdev(prior_rates) * scale, MarginSource.PRIORS
    cluster = _cluster_error(runs)
    if cluster is not None:
        return cluster * (len(runs) / planned) ** 0.5, MarginSource.RUNS
    if len(run_rates) >= 2:
        mean = sum(run_rates) / len(run_rates)
        variance = sum((r - mean) ** 2 for r in run_rates) / (
            len(run_rates) - 1
        )
        return (variance / planned) ** 0.5, MarginSource.RUNS
    raise AgentConfigurationError(
        f"_spread: needs a spread to size against — at least 2 run "
        f"rates, two runs of one script, or {MIN_SPREAD_PRIORS} "
        f"prior invocations. Got {len(run_rates)} run rates, "
        f"{len(runs)} runs and {len(prior_rates)} priors."
    )


def detectable_shift(
    run_rates: Sequence[float] = (),
    planned_runs: int | None = None,
    *,
    prior_invocations: int = MIN_BASELINE_RUNS,
    runs: Sequence[RunScore] = (),
    prior_rates: Sequence[float] = (),
    prior_runs_each: int | None = None,
) -> ShiftEstimate:
    """The smallest real move this many runs could notice.

    `compare_to_history` calls a rate changed when it clears the
    highest prior rate *and* clears it by more than
    `sampling_margin`. So the bar sits at two things stacked: the
    top of the band, which is the maximum of `prior_invocations`
    draws and therefore already above the rule's true rate, plus
    twice the standard error. Both shrink with the square root of
    the number of runs, and neither shrinks with more replies per
    run — the run is the independent unit, so a longer script buys
    almost nothing and another conversation buys everything.

    That is the whole reason this exists: asked how to make a
    contract more sensitive, the intuitive answer is to lengthen
    the scripts, and it is the wrong one.

    **Read the result as a coin flip, not a guarantee.** A move
    exactly this size clears the bar about half the time, since the
    invocation measuring it is itself a draw. Catching it reliably
    takes roughly twice the runs.

    **More history makes the bar higher, not lower.** The maximum
    of five prior invocations sits further above the truth than the
    maximum of two, so a long baseline is harder to escape. That is
    a property of a min/max band rather than a defect, but it does
    mean `prior_invocations` belongs in the calculation. It is also
    the answer to a question this function invites: adding
    invocations does not buy sensitivity, so the only lever on the
    bar is the runs inside one.

    **Which spread it uses is the whole accuracy of the answer**,
    and it takes the same three sources as `sampling_margin`, in
    the same order — see `MarginSource`. Measured on one agent's
    five committed invocations, filed under `fb973621bddd` since
    v8.0.0 and under `f52c25ec29b8` when this was written, the
    within-invocation proxy this function used to be given no
    choice about was wrong by between 0.38 and 3.76 times against
    the movement the priors measure: it claimed a 2.9-point move
    was visible on `no_paragraph_lines` where the real bar is 7.6,
    and demanded 16.5 points on `capitalises_after_stop` where 4.4
    would do. Wrong in both directions, so no constant corrects it
    — which is why the source is returned rather than assumed.

    :param run_rates: Rates from runs of the same configuration,
        one per run. The weakest source: weighted equally, so it
        assumes runs of the same length, and taken around the grand
        mean, so it counts the gap between two scripts as noise.
    :param planned_runs: How many runs the planned invocation would
        have. Defaults to as many as were observed.
    :param prior_invocations: How many earlier invocations the band
        would be drawn from.
    :param runs: The same runs as `run_rates`, labelled. Preferred
        over it: keeps the scripts apart and weights each run by
        its own length.
    :param prior_rates: This configuration's earlier invocations'
        rates for this rule. Preferred over both once there are
        `MIN_SPREAD_PRIORS` of them — draws of exactly the quantity
        being judged, rather than a proxy for it.
    :param prior_runs_each: How many runs each of those had.
        Required with `prior_rates`, since their spread means
        nothing without the invocation size it was measured at.
    :return: The shift, the run count it assumes, and the source.
        A shift of 0.0 means the runs agreed exactly, which makes
        the spread unmeasured rather than absent — a contract has
        no rule whose true spread is zero.
    :raises AgentConfigurationError: If nothing supplies a spread,
        if `prior_runs_each` is missing, or if `planned_runs` is
        below one.
    """
    if planned_runs is not None and planned_runs < 1:
        raise AgentConfigurationError(
            f"detectable_shift: runs must be at least 1, got {planned_runs}."
        )
    observed = len(runs) or len(run_rates) or prior_runs_each or 0
    planned = observed if planned_runs is None else planned_runs
    # A planned count of zero means nothing said how big an
    # invocation is — no runs, no rates, no prior run count. That is
    # a missing spread rather than a bad size, and `_spread` names it
    # far better than a count check could, so it goes first.
    error, source = _spread(
        max(planned, 1), run_rates, runs, prior_rates, prior_runs_each
    )
    # The gate reads a prior spread at its upper bound (see
    # `SPREAD_CONFIDENCE`), so the margin half of the bar sized here
    # does too. The band's top is still the maximum of draws at the
    # spread itself.
    factor = 1.0
    if source is MarginSource.PRIORS:
        df = len(prior_rates) - 1
        factor = (df / _chi2_quantile(1.0 - SPREAD_CONFIDENCE, df)) ** 0.5
    rate = (_expected_max(prior_invocations) + 2.0 * factor) * error
    return ShiftEstimate(rate=rate, runs=planned, source=source)


def runs_for_shift(
    shift: float,
    run_rates: Sequence[float] = (),
    *,
    prior_invocations: int = MIN_BASELINE_RUNS,
    runs: Sequence[RunScore] = (),
    prior_rates: Sequence[float] = (),
    prior_runs_each: int | None = None,
) -> ShiftEstimate:
    """How many runs it takes to notice a move of a given size.

    The inverse of `detectable_shift`, and the one to reach for
    when setting a contract's `repeats`: measure the spread once,
    name the move worth catching, and read off the count instead
    of guessing it.

    **The answer is in runs because runs are the only lever.** Once
    `prior_rates` supplies the spread, the quantity setting the bar
    is how much a whole invocation's score has moved between
    invocations — so it is fair to ask whether the honest answer is
    a number of invocations instead. It is not: more invocations
    raise the band's top rather than lower it (see
    `detectable_shift`), and the replies inside a run are
    correlated, so a longer script barely helps. Runs per
    invocation is what remains.

    :param shift: The move worth catching, as a rate — 0.10 for
        ten percentage points.
    :param run_rates: Rates from runs of the same configuration.
    :param prior_invocations: How many earlier invocations the band
        would be drawn from.
    :param runs: The same runs, labelled. Preferred over
        `run_rates`.
    :param prior_rates: Earlier invocations' rates for this rule.
        Preferred over both once there are `MIN_SPREAD_PRIORS`.
    :param prior_runs_each: How many runs each of those had.
        Required with `prior_rates`.
    :return: The runs per invocation, the shift it was asked for,
        and the source of the spread behind it. At least one run.
    :raises AgentConfigurationError: If `shift` is not positive, if
        nothing supplies a spread, or if `prior_runs_each` is
        missing.
    """
    if shift <= 0.0:
        raise AgentConfigurationError(
            f"runs_for_shift: shift must be positive, got {shift}."
        )
    at_one = detectable_shift(
        run_rates,
        1,
        prior_invocations=prior_invocations,
        runs=runs,
        prior_rates=prior_rates,
        prior_runs_each=prior_runs_each,
    )
    needed = max(1, ceil((at_one.rate / shift) ** 2))
    return ShiftEstimate(rate=shift, runs=needed, source=at_one.source)


def _compare_rule(
    rule: RuleRecord,
    baseline: Sequence[float],
    prior_violations: Sequence[int] = (),
) -> RuleComparison:
    """Judge one rule's current rate against its prior rates.

    A change needs the rate to sit outside the band *and* to sit
    outside it by more than its own sampling margin, so a rule
    carrying a handful of events does not flag on one more.

    A rule its priors broke fewer than
    `RARE_VIOLATIONS_PER_INVOCATION` times each on average is
    `Verdict.RARE` instead: shown, never judged.

    The margin is taken from the prior rates themselves once there
    are `MIN_SPREAD_PRIORS` of them — they are draws of exactly the
    quantity being judged. Below that it falls back to the spread
    across this invocation's own runs, and below that again, for a
    record that kept no per-run rates, to the binomial figure.
    """
    current = rule.rate
    margin, source = _margin(
        rule.violations,
        rule.applicable,
        rule.run_rates,
        rule.runs,
        baseline,
    )
    if current is None:
        verdict = Verdict.UNEXERCISED
    elif len(baseline) < MIN_BASELINE_RUNS:
        verdict = Verdict.NO_BASELINE
    elif (
        sum(prior_violations) / len(prior_violations)
        < RARE_VIOLATIONS_PER_INVOCATION
    ):
        verdict = Verdict.RARE
    elif current - max(baseline) > margin:
        verdict = Verdict.ABOVE_BASELINE
    elif min(baseline) - current > margin:
        verdict = Verdict.BELOW_BASELINE
    else:
        verdict = Verdict.WITHIN_BASELINE
    return RuleComparison(
        rule_id=rule.id,
        current=current,
        baseline=tuple(baseline),
        verdict=verdict,
        margin=margin,
        margin_source=source,
        by_script=rule.by_script,
    )


def relabel(record: StyleRunRecord, subject: str) -> StyleRunRecord:
    """Copy a record under a different subject label.

    The label is free text for reading; the fingerprint is what
    comparison uses, and this does not touch it.

    :param record: The record to relabel.
    :param subject: The new label.
    :return: A copy.
    """
    return replace(record, subject=subject)


__all__ = [
    "MIN_BASELINE_RUNS",
    "RARE_VIOLATIONS_PER_INVOCATION",
    "HistoryComparison",
    "RuleComparison",
    "RuleRecord",
    "StyleRunRecord",
    "Verdict",
    "append_style_run",
    "compare_to_history",
    "configuration_key",
    "detectable_shift",
    "fingerprint_run",
    "load_style_history",
    "record_style_run",
    "relabel",
    "runs_for_shift",
    "sampling_margin",
]
