"""Style contracts for chat agents.

The label scoring in `agents7.eval.metrics` needs a right answer
and the trajectory scoring in `agents7.eval.trajectory` needs a
tool call. A conversational agent has neither: two good replies to
"how are you" share no wording, and nothing it does is a tool call.
So the only instrument a chat agent had was a person reading a
conversation, which is slow, subjective, and — the reason this
module exists — blind to the half of a persona that is a plain
string property of the output.

A **style contract** scores that half. Length, message count,
forbidden words, how a message opens, whether two replies in a row
end on a question: every one of those is checkable without a model
and without a person. What is *not* checkable stays where it
belongs, with the human reading the transcript.

Two things make this different from the other scorers:

- **The counterpart is scripted, deliberately.** Judging whether an
  agent is good company needs a live counterpart, because a reply
  written before the agent spoke proves nothing. Detecting that a
  rule stopped holding needs the opposite — the same stimulus every
  time, so the only thing that changed is the config.
- **A rule is scored as a rate, not a pass.** Chat agents run at
  temperature, so one run of one script is an anecdote. A rule
  declares the rate it tolerates and the contract is run enough
  times to measure one.

Rules come in three strengths, set by `budget`:

- ``0.0`` — a hard rule. Any violation fails the contract.
- ``0.5`` — a soft rule. Up to half the applicable turns may break
  it. Use for a tendency rather than a prohibition.
- ``None`` — measured but never failed. Use for a number worth
  watching that has no agreed threshold yet.

Usage::

    contract = load_style_contract("data/style_contract.yml")
    report = run_style_contract(lambda: build_my_agent(), contract)
    print(report.summary())
    assert report.passed
"""

import asyncio
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml

from agents7.agents.chatter import Chatter
from agents7.call_log import CallRecord
from agents7.exceptions import AgentConfigurationError, ProviderError
from agents7.llm.retry import connection_errors, timeout_errors
from agents7.logs import logger

# A check answers one of three things about a turn: it broke the
# rule, it kept it, or the rule had nothing to say about this turn.
# The third is what makes a rate meaningful — a length rule that
# only applies to a one-word prompt must not be diluted by the
# turns it never looked at.
VIOLATION = True
COMPLIANT = False
NOT_APPLICABLE = None


def _percent(value: float) -> str:
    """Render a budget without rounding it into a different one.

    Whole percents for the ones that are whole, and a decimal
    where the value needs it. One agent sets two rules at 0.5% —
    one stray event in an invocation — and at `.0%` those printed
    as `<=0%`, which is what a rule that tolerates *nothing*
    prints. Two thresholds that differ by a whole event's worth
    read as the same number, so a reader could not tell which of
    them a run had actually cleared.
    """
    percent = value * 100
    if abs(percent - round(percent)) < 1e-9:
        return f"{round(percent)}%"
    return f"{percent:g}%"


@dataclass(frozen=True)
class ReplyContext:
    """One of the agent's replies, with what it needs to be judged.

    :param index: Position in the conversation, counting the
        agent's replies from zero.
    :param prompt: What the counterpart wrote immediately before,
        or None when the agent opened the conversation.
    :param text: The reply exactly as the agent produced it.
    :param messages: The reply split into the separate chat
        messages a client would render — its non-empty lines. A
        persona that writes "two or three short lines" produces one
        reply and several messages, and rules about length almost
        always mean this rather than the character count.
    :param previous: The reply before this one, or None on the
        first. Rules about repetition need it.
    """

    index: int
    prompt: str | None
    text: str
    messages: tuple[str, ...]
    previous: "ReplyContext | None" = None

    @property
    def prompt_words(self) -> int:
        """How many words the counterpart wrote. 0 if it opened."""
        return len(self.prompt.split()) if self.prompt else 0


@dataclass(frozen=True)
class StyleRule:
    """One checkable property of the agent's replies.

    :param id: Short stable identifier, used in the report and in
        whatever tracks the rule. Name the property, not the fix —
        ``one_message_when_dry`` rather than ``dont_ramble``.
    :param description: What the rule requires, in the words the
        persona itself uses. This is what a reader sees when the
        rule fails, so it earns its length.
    :param check: Callable taking a `ReplyContext` and returning
        True for a violation, False for a compliant turn, and None
        when the rule does not apply to that turn.
    :param budget: The violation rate this rule tolerates, between
        0 and 1. ``0.0`` fails on any violation; ``None`` measures
        without ever failing.
    :param definition: Hash of what the check measures — its name
        and arguments — set by `build_rule`. A rule's history is
        only comparable with runs scored under the same definition,
        so changing a check's arguments under an unchanged id starts
        that rule's band afresh instead of judging new numbers
        against old ones. ``None`` for a rule built in code around a
        bare callable, whose definition nothing can read.
    """

    id: str
    description: str
    check: Callable[[ReplyContext], bool | None]
    budget: float | None = 0.0
    definition: str | None = None

    def __post_init__(self) -> None:
        if self.budget is not None and not 0.0 <= self.budget <= 1.0:
            raise AgentConfigurationError(
                f"StyleRule `{self.id}`: budget must be between "
                f"0.0 and 1.0, or None to measure only. Got "
                f"{self.budget!r}."
            )


@dataclass(frozen=True)
class ConversationScript:
    """A fixed counterpart, replayed identically on every run.

    :param id: Short name, used in the report.
    :param lines: What the counterpart says, in order. Each line
        is one turn.
    :param opens: True when the agent speaks first, before the
        counterpart's first line. Needs `can_open` on the agent.
    :param description: What this script is for — what it puts the
        agent under, and which rules it exists to exercise.
    :param context: Values for the agent's declared
        `context_params`, handed to `set_context` before the first
        turn. Part of the stimulus, so it is part of the run's
        fingerprint. Empty leaves the agent as the factory built
        it.
    """

    id: str
    lines: tuple[str, ...]
    opens: bool = False
    description: str = ""
    context: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StyleContract:
    """The rules to check and the scripts to check them against.

    :param rules: The properties every reply is scored on.
    :param scripts: The conversations to run.
    :param repeats: How many times to run each script. Chat agents
        run at temperature, so one pass measures nothing; the
        default of 3 is a floor rather than a recommendation.
    """

    rules: tuple[StyleRule, ...]
    scripts: tuple[ConversationScript, ...]
    repeats: int = 3

    def __post_init__(self) -> None:
        if self.repeats < 1:
            raise AgentConfigurationError(
                f"StyleContract: repeats must be at least 1, got "
                f"{self.repeats}."
            )
        ids = [r.id for r in self.rules]
        if len(ids) != len(set(ids)):
            duplicated = sorted({i for i in ids if ids.count(i) > 1})
            raise AgentConfigurationError(
                f"StyleContract: rule ids must be unique. "
                f"Repeated: {', '.join(duplicated)}."
            )


@dataclass(frozen=True)
class RunScore:
    """How one rule scored in a single play of a single script.

    :param label: Which run this was — the script's id, and which
        repeat of it.
    :param violations: Replies in that run that broke the rule.
    :param applicable: Replies in that run the rule applied to.
    """

    label: str
    violations: int
    applicable: int

    @property
    def rate(self) -> float | None:
        """Violations over applicable, or None if unexercised."""
        if self.applicable == 0:
            return None
        return self.violations / self.applicable

    @property
    def script(self) -> str:
        """Which script this run played, read off the label.

        A label is ``<script id>#<repeat>``, so the script is what
        precedes the last ``#``. Derived rather than stored as a
        second field: the label is what the report prints, and two
        fields that have to agree eventually will not.

        Two scripts are two different stimuli with two different
        true rates, so anything estimating how much a rule varies
        has to keep them apart — see
        `agents7.eval.style_history.sampling_margin`.
        """
        head, separator, _ = self.label.rpartition("#")
        return head if separator else self.label


@dataclass(frozen=True)
class LostRun:
    """A conversation that raised before it finished.

    **Its replies are not scored, not even the ones it produced.**
    A five-turn prefix of a sixteen-turn script is a different
    stimulus, and folding it into a band built from whole
    conversations would compare two different things. It is
    reported instead, so a partial invocation says it is one.

    :param label: The run's label, as `RunScore.label` uses it.
    :param error: The exception, as text.
    :param replies: How many replies it had produced before it
        stopped. Reported rather than scored.
    """

    label: str
    error: str
    replies: int


@dataclass(frozen=True)
class RuleResult:
    """How one rule scored over every reply the run produced.

    :param rule: The rule this scores.
    :param violations: Replies that broke it, across every run.
    :param applicable: Replies the rule had something to say about.
        A rule with no applicable replies is reported rather than
        silently passing — a script that never triggers a rule is a
        gap in the script, not evidence about the agent.
    :param examples: Up to a few offending replies, so the report
        shows what broke rather than only how often.
    :param per_run: The same score, one entry per play of one
        script. **A pooled rate alone is not a measurement.** On
        one agent, one repeat scored a rule at 81% and three
        repeats scored it at 23%; pooling cannot say whether one
        run was extreme or every run was middling, and those two
        call for opposite responses.
    """

    rule: StyleRule
    violations: int
    applicable: int
    examples: tuple[str, ...] = ()
    per_run: tuple[RunScore, ...] = ()

    @property
    def rate(self) -> float | None:
        """Violations over applicable replies, or None if unused."""
        if self.applicable == 0:
            return None
        return self.violations / self.applicable

    @property
    def run_rates(self) -> tuple[float, ...]:
        """Each run's rate, skipping runs that never triggered it."""
        return tuple(run.rate for run in self.per_run if run.rate is not None)

    @property
    def spread(self) -> tuple[float, float] | None:
        """Lowest and highest run rate, or None below two runs.

        None when fewer than two runs exercised the rule, because
        a single run has no spread to report and printing one
        would imply a stability that was never measured.
        """
        rates = self.run_rates
        if len(rates) < 2:
            return None
        return (min(rates), max(rates))

    @property
    def volatile(self) -> bool:
        """True when the run rates disagree by more than half.

        A wide spread means the pooled rate is an average over runs
        that behaved differently, so it predicts nothing about the
        next one. Judged on the gap rather than a variance, because
        the denominators here are small enough that a variance
        would be its own kind of false precision.
        """
        spread = self.spread
        return spread is not None and spread[1] - spread[0] > 0.5

    @property
    def passed(self) -> bool:
        """Whether this rule is within its budget.

        Checked on the pooled rate, which is the expected rate
        over runs. A rule whose runs disagree wildly can pass here
        while a single run was far worse — `volatile` is what says
        so, and it is surfaced in `summary()`.

        A measure-only rule always passes. So does a rule nothing
        exercised — `unexercised` is what flags that, because a
        script gap is a different problem from a broken rule and
        must not be reported as a failure of the agent.
        """
        if self.rule.budget is None or self.rate is None:
            return True
        return self.rate <= self.rule.budget

    @property
    def unexercised(self) -> bool:
        """True when no reply the run produced triggered the rule."""
        return self.applicable == 0


@dataclass(frozen=True)
class StyleReport:
    """What a contract run found.

    :param results: One `RuleResult` per rule, in declaration
        order.
    :param replies: Every reply that was scored, across scripts and
        repeats.
    :param calls: Every LLM call the run made, guard checks and
        regenerations included. Present so a run can say what it
        cost — a contract that has to be run repeatedly to mean
        anything is worth being able to price.
    :param lost: Conversations that raised before finishing, and
        were therefore not scored. Empty on a clean run. **A run
        that dies takes only itself down**: every conversation that
        finished is still scored, so a dropped connection costs one
        conversation rather than the whole invocation. This is what
        says the invocation was a partial one.
    :param stopped_early: True when the run stopped because the
        provider stopped answering, so some of `lost` were never
        played at all rather than failing on their own.
    :param variants: The text each scored run's agent drew for each
        of its `prompt_variants`, keyed by the run's label. Since
        v9.12.0 two runs of one config can be shown different
        examples, which moves a rate as readily as a config edit
        does, so what a run was shown is kept beside its score.
        Held once per run rather than on every `RunScore`, since
        it is the same for every rule. Empty for an agent with no
        variants.
    """

    results: tuple[RuleResult, ...]
    replies: tuple[ReplyContext, ...]
    calls: tuple[CallRecord, ...] = ()
    lost: tuple[LostRun, ...] = ()
    stopped_early: bool = False
    variants: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether every rule stayed within its budget.

        Says nothing about whether every conversation finished —
        `complete` is what answers that, and a partial invocation
        can pass every budget on the runs it did score.
        """
        return all(r.passed for r in self.results)

    @property
    def complete(self) -> bool:
        """Whether every planned conversation finished."""
        return not self.lost

    @property
    def failures(self) -> tuple[RuleResult, ...]:
        """The rules that broke their budget."""
        return tuple(r for r in self.results if not r.passed)

    @property
    def unexercised(self) -> tuple[RuleResult, ...]:
        """The rules no reply in this run triggered."""
        return tuple(r for r in self.results if r.unexercised)

    @property
    def cost_cents(self) -> float | None:
        """What the run cost, or None if nothing was priced.

        None rather than 0.0 when no call carried a price, so an
        unpriced engine is never reported as a free one.
        """
        priced = [
            c.cost_cents for c in self.calls if c.cost_cents is not None
        ]
        return sum(priced) if priced else None

    @property
    def duration(self) -> float:
        """Seconds of LLM time the run spent."""
        return sum(c.total_duration for c in self.calls)

    def rate(self, rule_id: str) -> float | None:
        """Look up one rule's violation rate by id.

        :param rule_id: The rule's `id`.
        :return: Its rate, or None when nothing exercised it.
        :raises KeyError: If no rule has that id.
        """
        for result in self.results:
            if result.rule.id == rule_id:
                return result.rate
        raise KeyError(rule_id)

    @property
    def volatile(self) -> tuple[RuleResult, ...]:
        """Rules whose runs disagreed by more than half.

        Their pooled rate is an average over runs that behaved
        differently, so it predicts nothing about the next run and
        is not a baseline.
        """
        return tuple(r for r in self.results if r.volatile)

    def summary(self) -> str:
        """Readable per-rule table, worst first.

        Each row carries the pooled rate *and* the range across
        runs, because a pooled rate on its own cannot distinguish
        one bad run from several middling ones.

        :return: Multi-line string.
        """
        runs = max((len(r.per_run) for r in self.results), default=0)
        lines = [
            f"Style contract: {len(self.replies)} replies over "
            f"{runs} runs, {len(self.calls)} LLM calls"
        ]
        if self.cost_cents is not None:
            lines[0] += f", {self.cost_cents:.4f}c"
        if self.stopped_early:
            lines.append(
                "  !! stopped early: the provider stopped answering, so "
                "the conversations after that point were not played"
            )
        for run in self.lost:
            lines.append(
                f"  !! {run.label} stopped after {run.replies} "
                f"replies and is not scored: {run.error}"
            )
        width = max((len(r.rule.id) for r in self.results), default=4)
        ordered = sorted(
            self.results,
            key=lambda r: (r.passed, not r.volatile, -(r.rate or 0.0)),
        )
        for result in ordered:
            if result.unexercised:
                mark, rate = "  ??  ", "  n/a"
            else:
                mark = "  ok  " if result.passed else "BROKEN"
                rate = f"{result.rate:5.0%}"
            budget = (
                "measure"
                if result.rule.budget is None
                else f"<={_percent(result.rule.budget)}"
            )
            spread = result.spread
            if spread is None:
                across = ""
            else:
                across = f"  runs {spread[0]:.0%}-{spread[1]:.0%}"
                if result.volatile:
                    across += " !"
            lines.append(
                f"  {mark} {result.rule.id:<{width}} {rate} "
                f"({result.violations}/{result.applicable}, "
                f"{budget}){across}"
            )
        if self.unexercised:
            lines.append(
                "  ?? = no reply triggered this rule; the script "
                "does not reach it."
            )
        if self.volatile:
            lines.append(
                "  !  = runs disagreed by more than half. The "
                "pooled rate is not a baseline."
            )
        return "\n".join(lines)


# --- Built-in checks ---
#
# The shapes a persona's style rules actually take. Each returns a
# `check` callable for `StyleRule`. Declaring these by name is what
# lets a contract live in YAML beside the agent rather than in
# code, so a config-driven agent's rules and its contract can be
# read side by side.


def forbidden_text(values: Iterable[str]) -> Callable:
    """Reply must contain none of *values* (case-insensitive)."""
    needles = tuple(v.lower() for v in values)

    def check(ctx: ReplyContext) -> bool:
        haystack = ctx.text.lower()
        return any(n in haystack for n in needles)

    return check


def forbidden_pattern(
    pattern: str,
    when_prompt_matches: str | None = None,
) -> Callable:
    """Reply must not match *pattern* anywhere.

    :param pattern: Regular expression the reply must not match.
    :param when_prompt_matches: Only apply when what the
        counterpart wrote matches this regular expression. For a
        rule about how the agent answers one kind of question,
        which every other turn would otherwise dilute.
    """
    compiled = re.compile(pattern)
    prompt_pattern = (
        None
        if when_prompt_matches is None
        else re.compile(when_prompt_matches)
    )

    def check(ctx: ReplyContext) -> bool | None:
        if prompt_pattern is not None:
            if ctx.prompt is None or not prompt_pattern.search(ctx.prompt):
                return NOT_APPLICABLE
        return bool(compiled.search(ctx.text))

    return check


def max_messages(
    limit: int,
    when_prompt_words_at_most: int | None = None,
) -> Callable:
    """Reply must be at most *limit* separate chat messages.

    :param limit: Most messages allowed.
    :param when_prompt_words_at_most: Only apply when the
        counterpart wrote this many words or fewer. This is how a
        persona's "when she is dry, one line not two" is expressed:
        without it the rule would be diluted by every turn it was
        never about.
    """

    def check(ctx: ReplyContext) -> bool | None:
        if when_prompt_words_at_most is not None:
            if ctx.prompt is None:
                return NOT_APPLICABLE
            if ctx.prompt_words > when_prompt_words_at_most:
                return NOT_APPLICABLE
        return len(ctx.messages) > limit

    return check


def max_line_chars(limit: int) -> Callable:
    """No single message may run longer than *limit* characters."""

    def check(ctx: ReplyContext) -> bool:
        return any(len(m) > limit for m in ctx.messages)

    return check


def ends_with_question() -> Callable:
    """Reply's last message ends on a question mark.

    Usually declared with a budget rather than as a prohibition —
    asking is a move, and the question is how often.
    """

    def check(ctx: ReplyContext) -> bool:
        return bool(ctx.messages) and ctx.messages[-1].rstrip().endswith("?")

    return check


def repeats_question_ending() -> Callable:
    """This reply and the one before it both end on a question.

    Not applicable to the first reply, which has nothing before it.
    """

    def check(ctx: ReplyContext) -> bool | None:
        if ctx.previous is None:
            return NOT_APPLICABLE
        ends = ends_with_question()
        return bool(ends(ctx)) and bool(ends(ctx.previous))

    return check


def first_char_lowercase(exempt_prefixes: Iterable[str] = ()) -> Callable:
    """Reply's first message must start in lower case.

    :param exempt_prefixes: Openings the persona allows in upper
        case anyway — a standalone greeting, a laugh. Matched
        case-insensitively against the start of the message.

    Leading punctuation is skipped before the test, so a Spanish
    ``¿`` or ``¡`` does not make every reply pass by accident.
    """
    exempt = tuple(p.lower() for p in exempt_prefixes)

    def check(ctx: ReplyContext) -> bool | None:
        if not ctx.messages:
            return NOT_APPLICABLE
        message = ctx.messages[0]
        if any(message.lower().startswith(p) for p in exempt):
            return COMPLIANT
        stripped = message.lstrip("¿¡\"'()[[]-–—… ")
        first = stripped[:1]
        if not first.isalpha():
            return NOT_APPLICABLE
        return first.isupper()

    return check


def capitalises_after_stop() -> Callable:
    """A new sentence inside a message starts in upper case.

    The companion to `first_char_lowercase`, and the opposite kind
    of rule. Opening a chat message in lower case is a register
    choice — a person may or may not. Writing ``vale. ni idea``
    once a sentence has ended is simply wrong, and a persona that
    asks for chat casing usually means the first letter only.

    Separating the two matters because conflating them is a real
    failure: one agent's persona says the message opens in lower
    case *and* that sentences inside it are capitalised as normal,
    and calls only the second non-optional. A contract that read
    the insistence as attaching to the first enforced a preference
    and left the actual rule unchecked.

    An ellipsis is deliberately not a sentence end — ``no sé...
    igual mañana`` trails off into the same thought, and Spanish
    keeps it lower case. Spanish opening marks are skipped, so
    ``. ¿abro yo?`` is judged on the ``a``.

    :return: The check.
    """
    # `(?<!\.)` drops the last dot of an ellipsis; the earlier dots
    # cannot match anyway, having no space after them.
    pattern = re.compile(r"(?<!\.)[.!?]\s+([¿¡\"'(\[]*)([^\W\d_])")

    def check(ctx: ReplyContext) -> bool | None:
        applicable = False
        for message in ctx.messages:
            for match in pattern.finditer(message):
                applicable = True
                if match.group(2).islower():
                    return VIOLATION
        return COMPLIANT if applicable else NOT_APPLICABLE

    return check


def max_emoji(limit: int, emoji: Iterable[str]) -> Callable:
    """No message may carry more than *limit* of *emoji*."""
    marks = tuple(emoji)

    def check(ctx: ReplyContext) -> bool:
        return any(
            sum(m.count(e) for e in marks) > limit for m in ctx.messages
        )

    return check


def greets_after_opening_with(openings: Iterable[str]) -> Callable:
    """Reply opens with a greeting after the conversation started.

    :param openings: The persona's own opening lines, or their
        distinctive first words. Matched case-insensitively against
        the start of the reply's first message.
    """
    needles = tuple(o.lower() for o in openings)

    def check(ctx: ReplyContext) -> bool | None:
        if ctx.index == 0:
            return NOT_APPLICABLE
        if not ctx.messages:
            return NOT_APPLICABLE
        opener = ctx.messages[0].lower()
        return any(opener.startswith(n) for n in needles)

    return check


def repeats_opener() -> Callable:
    """A conversation opens with the same line as another one.

    The one built-in check that cannot judge a reply on its own. A
    persona told to vary its opening can open every conversation
    identically and break no per-reply rule: one agent opened all
    three conversations of its 6th September 2026 read with
    "¡Buenas! ¿Qué tal?", and all five of the 13th September read
    with "Toc toc.", while its instructions said to vary it.

    So the runner hands the check every conversation in the
    invocation before scoring (`bind_runs`), and the check reads
    each conversation's first message. A first message is a
    violation when at least one other conversation opened with the
    same line, compared on collapsed whitespace and casing with
    punctuation stripped from the ends. Only the first reply of a
    conversation is applicable, so the rate is the share of
    conversations whose opening someone else also used.

    **The rate depends on how many conversations there are.** Six
    conversations with a few openers collide more often than two, so
    compare it only against invocations of the same size — which a
    history kept per repeat count already does.

    :return: The check. Calling it directly on one reply raises,
        because one reply has nothing to be compared with.
    """

    def normalise(text: str) -> str:
        return " ".join(text.lower().split()).strip(" .,;:!¡?¿")

    def opening(ctx: ReplyContext) -> str | None:
        if ctx.index != 0 or not ctx.messages:
            return None
        return normalise(ctx.messages[0]) or None

    def unbound(ctx: ReplyContext) -> bool | None:
        raise AgentConfigurationError(
            "repeats_opener compares the conversations of a whole "
            "invocation, so it is scored by `run_style_contract` or "
            "`score_transcripts`, not called on a single reply."
        )

    def bind_runs(
        runs: Sequence[tuple[str, Sequence[ReplyContext]]],
    ) -> Callable[[ReplyContext], bool | None]:
        seen = Counter(
            line
            for _, replies in runs
            if replies and (line := opening(replies[0])) is not None
        )

        def check(ctx: ReplyContext) -> bool | None:
            line = opening(ctx)
            if line is None:
                return NOT_APPLICABLE
            return seen[line] > 1

        return check

    unbound.bind_runs = bind_runs  # type: ignore[attr-defined]
    return unbound


def reproduces_verbatim(lines: Iterable[str]) -> Callable:
    """Reply reproduces one of *lines* word for word.

    For catching a persona's own written examples coming back as a
    script. Compares on collapsed whitespace and casing, and
    ignores trailing punctuation, so a line copied with a full stop
    dropped still counts.
    """

    def normalise(text: str) -> str:
        return " ".join(text.lower().split()).strip(" .!¡")

    needles = tuple(normalise(line) for line in lines if line.strip())

    def check(ctx: ReplyContext) -> bool:
        return any(normalise(m) in needles for m in ctx.messages)

    return check


def copies_example(
    lines: Iterable[str],
    min_similarity: float = 0.7,
    min_chars: int = 16,
) -> Callable:
    """A message is one of *lines*, or close enough to be a copy.

    `reproduces_verbatim` only fires on a whole message equal to a
    line, so a copy with a word swapped or a sentence appended is
    invisible to it. Measured on one agent's 13th September 2026
    read, it saw fewer than half the copies a reader could point
    at. This counts a message as a copy when either:

    - it contains a whole line, or
    - its similarity to a line (`difflib.SequenceMatcher.ratio`)
      reaches *min_similarity*.

    The default of 0.7 is where that read split: every copy scored
    0.71 or above or contained its line, and the closest message
    the agent wrote in its own words scored 0.58.

    Lines and messages shorter than *min_chars* are ignored. A
    short stock reply — "recibido", "me has pillado" — is how a
    person answers every time, not a copied example. A reply made
    only of such messages is not applicable, so it does not dilute
    the rate.

    :param lines: The example lines the agent is shown.
    :param min_similarity: Similarity at which a message counts as
        a copy, above 0 and at most 1.
    :param min_chars: Shortest normalised line or message judged.
    :raises AgentConfigurationError: If a threshold is out of range.
    """
    if not 0.0 < min_similarity <= 1.0:
        raise AgentConfigurationError(
            f"copies_example: min_similarity must be above 0 and at "
            f"most 1. Got {min_similarity!r}."
        )
    if min_chars < 1:
        raise AgentConfigurationError(
            f"copies_example: min_chars must be at least 1. "
            f"Got {min_chars!r}."
        )

    def normalise(text: str) -> str:
        return " ".join(text.lower().split()).strip(" .,;:!¡?¿")

    needles = tuple(
        sorted(
            {
                n
                for n in (normalise(line) for line in lines)
                if len(n) >= min_chars
            }
        )
    )

    def check(ctx: ReplyContext) -> bool | None:
        judged = [
            said
            for said in (normalise(m) for m in ctx.messages)
            if len(said) >= min_chars
        ]
        if not judged:
            return NOT_APPLICABLE
        return any(
            needle in said
            or SequenceMatcher(None, said, needle).ratio() >= min_similarity
            for said in judged
            for needle in needles
        )

    return check


# Name → factory, for contracts declared in YAML. A check the
# registry does not know is an error rather than a skip: a contract
# that silently drops a rule reports a clean run it never made.
CHECK_FACTORIES: dict[str, Callable[..., Callable]] = {
    "forbidden_text": forbidden_text,
    "forbidden_pattern": forbidden_pattern,
    "max_messages": max_messages,
    "max_line_chars": max_line_chars,
    "ends_with_question": ends_with_question,
    "repeats_question_ending": repeats_question_ending,
    "capitalises_after_stop": capitalises_after_stop,
    "first_char_lowercase": first_char_lowercase,
    "max_emoji": max_emoji,
    "greets_after_opening": greets_after_opening_with,
    "repeats_opener": repeats_opener,
    "reproduces_verbatim": reproduces_verbatim,
    "copies_example": copies_example,
}


def _definition(check: str, args: dict[str, Any]) -> str:
    """Hash a declared check and its arguments.

    Key order does not matter, so reformatting a contract's YAML
    leaves every definition alone.

    :param check: The name from `CHECK_FACTORIES`.
    :param args: Its arguments, as declared.
    :return: Twelve hex characters.
    """
    canonical = json.dumps(
        {"check": check, "args": args},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def build_rule(spec: dict[str, Any]) -> StyleRule:
    """Build one `StyleRule` from its declared form.

    :param spec: Mapping with ``id``, ``description``, ``check``
        (a name from `CHECK_FACTORIES`), optional ``args`` for that
        factory, and optional ``budget``.
    :return: The rule.
    :raises AgentConfigurationError: If the check is unknown or the
        arguments do not fit it.
    """
    for required in ("id", "description", "check"):
        if required not in spec:
            raise AgentConfigurationError(
                f"Style rule is missing `{required}`: {spec!r}"
            )
    name = spec["check"]
    if name not in CHECK_FACTORIES:
        known = ", ".join(sorted(CHECK_FACTORIES))
        raise AgentConfigurationError(
            f"Style rule `{spec['id']}`: unknown check `{name}`. "
            f"Available: {known}."
        )
    args = spec.get("args") or {}
    if not isinstance(args, dict):
        raise AgentConfigurationError(
            f"Style rule `{spec['id']}`: `args` must be a mapping, "
            f"got {type(args).__name__}."
        )
    try:
        check = CHECK_FACTORIES[name](**args)
    except TypeError as exc:
        raise AgentConfigurationError(
            f"Style rule `{spec['id']}`: check `{name}` rejected "
            f"its args ({exc})."
        ) from exc
    budget = spec.get("budget", 0.0)
    return StyleRule(
        id=str(spec["id"]),
        description=str(spec["description"]),
        check=check,
        budget=None if budget is None else float(budget),
        definition=_definition(name, args),
    )


def load_style_contract(source: str | Path | dict) -> StyleContract:
    """Load a contract from a YAML file or an already-parsed dict.

    Schema::

        repeats: 3
        rules:
            - id: one_message_when_dry
              description: <what the persona requires>
              check: max_messages
              args: {limit: 1, when_prompt_words_at_most: 3}
              budget: 0.0
        scripts:
            - id: cold
              description: <what this script puts the agent under>
              opens: true
              lines: ["bien", "normal", "ya"]
              context: {nick: Marta_87}   # optional

    :param source: Path to the YAML file, or the parsed mapping.
    :return: The contract.
    :raises AgentConfigurationError: If the file declares no rules
        or no scripts.
    """
    if isinstance(source, dict):
        raw = source
    else:
        raw = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    if not raw or not raw.get("rules"):
        raise AgentConfigurationError(
            f"Style contract declares no rules: {source!r}"
        )
    if not raw.get("scripts"):
        raise AgentConfigurationError(
            f"Style contract declares no scripts: {source!r}. "
            f"Rules with nothing to run against score nothing."
        )
    scripts = tuple(
        ConversationScript(
            id=str(s["id"]),
            lines=tuple(str(line) for line in s.get("lines", ())),
            opens=bool(s.get("opens", False)),
            description=str(s.get("description", "")),
            context={
                str(name): str(value)
                for name, value in (s.get("context") or {}).items()
            },
        )
        for s in raw["scripts"]
    )
    return StyleContract(
        rules=tuple(build_rule(r) for r in raw["rules"]),
        scripts=scripts,
        repeats=int(raw.get("repeats", 3)),
    )


# --- Running ---


def _split_messages(text: str) -> tuple[str, ...]:
    """Split a reply into the messages a chat client would render."""
    return tuple(line.strip() for line in text.splitlines() if line.strip())


_NOT_PLAYED = "not played: the provider stopped answering"
_STOPPED = "stopped before its next turn: the provider stopped answering"


class _RunStopped(Exception):
    """Raised between turns once the run has decided to stop."""


def _provider_unavailable(error: BaseException) -> bool:
    """Whether *error* means the provider did not answer at all.

    Once a timeout or a dropped connection has used every retry, the
    engine raises a `ProviderError` whose cause is the SDK's own
    error; without retries the SDK's error arrives as it is. Either
    way that error is what separates "nobody answered" from an
    ordinary failure — a rejected request, an unparseable reply —
    which says nothing about the conversations still to come.
    """
    import ollama

    cause = error.__cause__ if isinstance(error, ProviderError) else error
    return isinstance(
        cause, (*timeout_errors(), *connection_errors(), ollama.RequestError)
    )


class _StopTracker:
    """Decides when a run stops starting conversations.

    Counts conversations lost in a row to a provider that did not
    answer. A finished conversation, or one lost to anything else,
    resets the count; reaching *limit* stops the run for good.
    """

    def __init__(self, limit: int | None) -> None:
        if limit is not None and limit < 1:
            raise AgentConfigurationError(
                f"stop_after_unavailable must be at least 1 or None, "
                f"got {limit}."
            )
        self._limit = limit
        self._in_a_row = 0
        self.stopped = False

    def lost(self, error: BaseException) -> None:
        if isinstance(error, _RunStopped):
            return
        if not _provider_unavailable(error):
            self._in_a_row = 0
            return
        self._in_a_row += 1
        if self._limit is not None and self._in_a_row >= self._limit:
            if not self.stopped:
                logger.warning(
                    f"Style contract: {self._in_a_row} conversations in "
                    f"a row got no answer from the provider; playing "
                    f"no more"
                )
            self.stopped = True

    def finished(self) -> None:
        self._in_a_row = 0


def _play(
    agent: Chatter,
    script: ConversationScript,
    replies: list[ReplyContext] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[ReplyContext]:
    """Run one script once and collect the agent's replies.

    :param agent: A fresh agent, conversation empty.
    :param script: The script to play.
    :param replies: Filled turn by turn, so a caller that passes
        its own list still holds the partial conversation when a
        turn raises. A fresh list when None.
    :param should_stop: Asked before each turn. When it says yes
        the conversation ends there, and a turn already sent is
        never cut off, so its call record stays complete.
    :return: The replies, which is *replies* when one was given.
    :raises _RunStopped: When *should_stop* ended it.
    """
    replies = [] if replies is None else replies
    previous: ReplyContext | None = replies[-1] if replies else None

    def record(prompt: str | None, text: str) -> None:
        nonlocal previous
        ctx = ReplyContext(
            index=len(replies),
            prompt=prompt,
            text=text,
            messages=_split_messages(text),
            previous=previous,
        )
        replies.append(ctx)
        previous = ctx

    def check() -> None:
        if should_stop is not None and should_stop():
            raise _RunStopped(_STOPPED)

    if script.context:
        agent.set_context(**script.context)
    if script.opens:
        check()
        record(None, agent.open_chat().response)
    for line in script.lines:
        check()
        record(line, agent.chat(line).response)
    return replies


async def _play_async(
    agent: Chatter,
    script: ConversationScript,
    replies: list[ReplyContext] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[ReplyContext]:
    """Async twin of `_play`.

    A conversation stays strictly in order — each turn answers the
    one before it. Only whole conversations run concurrently, which
    is why *should_stop* matters more here: the conversations still
    in flight when the provider stops answering end at their next
    turn instead of each sending it and waiting out every retry.

    :param agent: A fresh agent, conversation empty.
    :param script: The script to play.
    :param replies: Filled turn by turn, so a caller that passes
        its own list still holds the partial conversation when a
        turn raises. A fresh list when None.
    :param should_stop: Asked before each turn — see `_play`.
    :return: The replies, which is *replies* when one was given.
    :raises _RunStopped: When *should_stop* ended it.
    """
    replies = [] if replies is None else replies
    previous: ReplyContext | None = replies[-1] if replies else None

    def record(prompt: str | None, text: str) -> None:
        nonlocal previous
        ctx = ReplyContext(
            index=len(replies),
            prompt=prompt,
            text=text,
            messages=_split_messages(text),
            previous=previous,
        )
        replies.append(ctx)
        previous = ctx

    def check() -> None:
        if should_stop is not None and should_stop():
            raise _RunStopped(_STOPPED)

    if script.context:
        agent.set_context(**script.context)
    if script.opens:
        check()
        record(None, (await agent.open_chat_async()).response)
    for line in script.lines:
        check()
        record(line, (await agent.chat_async(line)).response)
    return replies


def _plan(
    contract: StyleContract,
) -> list[tuple[str, ConversationScript]]:
    """Every conversation the contract will play, labelled, in order."""
    return [
        (f"{script.id}#{repeat + 1}", script)
        for repeat in range(contract.repeats)
        for script in contract.scripts
    ]


def _fresh_agent(
    agent_factory: Callable[[], Chatter],
    calls: list[CallRecord],
) -> Chatter:
    """A new agent whose every LLM call lands in *calls*.

    Captures the guard's checks and any regenerations too, so the
    report can price a run rather than only count its replies.
    """
    agent = agent_factory()
    existing = agent.call_logger

    def collect(record: CallRecord, _prior=existing) -> None:
        calls.append(record)
        if _prior is not None:
            _prior(record)

    agent._set_attr("call_logger", collect)
    return agent


async def _close_agents(agents: Sequence[Chatter]) -> None:
    """Close the async clients of every agent a run built.

    An agent without `aclose` has nothing to close. Each agent is
    closed once, however many times the factory returned it.
    """
    seen: set[int] = set()
    for agent in agents:
        close = getattr(agent, "aclose", None)
        if close is None or id(agent) in seen:
            continue
        seen.add(id(agent))
        await close()


def _assemble(
    contract: StyleContract,
    runs: Sequence[tuple[str, Sequence[ReplyContext]]],
    calls: Sequence[CallRecord],
    lost: Sequence[LostRun] = (),
    stopped_early: bool = False,
    variants: Mapping[str, Mapping[str, str]] | None = None,
) -> StyleReport:
    """Score every rule over the runs and build the report.

    :param variants: What each run's agent drew, by label. Only the
        scored runs are kept.
    """
    scored = {label for label, _ in runs}
    return StyleReport(
        results=tuple(_score(rule, runs) for rule in contract.rules),
        replies=tuple(ctx for _, replies in runs for ctx in replies),
        calls=tuple(calls),
        lost=tuple(lost),
        stopped_early=stopped_early,
        variants={
            label: dict(drawn)
            for label, drawn in (variants or {}).items()
            if label in scored and drawn
        },
    )


def _drawn(agent: Chatter) -> dict[str, str]:
    """The `prompt_variants` texts *agent* drew, if it has any."""
    return dict(getattr(agent, "variant_values", None) or {})


def run_style_contract(
    agent_factory: Callable[[], Chatter],
    contract: StyleContract,
    stop_after_unavailable: int | None = 2,
) -> StyleReport:
    """Run every script *repeats* times and score every rule.

    A factory rather than an agent, because each run needs a
    conversation that starts empty. Building a fresh agent also
    keeps a run from inheriting whatever state the last one left.

    Conversations run one after another. `run_style_contract_async`
    runs them concurrently, which is the same measurement in a
    fraction of the wall clock — a contract meant to be run on
    every version wants that one.

    **A conversation that raises loses only itself.** The
    provider dropping one connection used to end the invocation
    and discard every conversation already scored: on one agent
    that was fifteen minutes and most of seventeen cents thrown
    away for one failed turn, with nothing recorded. A run that
    dies is reported in `StyleReport.lost` and left unscored, and
    the ones that finished are still scored. The invocation only
    raises when *nothing* finished, and then it raises the first
    error it hit, because a report over no runs would pass every
    budget and say nothing.

    Recovery is at the conversation rather than at the turn: the
    engine already retries the provider call and counts what that
    cost in `LLMMetrics.retries`, so a turn arriving here has
    exhausted that mechanism and repeating it buys little. What
    was actually lost was the other nineteen conversations.

    **It stops when the provider stops answering.** Once
    *stop_after_unavailable* conversations in a row are lost to a
    timeout or a dropped connection that used every retry, no
    further conversation starts. On 14th September 2026 DeepSeek
    accepted requests for hours and answered none, and every
    conversation spent its whole retry budget finding that out.
    The ones not played are in `StyleReport.lost`, and
    `StyleReport.stopped_early` says why.

    :param agent_factory: Returns a ready agent. Called once per
        script per repeat.
    :param contract: The rules, scripts and repeat count.
    :param stop_after_unavailable: How many conversations in a row
        may go unanswered before the run stops. A finished
        conversation, or one lost to any other error, resets the
        count. None never stops.
    :return: The report, over however many runs finished.
    :raises AgentConfigurationError: If *stop_after_unavailable* is
        below 1.
    :raises Exception: The first error, if no run finished at all.
    """
    tracker = _StopTracker(stop_after_unavailable)
    runs: list[tuple[str, list[ReplyContext]]] = []
    lost: list[LostRun] = []
    calls: list[CallRecord] = []
    variants: dict[str, dict[str, str]] = {}
    first_error: Exception | None = None
    for label, script in _plan(contract):
        if tracker.stopped:
            lost.append(LostRun(label=label, error=_NOT_PLAYED, replies=0))
            continue
        logger.debug(f"Style contract: {label}")
        partial: list[ReplyContext] = []
        try:
            agent = _fresh_agent(agent_factory, calls)
            variants[label] = _drawn(agent)
            _play(agent, script, partial)
        except Exception as exc:
            logger.warning(
                f"Style contract: {label} stopped after "
                f"{len(partial)} replies and is not scored ({exc})"
            )
            lost.append(
                LostRun(label=label, error=str(exc), replies=len(partial))
            )
            tracker.lost(exc)
            if first_error is None:
                first_error = exc
            continue
        tracker.finished()
        runs.append((label, partial))
    if not runs and first_error is not None:
        raise first_error
    return _assemble(contract, runs, calls, lost, tracker.stopped, variants)


async def run_style_contract_async(
    agent_factory: Callable[[], Chatter],
    contract: StyleContract,
    concurrency: int = 5,
    stop_after_unavailable: int | None = 2,
) -> StyleReport:
    """Run the contract with several conversations at once.

    **The conversations were always independent.** Each gets its
    own agent from the factory and shares no state with the
    others, so running them one at a time bought nothing and cost
    everything: on one agent, 20 conversations took about 17
    minutes serially, which is too slow to run on every version
    and therefore too slow to be run at all.

    Turns *within* one conversation stay strictly ordered — each
    answers the one before it — so this changes the wall clock and
    nothing about what is measured. The report is identical to the
    sequential one, run for run, including the labels.

    **A conversation that raises loses only itself**, exactly as
    in the sequential runner — see there for why recovery sits at
    the conversation rather than at the turn.

    **It closes every agent it built before returning**, while the
    loop is still running. Left to the garbage collector, each
    agent's async HTTP clients try to close on a loop that has
    already shut, and every `asyncio.run` of this printed an *"Event
    loop is closed"* traceback after a clean run. The agents stay
    usable; their next async call opens fresh connections.

    :param agent_factory: Returns a ready agent. Called once per
        script per repeat.
    :param contract: The rules, scripts and repeat count.
    :param concurrency: How many conversations to keep in flight.
        The ceiling is the provider's rate limit, not this
        machine — every one of these is waiting on a network call.
    :param stop_after_unavailable: How many conversations in a row
        may go unanswered before the run stops — see
        `run_style_contract`. Conversations still in flight then
        end before their next turn.
    :return: The report, over however many runs finished.
    :raises AgentConfigurationError: If *concurrency* is below 1.
    :raises Exception: The first error, if no run finished at all.
    """
    if concurrency < 1:
        raise AgentConfigurationError(
            f"run_style_contract_async: concurrency must be at "
            f"least 1, got {concurrency}."
        )
    tracker = _StopTracker(stop_after_unavailable)
    calls: list[CallRecord] = []
    built: list[Chatter] = []
    variants: dict[str, dict[str, str]] = {}
    limit = asyncio.Semaphore(concurrency)

    async def play(
        label: str, script: ConversationScript
    ) -> tuple[str, list[ReplyContext], Exception | None]:
        async with limit:
            if tracker.stopped:
                return label, [], _RunStopped(_NOT_PLAYED)
            logger.debug(f"Style contract: {label}")
            agent = _fresh_agent(agent_factory, calls)
            built.append(agent)
            variants[label] = _drawn(agent)
            partial: list[ReplyContext] = []
            try:
                await _play_async(
                    agent, script, partial, lambda: tracker.stopped
                )
            except Exception as exc:
                logger.warning(
                    f"Style contract: {label} stopped after "
                    f"{len(partial)} replies and is not scored ({exc})"
                )
                tracker.lost(exc)
                return label, partial, exc
            tracker.finished()
            return label, partial, None

    # gather preserves argument order, so the runs come back in the
    # same order the sequential version would have produced them.
    # Each conversation catches its own error rather than letting
    # gather cancel the siblings that were still in flight.
    played = await asyncio.gather(
        *(play(label, script) for label, script in _plan(contract))
    )
    await _close_agents(built)
    runs = [
        (label, replies) for label, replies, error in played if error is None
    ]
    lost = [
        LostRun(label=label, error=str(error), replies=len(replies))
        for label, replies, error in played
        if error is not None
    ]
    if not runs and lost:
        raise next(
            error
            for _, _, error in played
            if error is not None and not isinstance(error, _RunStopped)
        )
    return _assemble(contract, runs, calls, lost, tracker.stopped, variants)


def _score(
    rule: StyleRule,
    runs: Sequence[tuple[str, Sequence[ReplyContext]]],
) -> RuleResult:
    """Apply one rule to every reply, keeping each run separate.

    Scores each run on its own and pools the totals afterwards,
    rather than pooling first. The pooled figure is identical
    either way; what pooling first destroys is the ability to say
    whether the runs agreed.

    :param rule: The rule to apply.
    :param runs: ``(label, replies)`` per play of one script.
    :return: The rule's pooled and per-run scores.
    """
    examples: list[str] = []
    per_run: list[RunScore] = []
    # A check that compares conversations with each other declares
    # `bind_runs`, and is given every run before it scores one.
    bind = getattr(rule.check, "bind_runs", None)
    check = rule.check if bind is None else bind(runs)
    for label, replies in runs:
        violations = 0
        applicable = 0
        for ctx in replies:
            verdict = check(ctx)
            if verdict is NOT_APPLICABLE:
                continue
            applicable += 1
            if verdict:
                violations += 1
                if len(examples) < 3:
                    examples.append(ctx.text)
        per_run.append(
            RunScore(
                label=label,
                violations=violations,
                applicable=applicable,
            )
        )
    return RuleResult(
        rule=rule,
        violations=sum(r.violations for r in per_run),
        applicable=sum(r.applicable for r in per_run),
        examples=tuple(examples),
        per_run=tuple(per_run),
    )


def _contexts(
    turns: Iterable[tuple[str | None, str]],
) -> list[ReplyContext]:
    """Build one conversation's contexts, numbered from zero."""
    replies: list[ReplyContext] = []
    previous: ReplyContext | None = None
    for prompt, text in turns:
        ctx = ReplyContext(
            index=len(replies),
            prompt=prompt,
            text=text,
            messages=_split_messages(text),
            previous=previous,
        )
        replies.append(ctx)
        previous = ctx
    return replies


def score_transcript(
    contract: StyleContract,
    turns: Iterable[tuple[str | None, str]],
) -> StyleReport:
    """Score one conversation that has already happened.

    For a transcript recorded elsewhere — a past read, a real
    conversation, a run someone else made. Makes no LLM calls, so
    the report carries no cost.

    :param contract: The rules to apply. Its scripts are ignored.
    :param turns: ``(what the counterpart wrote, what the agent
        replied)`` pairs in order. Use None for the counterpart on
        a turn the agent opened.
    :return: The report.
    """
    return score_transcripts(contract, [turns])


def score_transcripts(
    contract: StyleContract,
    conversations: Iterable[Iterable[tuple[str | None, str]]],
) -> StyleReport:
    """Score several recorded conversations together.

    Each conversation is numbered from zero and its first reply has
    nothing before it, so rules that read position or the previous
    reply behave as they would in a live run. **Pooling several
    conversations into one flat list instead would misjudge exactly
    those rules** — a second conversation's legitimate opening
    greeting reads as a greeting mid-conversation, and its first
    reply reads as following the last reply of the one before.

    :param contract: The rules to apply. Its scripts are ignored.
    :param conversations: One sequence of turns per conversation.
    :return: One report over all of them.
    """
    runs = [
        (f"conversation#{i + 1}", _contexts(turns))
        for i, turns in enumerate(conversations)
    ]
    return StyleReport(
        results=tuple(_score(rule, runs) for rule in contract.rules),
        replies=tuple(ctx for _, replies in runs for ctx in replies),
    )


__all__ = [
    "CHECK_FACTORIES",
    "ConversationScript",
    "LostRun",
    "ReplyContext",
    "RuleResult",
    "RunScore",
    "StyleContract",
    "StyleReport",
    "StyleRule",
    "build_rule",
    "ends_with_question",
    "capitalises_after_stop",
    "copies_example",
    "first_char_lowercase",
    "forbidden_pattern",
    "forbidden_text",
    "greets_after_opening_with",
    "load_style_contract",
    "max_emoji",
    "max_line_chars",
    "max_messages",
    "repeats_opener",
    "repeats_question_ending",
    "reproduces_verbatim",
    "run_style_contract",
    "run_style_contract_async",
    "score_transcript",
    "score_transcripts",
]
