"""Read a chat agent end to end, and publish what a person saw.

The other instrument. `style.py` asks *did a rule stop holding?* and
answers it mechanically, over a scripted counterpart, with no person
in the loop. This module serves the question that has no score: **is
the agent good company?** A person writes the other side one line at
a time, reads what came back, and writes down what they thought. See
the project rule `chat-agent-validation.md`.

**Recording and rendering are separate, and deliberately so.**
Recording holds a real conversation, so it costs money and cannot be
repeated -- at any temperature above zero a re-run is a different
conversation, and the metrics it produced are gone the moment the
process ends. Rendering reads recordings back off disk and makes a
page out of them. So a conversation is paid for once and can appear
on any number of later pages, beside conversations recorded months
apart at different versions, without spending anything again.

Recording one conversation::

    from agents7.agents.guarded_scene import GuardedSceneChatter
    from agents7.eval.chat_review import ConversationRecorder

    agent = GuardedSceneChatter.from_yaml("my_agent/config.yml")
    rec = ConversationRecorder(agent, "warm", "docs/reads/warm.json")
    print(rec.open())              # the agent writes first
    print(rec.say("hola, que tal"))

Every turn is written to the file as it happens, so a conversation
that crashes on turn nine keeps its first eight.

Rendering a page from recordings plus a person's notes::

    from agents7.eval.chat_review import render_review

    html = render_review("docs/reads/2026-09-06.yml")

The spec is YAML because the notes in it are the deliverable: they
are somebody's judgement about a turn, they get argued over, and
they belong in version control beside the transcripts rather than
inside a script.
"""

import html as html_mod
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import yaml

from agents7.exceptions import AgentConfigurationError


# Note kinds, and the label each shows. The four are fixed: a reader
# learns them once and then reads every page in the same colours.
NOTE_KINDS: dict[str, str] = {
    "worked": "Worked",
    "call": "Your call",
    "defect": "Defect",
    "guard": "Guard",
}

# Verdict stamps. `new` is for a finding nobody was looking for,
# which is usually the most valuable thing a read produces.
STAMPS: dict[str, str] = {
    "pass": "Fixed",
    "fail": "Still here",
    "new": "New",
}


# ─────────────────────────── recording ────────────────────────────


@dataclass(frozen=True)
class GuardAttempt:
    """One guard check on one reply.

    A guarded reply costs two LLM calls and this is the second one;
    a rejected one costs four. Both are recorded, because the
    expensive turns are exactly the ones a single figure hides.

    :param attempt: 0 for the first check, 1+ for a retry.
    :param accepted: Whether the guard let the reply through.
    :param reason: The guard's verdict, or None if it did not parse.
    :param text: The reply that was judged.
    :param fix: The guard's suggested rewrite, when it rejected.
    :param used_fix: True when the shipped reply came from `fix`.
    :param parse_failed: True when the guard's output was unusable.
    :param metrics: The guard call's own `LLMMetrics`, as a dict.
        None only when a caller built this without one.
    """

    attempt: int
    accepted: bool
    reason: str | None
    text: str
    fix: str | None
    used_fix: bool = False
    parse_failed: bool = False
    metrics: dict[str, Any] | None = None

    @property
    def fix_repeats_the_rejection(self) -> bool:
        """True when the suggested fix is the rejected text again.

        Worth surfacing rather than hiding: it means the repair path
        had nothing in it, and whatever saved the turn was the
        regeneration rather than the guard.
        """
        if self.accepted or not self.fix:
            return False
        return self.fix.strip() == self.text.strip()

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the recording file."""
        return {
            "attempt": self.attempt,
            "accepted": self.accepted,
            "reason": self.reason,
            "text": self.text,
            "fix": self.fix,
            "used_fix": self.used_fix,
            "parse_failed": self.parse_failed,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GuardAttempt":
        """Rebuild from a recording file."""
        return cls(
            attempt=int(raw["attempt"]),
            accepted=bool(raw["accepted"]),
            reason=raw.get("reason"),
            text=raw["text"],
            fix=raw.get("fix"),
            used_fix=bool(raw.get("used_fix", False)),
            parse_failed=bool(raw.get("parse_failed", False)),
            metrics=raw.get("metrics"),
        )


@dataclass(frozen=True)
class TurnRecord:
    """One reply, with everything measured about producing it.

    :param index: The agent's replies, numbered from zero.
    :param prompt: What the counterpart wrote, or None when the
        agent opened the conversation.
    :param reply: The reply exactly as the agent produced it.
    :param chatter: The chatter call's `LLMMetrics`, as a dict.
    :param guard: Every guard check the turn made, in order.
    :param wall_seconds: Clock time measured around the whole call,
        by the caller. Chatter plus guard will not add up to it on a
        retried turn, because a regenerated reply is not in the
        chatter's own metrics -- the difference is reported rather
        than hidden.
    """

    index: int
    prompt: str | None
    reply: str
    chatter: dict[str, Any]
    guard: tuple[GuardAttempt, ...] = ()
    wall_seconds: float = 0.0

    @property
    def messages(self) -> tuple[str, ...]:
        """The reply split into the messages a client would show.

        A persona told to write two or three short lines produces one
        reply and several messages, and joining them back together
        hides the rhythm the config is trying to produce.
        """
        return tuple(
            line.strip() for line in self.reply.split("\n") if line.strip()
        )

    @property
    def prompt_words(self) -> int:
        """How many words the counterpart wrote. 0 if it opened."""
        return len(self.prompt.split()) if self.prompt else 0

    @property
    def chatter_seconds(self) -> float:
        """What the agent's own call took."""
        return float(self.chatter.get("total_duration") or 0.0)

    @property
    def guard_seconds(self) -> float:
        """What every guard check took, added up."""
        return sum(
            float((a.metrics or {}).get("total_duration") or 0.0)
            for a in self.guard
        )

    @property
    def gap_seconds(self) -> float:
        """Wall clock the two halves do not account for."""
        return self.wall_seconds - self.chatter_seconds - self.guard_seconds

    @property
    def chatter_cents(self) -> float | None:
        """The agent's own call, in cents. None when unpriced."""
        return self.chatter.get("cost_cents")

    @property
    def guard_cents(self) -> float | None:
        """Every guard check, in cents. None when any is unpriced."""
        if not self.guard:
            return 0.0
        costs = [(a.metrics or {}).get("cost_cents") for a in self.guard]
        if any(c is None for c in costs):
            return None
        return sum(float(c) for c in costs)

    @property
    def cents(self) -> float | None:
        """The whole turn, in cents. None when anything is unpriced."""
        parts = (self.chatter_cents, self.guard_cents)
        if any(p is None for p in parts):
            return None
        return sum(float(p) for p in parts)

    @property
    def rejections(self) -> tuple[GuardAttempt, ...]:
        """The guard checks that said no."""
        return tuple(a for a in self.guard if not a.accepted)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the recording file."""
        return {
            "index": self.index,
            "prompt": self.prompt,
            "reply": self.reply,
            "chatter": self.chatter,
            "guard": [a.to_dict() for a in self.guard],
            "wall_seconds": self.wall_seconds,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TurnRecord":
        """Rebuild from a recording file."""
        return cls(
            index=int(raw["index"]),
            prompt=raw.get("prompt"),
            reply=raw["reply"],
            chatter=raw.get("chatter") or {},
            guard=tuple(
                GuardAttempt.from_dict(a) for a in raw.get("guard", ())
            ),
            wall_seconds=float(raw.get("wall_seconds") or 0.0),
        )


@dataclass(frozen=True)
class Recording:
    """One whole conversation, as it happened, with its metrics.

    This is the artefact that costs money. It carries what it needs
    to be understood years later beside conversations recorded at
    other versions: which agent, which version of it, when, and --
    read off the provider's own replies rather than off the config --
    which models actually answered.

    :param conversation_id: Short name, unique within one read.
    :param agent: The agent's name.
    :param agent_version: The package version it ran at, as a
        string. `unknown` when the caller could not determine it.
    :param recorded: ISO-8601 UTC timestamp of the first turn.
    :param turns: The replies, in order.
    :param variants: The text drawn for each of the agent's
        `prompt_variants`. Two conversations of one config can be
        shown different examples, so a read has to be able to say
        which this one saw.
    :param context: What the agent's `context_params` resolved to
        when the conversation opened -- the nickname it was given,
        the time of day it thought it was, any hint computed for
        it. Unrecoverable afterwards: a resolver runs inside
        substitution and a re-run at temperature is a different
        conversation. Empty for a recording made before agents7
        v9.17.0, which is not the same as a conversation that ran
        with no context, so a reader is shown "not recorded".
    :param facts: The character facts the store held when the
        conversation opened, as plain strings. A reply can be
        shaped by something said in an earlier conversation, and a
        reviewer cannot tell that from the transcript alone.
    """

    conversation_id: str
    agent: str
    agent_version: str
    recorded: str
    turns: tuple[TurnRecord, ...] = ()
    variants: dict[str, str] = field(default_factory=dict)
    context: dict[str, str] = field(default_factory=dict)
    facts: tuple[str, ...] = ()

    @property
    def chatter_model(self) -> str | None:
        """The model that answered, or None if they disagreed.

        Read from each reply rather than from the engine, because a
        provider may resolve an alias and a fallback answers under
        its own name. None when more than one model answered -- the
        per-turn truth is on each record.
        """
        return _one_model(t.chatter.get("model") for t in self.turns)

    @property
    def guard_model(self) -> str | None:
        """The model the guard ran on, by the same rule."""
        return _one_model(
            (a.metrics or {}).get("model")
            for t in self.turns
            for a in t.guard
        )

    @property
    def chatter_seconds(self) -> float:
        """Every agent call in the conversation."""
        return sum(t.chatter_seconds for t in self.turns)

    @property
    def guard_seconds(self) -> float:
        """Every guard call in the conversation."""
        return sum(t.guard_seconds for t in self.turns)

    @property
    def wall_seconds(self) -> float:
        """Clock time across every turn."""
        return sum(t.wall_seconds for t in self.turns)

    @property
    def cents(self) -> float | None:
        """What the conversation cost. None when anything is unpriced."""
        costs = [t.cents for t in self.turns]
        if any(c is None for c in costs):
            return None
        return sum(float(c) for c in costs)

    @property
    def guard_calls(self) -> int:
        """How many times the guard was asked."""
        return sum(len(t.guard) for t in self.turns)

    @property
    def rejections(self) -> int:
        """How many of those said no."""
        return sum(len(t.rejections) for t in self.turns)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the whole recording."""
        return {
            "conversation_id": self.conversation_id,
            "agent": self.agent,
            "agent_version": self.agent_version,
            "recorded": self.recorded,
            "variants": dict(self.variants),
            "context": dict(self.context),
            "facts": list(self.facts),
            "turns": [t.to_dict() for t in self.turns],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Recording":
        """Rebuild a recording from its file."""
        return cls(
            conversation_id=raw["conversation_id"],
            agent=raw.get("agent", "unknown"),
            agent_version=str(raw.get("agent_version", "unknown")),
            recorded=raw.get("recorded", ""),
            variants=dict(raw.get("variants") or {}),
            context=dict(raw.get("context") or {}),
            facts=tuple(raw.get("facts") or ()),
            turns=tuple(
                TurnRecord.from_dict(t) for t in raw.get("turns", ())
            ),
        )


def _one_model(names: Iterable[str | None]) -> str | None:
    """The single model every call named, or None."""
    seen = {n for n in names if n}
    return seen.pop() if len(seen) == 1 else None


def save_recording(recording: Recording, path: str | Path) -> Path:
    """Write a recording to disk, replacing what was there.

    :param recording: What to write.
    :param path: Destination. Parents are created.
    :return: The path written to.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(recording.to_dict(), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return target


def load_recording(path: str | Path) -> Recording:
    """Read a recording back.

    :param path: The file to read.
    :return: The recording.
    :raises AgentConfigurationError: If it will not parse.
    """
    target = Path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        return Recording.from_dict(raw)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise AgentConfigurationError(
            f"{target} is not a chat recording ({exc})."
        ) from exc


def _package_version(agent_name: str) -> str:
    """The installed package version an agent's name resolves to.

    **Not** the agent's own `version` field, which for a
    config-loaded agent is the config *schema* version -- a
    recording came out stamped `1` that way once, and a recording's
    version is what tells two reads apart.

    :param agent_name: The agent's `name`, taken as the name of the
        distribution that ships it.
    :return: That distribution's installed version.
    :raises AgentConfigurationError: If no installed package answers
        to the name.
    """
    try:
        return distribution_version(agent_name)
    except PackageNotFoundError as exc:
        raise AgentConfigurationError(
            f"Could not read a package version for {agent_name!r}: "
            "no installed package answers to that name. Pass "
            "`agent_version` explicitly -- a recording's version is "
            "what tells two reads apart, so it is never guessed."
        ) from exc


class ConversationRecorder:
    """Drives one conversation and records every measurement.

    Wraps a chatter, attaching the guard callback itself so a caller
    cannot forget it -- **nothing recorded here can be reconstructed
    afterwards**, and a read whose runner logged durations only has
    already lost its token counts once.

    The counterpart's lines are written one at a time by whoever is
    running the read, in reaction to what the agent just said. Do not
    script them in advance: a reply written before the agent spoke
    proves nothing about whether it is good company. (The scripted
    counterpart belongs to the other instrument, `style.py`.)

    :param agent: A `Chatter` or any subclass. Guard events are
        captured when it has a `guard_callback` field.
    :param conversation_id: Short name, unique within one read.
    :param path: Where to write. Rewritten after every turn, so a
        crash keeps the turns that already happened.
    :param agent_version: The package version, for the record. When
        not given, read from the installed distribution the agent's
        `name` resolves to -- never from the agent's own `version`,
        which for a config-loaded agent is the config *schema*
        version.
    :raises AgentConfigurationError: If `agent_version` is not given
        and no installed package answers to the agent's name.
    """

    def __init__(
        self,
        agent: Any,
        conversation_id: str,
        path: str | Path,
        agent_version: str | None = None,
    ) -> None:
        self._events: list[GuardAttempt] = []
        self._agent = self._instrument(agent)
        self._path = Path(path)
        self._turns: list[TurnRecord] = []
        self._recording = Recording(
            conversation_id=conversation_id,
            agent=getattr(agent, "name", "unknown"),
            agent_version=(
                agent_version
                if agent_version is not None
                else _package_version(str(getattr(agent, "name", "")))
            ),
            recorded=datetime.now(UTC).isoformat(timespec="seconds"),
            # From the instrumented copy, which is the agent that
            # talks — building it drew the variants afresh.
            variants=getattr(self._agent, "variant_values", {}),
            # Captured once, here, because a `callable` resolver
            # returns a fresh value on every read: asking again at
            # the end of the conversation could report a different
            # time of day from the one the agent opened on.
            context=dict(getattr(self._agent, "resolved_context", {})),
            facts=tuple(
                fact.text
                for fact in getattr(self._agent, "character_facts", ())
            ),
        )

    def _instrument(self, agent: Any) -> Any:
        """Attach the guard callback, when the agent has one."""
        if "guard_callback" not in getattr(agent, "__dataclass_fields__", {}):
            return agent

        def collect(event: Any) -> None:
            self._events.append(
                GuardAttempt(
                    attempt=event.attempt,
                    accepted=event.accepted,
                    reason=getattr(event.reason, "value", event.reason),
                    text=event.response_text,
                    fix=event.fix,
                    used_fix=event.used_fix,
                    parse_failed=event.parse_failed,
                    metrics=(
                        event.metrics.model_dump() if event.metrics else None
                    ),
                )
            )

        # `with_overrides` and not a bare `replace`: the copy is the
        # agent that talks, and `replace` drops whatever the caller
        # passed to `set_context`.
        return agent.with_overrides(guard_callback=collect)

    @property
    def agent(self) -> Any:
        """The instrumented agent, for a caller that needs it."""
        return self._agent

    @property
    def recording(self) -> Recording:
        """What has been recorded so far."""
        return replace(self._recording, turns=tuple(self._turns))

    def open(self) -> str:
        """Let the agent write first, and record the turn.

        :return: What it said.
        """
        return self._run(None)

    def say(self, message: str) -> str:
        """Send one line from the counterpart, and record the reply.

        :param message: What the other person wrote.
        :return: What the agent said back.
        """
        return self._run(message)

    def _run(self, message: str | None) -> str:
        self._events = []
        started = time.perf_counter()
        if message is None:
            response = self._agent.open_chat()
        else:
            response = self._agent.chat(message)
        wall = time.perf_counter() - started
        self._turns.append(
            TurnRecord(
                index=len(self._turns),
                prompt=message,
                reply=response.response,
                chatter=response.metrics.model_dump(),
                guard=tuple(self._events),
                wall_seconds=wall,
            )
        )
        self.save()
        return response.response

    def save(self) -> Path:
        """Write everything recorded so far.

        Called after every turn already. Exposed for a caller driving
        one turn per process.
        """
        return save_recording(self.recording, self._path)


# ──────────────────────── the read's judgement ────────────────────


@dataclass(frozen=True)
class Note:
    """One thing a person noticed, anchored to the turn it is about.

    Notes sit under the exchange they describe rather than in a list
    at the end. A finding a reader has to scroll back to match
    against a turn number is a finding they will skim.

    :param turn: Which reply, numbered as the recording numbers them.
    :param kind: One of `worked`, `call`, `defect`, `guard`.
    :param text: What was noticed. Inline HTML is allowed and is not
        escaped -- these are written by whoever ran the read.
    """

    turn: int
    kind: str
    text: str

    def __post_init__(self) -> None:
        if self.kind not in NOTE_KINDS:
            raise AgentConfigurationError(
                f"Note on turn {self.turn}: kind must be one of "
                f"{', '.join(sorted(NOTE_KINDS))}. Got {self.kind!r}."
            )


@dataclass(frozen=True)
class ReadVerdict:
    """One conclusion, shown before any of the evidence.

    Named for the read rather than plainly, because
    `style_history.Verdict` already means something else in this
    package: whether a rate left the band its own history set.

    :param claim: What was judged, as a short phrase.
    :param stamp: `pass`, `fail`, or `new`.
    :param detail: A sentence or two of why.
    :param label: Overrides the stamp's own wording.
    """

    claim: str
    stamp: str
    detail: str
    label: str | None = None

    def __post_init__(self) -> None:
        if self.stamp not in STAMPS:
            raise AgentConfigurationError(
                f"ReadVerdict {self.claim!r}: stamp must be one of "
                f"{', '.join(sorted(STAMPS))}. Got {self.stamp!r}."
            )

    @property
    def wording(self) -> str:
        """What the stamp reads on the page."""
        return self.label or STAMPS[self.stamp]


@dataclass(frozen=True)
class ReviewedConversation:
    """One recording, plus what the reader made of it.

    :param recording: The conversation as it happened.
    :param title: Heading, e.g. `Cold -- she gives him nothing`.
    :param tab: Short name for this conversation's tab. Taken from
        the part of `title` before the dash when not given.
    :param note: A sentence on what this conversation was for.
    :param notes: The turn-anchored findings.
    """

    recording: Recording
    title: str
    tab: str = ""
    note: str = ""
    notes: tuple[Note, ...] = ()

    def notes_for(self, turn: int) -> tuple[Note, ...]:
        """Every note anchored to one reply."""
        return tuple(n for n in self.notes if n.turn == turn)


@dataclass(frozen=True)
class Read:
    """One whole read: the conversations and the judgement of them.

    :param read_id: This read's own name in the history, so it can
        be told apart from the ones before it. **A read must never
        be its own previous read**, which is what happens without
        this the moment its figures have been recorded once.
    :param title: The page's name.
    :param eyebrow: The line above it -- agent, version, date.
    :param standfirst: What this read set out to answer.
    :param caveat: Who wrote the other side. **Say plainly that the
        counterpart is invented**; it is written by somebody who
        knows what is being tested, so a warm conversation is nearer
        a best case than an average one.
    :param conversations: In the order they should be read.
    :param verdicts: The conclusions, shown first.
    :param closing: Paragraphs after the transcripts. Inline HTML.
    :param pricing: The footnote naming models and rates.
    :param comparison: Rows for the numbers table, each a mapping of
        `measure`, `previous`, `current`, `reading`. Filled from the
        read history when one is configured.
    :param style_report: The mechanical score of these transcripts,
        when a style contract was given.
    """

    title: str
    eyebrow: str
    standfirst: str
    caveat: str
    conversations: tuple[ReviewedConversation, ...]
    verdicts: tuple[ReadVerdict, ...] = ()
    closing: tuple[str, ...] = ()
    pricing: str = ""
    read_id: str = ""
    comparison: tuple[dict[str, str], ...] = ()
    previous_counts: dict[str, str] = field(default_factory=dict)
    style_report: Any = None

    @property
    def turns(self) -> tuple[TurnRecord, ...]:
        """Every reply across every conversation."""
        return tuple(t for c in self.conversations for t in c.recording.turns)

    @property
    def cents(self) -> float | None:
        """What the whole read cost. None when anything is unpriced."""
        costs = [c.recording.cents for c in self.conversations]
        if any(c is None for c in costs):
            return None
        return sum(float(c) for c in costs)

    @property
    def guard_share(self) -> float | None:
        """Guard time over wall clock, as a fraction."""
        wall = sum(t.wall_seconds for t in self.turns)
        if wall <= 0:
            return None
        return sum(t.guard_seconds for t in self.turns) / wall


# ─────────────────────────── the history ──────────────────────────


@dataclass(frozen=True)
class ReadRecord:
    """What one read measured, so the next one can say what moved.

    A read is one sample of a system that varies, so two of these
    are two samples and not a measurement -- the page says so where
    it prints the comparison. The history exists to stop a figure
    being misremembered, not to license a claim about a trend.

    :param read_id: Short name, unique per agent.
    :param date: ISO-8601 date the read was published.
    :param agent: Which agent.
    :param agent_version: Which version of it.
    :param counts: Measure name to value, as strings so a rate and a
        count can sit in one table.
    """

    read_id: str
    date: str
    agent: str
    agent_version: str
    counts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the history file."""
        return {
            "read_id": self.read_id,
            "date": self.date,
            "agent": self.agent,
            "agent_version": self.agent_version,
            "counts": self.counts,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ReadRecord":
        """Rebuild from the history file."""
        return cls(
            read_id=raw["read_id"],
            date=raw.get("date", ""),
            agent=raw.get("agent", "unknown"),
            agent_version=str(raw.get("agent_version", "unknown")),
            counts=dict(raw.get("counts") or {}),
        )


def append_read(record: ReadRecord, path: str | Path) -> Path:
    """Append one read to a history file, as JSON Lines.

    Append-only, for the same reason the style history is: it is the
    only thing that can say what a figure was last time, and tidying
    an inconvenient read away destroys exactly that.

    :param record: The read to store.
    :param path: History file, created with its parents if absent.
    :return: The path written to.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return target


def load_read_history(
    path: str | Path,
    agent: str | None = None,
) -> tuple[ReadRecord, ...]:
    """Read a history file, oldest first.

    :param path: History file. A missing file is an empty history,
        not an error -- a first read has none.
    :param agent: Keep only this agent's reads.
    :return: The records.
    :raises AgentConfigurationError: If a line will not parse.
    """
    target = Path(path)
    if not target.is_file():
        return ()
    records: list[ReadRecord] = []
    for number, line in enumerate(
        target.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(ReadRecord.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise AgentConfigurationError(
                f"{target}: line {number} is not a read record ({exc})."
            ) from exc
    if agent is not None:
        records = [r for r in records if r.agent == agent]
    return tuple(records)


#: The measures the comparison table shows, in order. Everything
#: else `measure_read` returns is a style rule, and those get their
#: own table with a budget column rather than being listed twice.
HEADLINE_MEASURES: tuple[str, ...] = (
    "Replies",
    "Replies containing a question",
    "Guard rejections",
    "Guard share of wall clock",
    "Cost",
)


def measure_read(read: Read) -> dict[str, str]:
    """The figures a read contributes to the history.

    Derived once, here, so the prose and the page cannot disagree
    about a count. Counting by hand has already put two wrong
    numbers into a committed document.

    :param read: The read to measure.
    :return: Measure name to formatted value. The keys in
        `HEADLINE_MEASURES` come first; any others are style rule
        ids, keyed so the next read can show what each rule scored
        last time.
    """
    turns = read.turns
    replies = len(turns)
    counts: dict[str, str] = {"Replies": str(replies)}
    if replies:
        with_q = sum(1 for t in turns if "?" in t.reply)
        counts["Replies containing a question"] = _pct(with_q, replies)
    counts["Guard rejections"] = str(sum(len(t.rejections) for t in turns))
    share = read.guard_share
    if share is not None:
        counts["Guard share of wall clock"] = f"{round(100 * share)}%"
    cents = read.cents
    counts["Cost"] = "unpriced" if cents is None else f"{cents:.2f}c"

    report = read.style_report
    if report is not None:
        for result in report.results:
            if result.applicable:
                counts[result.rule.id] = _pct(
                    result.violations, result.applicable
                )
    return counts


def _pct(part: int, whole: int) -> str:
    return f"{round(100 * part / whole)}%" if whole else "n/a"


# ──────────────────────────── the spec ────────────────────────────


def load_read(source: str | Path) -> Read:
    """Build a `Read` from its YAML spec.

    The spec names the recordings to show -- **any recordings**,
    including ones made months earlier at another version, which is
    the point of keeping recording and rendering apart. Paths inside
    it resolve relative to the spec's own directory.

    :param source: Path to the spec.
    :return: The read, with its recordings loaded and, when the spec
        names a style contract, the mechanical score alongside.
    :raises AgentConfigurationError: If the spec is unusable.
    """
    path = Path(source)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise AgentConfigurationError(
            f"{path} is not a readable read spec ({exc})."
        ) from exc
    if not isinstance(raw, dict):
        raise AgentConfigurationError(
            f"{path}: a read spec is a mapping, not {type(raw).__name__}."
        )

    base = path.parent
    entries = raw.get("conversations") or []
    if not entries:
        raise AgentConfigurationError(
            f"{path}: a read needs at least one conversation."
        )

    conversations = []
    for entry in entries:
        recording = load_recording(base / entry["recording"])
        conversations.append(
            ReviewedConversation(
                recording=recording,
                title=entry.get("title", recording.conversation_id),
                tab=entry.get("tab", ""),
                note=_text(entry.get("note", "")),
                notes=tuple(
                    Note(
                        turn=int(n["turn"]),
                        kind=n["kind"],
                        text=_text(n["text"]),
                    )
                    for n in entry.get("notes") or []
                ),
            )
        )

    report = None
    contract_path = raw.get("style_contract")
    if contract_path:
        from agents7.eval.style import load_style_contract, score_transcripts

        contract = load_style_contract(base / contract_path)
        report = score_transcripts(
            contract,
            [
                [(t.prompt, t.reply) for t in conv.recording.turns]
                for conv in conversations
            ],
        )

    read = Read(
        title=raw.get("title", "Chat review"),
        eyebrow=raw.get("eyebrow", ""),
        standfirst=_text(raw.get("standfirst", "")),
        caveat=_text(raw.get("caveat", _DEFAULT_CAVEAT)),
        conversations=tuple(conversations),
        verdicts=tuple(
            ReadVerdict(
                claim=v["claim"],
                stamp=v["stamp"],
                detail=_text(v["detail"]),
                label=v.get("label"),
            )
            for v in raw.get("verdicts") or []
        ),
        closing=tuple(_text(p) for p in raw.get("closing") or []),
        pricing=_text(raw.get("pricing", "")),
        read_id=str(raw.get("read_id") or path.parent.name),
        style_report=report,
    )

    history_path = raw.get("history")
    if history_path:
        agent = read.conversations[0].recording.agent
        history = [
            record
            for record in load_read_history(base / history_path, agent)
            if record.read_id != read.read_id
        ]
        previous = dict(history[-1].counts) if history else {}
        read = replace(
            read,
            previous_counts=previous,
            comparison=_comparison(
                read, previous, raw.get("comparison_readings") or {}
            ),
        )
    return read


def _text(value: Any) -> str:
    """Collapse YAML's folded whitespace into one paragraph."""
    return " ".join(str(value).split())


_DEFAULT_CAVEAT = (
    "<strong>The other person is invented.</strong> The counterpart "
    "was written one line at a time, in reaction to what the agent "
    "actually said, by somebody who knew what was being tested. That "
    "makes a warm conversation nearer a best case than an average "
    "one. This read narrows the open questions; it does not close "
    "them."
)


def _comparison(
    read: Read,
    previous: dict[str, str],
    readings: dict[str, str],
) -> tuple[dict[str, str], ...]:
    """Build the numbers table against the most recent prior read."""
    current = measure_read(read)
    return tuple(
        {
            "measure": measure,
            "previous": previous.get(measure, "—"),
            "current": current[measure],
            "reading": readings.get(measure, ""),
        }
        for measure in HEADLINE_MEASURES
        if measure in current
    )


# ────────────────────────── the page ──────────────────────────────


def render_review(source: str | Path | Read) -> str:
    """Render a read as a self-contained HTML page.

    Makes no LLM calls and reads nothing but the files the spec
    names, so it is free and can be run as many times as the wording
    needs. Publish it: a conversation that is only readable as a
    diff will not get read closely, and being read closely is the
    entire point.

    :param source: A `Read`, or the path to a YAML spec.
    :return: The page, ready to write to a file.
    """
    from agents7.eval._review_theme import FONT_LINK, REVIEW_CSS

    read = source if isinstance(source, Read) else load_read(source)
    conversations = [
        _conversation_payload(conv) for conv in read.conversations
    ]
    payload = {"conversations": conversations}
    tabs = _tabs(read, conversations)

    parts = [
        f"<title>{esc(read.title)}</title>",
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="preconnect" href="https://fonts.gstatic.com" '
        "crossorigin>",
        FONT_LINK,
        f"<style>{REVIEW_CSS}\n{_EXTRA_CSS}</style>",
        '<div class="wrap">',
        _masthead(read),
        _control(tabs),
        _panel("verdicts", _verdicts(read)),
        _panel("numbers", _numbers(read)),
        '<div id="threads"></div>',
        _panel("closing", _closing(read)),
        "</div>",
        _script(payload),
    ]
    return "\n".join(p for p in parts if p)


def _tabs(read: Read, conversations: list[dict]) -> list[dict]:
    """The tab bar, in reading order.

    The conversations are not marked `review`, so they are the only
    tabs left standing in Read mode -- a first pass is the
    transcripts and nothing else, which is what the toggle is for.
    """
    tabs: list[dict] = []
    if read.verdicts:
        tabs.append({"id": "verdicts", "label": "Verdicts", "review": True})
    if read.comparison or read.style_report is not None:
        tabs.append({"id": "numbers", "label": "Numbers", "review": True})
    tabs.extend(
        {
            "id": f"conv-{index}",
            "label": conv["tab"],
            "review": False,
        }
        for index, conv in enumerate(conversations)
    )
    if read.closing or read.pricing:
        tabs.append({"id": "closing", "label": "Findings", "review": True})
    return tabs


def _panel(panel_id: str, inner: str) -> str:
    """Wrap a section as a tab panel, or nothing when it is empty."""
    if not inner:
        return ""
    return (
        f'<section class="panel" id="panel-{panel_id}" role="tabpanel" '
        f'aria-labelledby="tab-{panel_id}" hidden>{inner}</section>'
    )


def esc(text: str) -> str:
    """Escape text that came from a transcript, never from a note."""
    return html_mod.escape(str(text), quote=False)


# The template zeroes the right padding on figure cells, which was
# right when a figure was the last column. Here a reading follows it.
_EXTRA_CSS = """
  .scroller table td.num,
  .scroller table th.num { padding-right: 20px; }

  .panel > .section:first-child,
  .panel > .conv:first-child { padding-top: 30px; }

  /* What the conversation ran with. Reads as the header of a chat
     window rather than as an annotation, so it survives Read mode. */
  .ctx {
    margin: 0 0 22px;
    padding: 12px 16px;
    background: var(--surface);
    border: 1px solid var(--line-soft);
    border-radius: var(--radius);
    display: grid;
    /* 300px, not 240: a resolved model name is ~30 characters and a
       narrower track broke `openai_compat::deepseek-flash` mid-word. */
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2px 26px;
  }

  .ctx-row {
    display: flex;
    gap: 10px;
    align-items: baseline;
    padding: 3px 0;
    min-width: 0;
  }

  .ctx dt {
    flex: 0 0 auto;
    color: var(--text-3);
    font-size: .9rem;
    letter-spacing: .02em;
  }

  .ctx dd {
    margin: 0;
    min-width: 0;
    color: var(--text-2);
    font-size: .95rem;
    overflow-wrap: anywhere;
  }

  @media (max-width: 560px) {
    .ctx { grid-template-columns: 1fr; }
  }
"""


def _masthead(read: Read) -> str:
    return (
        '<header class="mast">'
        f'<div class="eyebrow">{esc(read.eyebrow)}</div>'
        f"<h1>{esc(read.title)}</h1>"
        f'<p class="standfirst">{read.standfirst}</p>'
        f'<div class="caveat"><div>{read.caveat}</div></div>'
        "</header>"
    )


def _verdicts(read: Read) -> str:
    if not read.verdicts:
        return ""
    cards = "".join(
        '<article class="verdict"><div class="verdict-head">'
        f"<h3>{esc(v.claim)}</h3>"
        f'<span class="stamp {v.stamp}">{esc(v.wording)}</span>'
        f"</div><p>{v.detail}</p></article>"
        for v in read.verdicts
    )
    return (
        '<section class="section"><h2>The verdicts</h2>'
        f'<div class="verdicts">{cards}</div></section>'
    )


def _numbers(read: Read) -> str:
    if not read.comparison and read.style_report is None:
        return ""
    blocks = [
        '<section class="section"><h2>The numbers</h2>',
        '<p class="section-note">Counted from the recordings, not by '
        "hand. Each read is one sample of something that varies, so a "
        "difference between two of them is two samples and not a "
        "measurement.</p>",
        f'<div class="figures">{_figures(read)}</div>',
    ]
    if read.comparison:
        rows = "".join(
            f"<tr><td>{esc(row['measure'])}</td>"
            f'<td class="num">{esc(row["previous"])}</td>'
            f'<td class="num">{esc(row["current"])}</td>'
            f"<td>{esc(row['reading'])}</td></tr>"
            for row in read.comparison
        )
        blocks.append(
            '<div class="scroller"><table><thead><tr>'
            "<th>Measure</th>"
            '<th class="num">Previous read</th>'
            '<th class="num">This read</th>'
            "<th>Reading</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
    if read.style_report is not None:
        blocks.append(_rules_table(read.style_report, read.previous_counts))
    blocks.append("</section>")
    return "".join(blocks)


def _figures(read: Read) -> str:
    turns = read.turns
    rejections = sum(len(t.rejections) for t in turns)
    calls = sum(len(t.guard) for t in turns)
    cents = read.cents
    share = read.guard_share
    cells: list[tuple[str, str, str]] = [
        (
            str(len(turns)),
            "replies",
            f"across {len(read.conversations)} conversations",
        ),
        (
            f"{rejections} / {calls}",
            "guard rejections",
            "over all turns",
        ),
        (
            "unpriced" if cents is None else f"{cents:.2f}c",
            "total cost",
            "agent plus guard",
        ),
        (
            "—" if share is None else f"{round(100 * share)}%",
            "guard share",
            "of wall clock",
        ),
    ]
    return "".join(
        f'<div class="fig"><span class="n">{esc(n)}</span>'
        f'<span class="k">{esc(k)}</span>'
        f'<span class="d">{esc(d)}</span></div>'
        for n, k, d in cells
    )


def _rules_table(report: Any, previous: dict[str, str]) -> str:
    rows = []
    for result in report.results:
        if result.applicable:
            score = f"{result.violations} / {result.applicable}"
            rate = _pct(result.violations, result.applicable)
        else:
            score, rate = "—", "not exercised"
        budget = (
            "watched only"
            if result.rule.budget is None
            else f"{100 * result.rule.budget:.1f}%"
        )
        rows.append(
            f"<tr><td>{esc(result.rule.id)}</td>"
            f'<td class="num">{score}</td>'
            f'<td class="num">{rate}</td>'
            f'<td class="num">{esc(previous.get(result.rule.id, "—"))}</td>'
            f"<td>{budget}</td></tr>"
        )
    return (
        '<h2 style="margin-top:34px">The contract\'s own score</h2>'
        '<p class="section-note">The agent\'s style rules applied to '
        "these transcripts, with no LLM calls and nothing re-run. It "
        "answers whether a rule stopped holding, which is a different "
        "question from whether the agent is good company. Two reads "
        "are two samples: a rule needs a run history behind it before "
        "a move between them means anything.</p>"
        '<div class="scroller"><table><thead><tr>'
        "<th>Rule</th>"
        '<th class="num">Broke</th>'
        '<th class="num">Rate</th>'
        '<th class="num">Last read</th>'
        "<th>Budget</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _control(tabs: list[dict]) -> str:
    legend = "".join(
        f'<span><i style="background: var(--{kind})"></i>{label}</span>'
        for kind, label in NOTE_KINDS.items()
    )
    buttons = "".join(
        f'<button type="button" role="tab" id="tab-{t["id"]}" '
        f'data-panel="{t["id"]}"'
        + (' data-review="1"' if t["review"] else "")
        + f' aria-controls="panel-{t["id"]}" aria-selected="false">'
        f"{esc(t['label'])}</button>"
        for t in tabs
    )
    return (
        '<div class="control">'
        f'<div class="tabs" role="tablist" aria-label="Sections">'
        f"{buttons}</div>"
        '<div class="switch" role="group" aria-label="Display mode">'
        '<button type="button" id="btn-read" aria-pressed="false">'
        "Read</button>"
        '<button type="button" id="btn-review" aria-pressed="true">'
        "Review</button></div>"
        f'<div class="legend">{legend}</div></div>'
    )


def _closing(read: Read) -> str:
    if not read.closing and not read.pricing:
        return ""
    body = "".join(f"<p>{p}</p>" for p in read.closing)
    pricing = f'<p class="pricing">{read.pricing}</p>' if read.pricing else ""
    heading = "<h2>What this read leaves you with</h2>" if body else ""
    return (
        f'<section class="section">{heading}'
        f'<div class="closing">{body}</div>{pricing}</section>'
    )


def read_payload(source: "Read | str | Path") -> dict[str, Any]:
    """A whole read as plain data, ready for any renderer.

    `render_review` writes a self-contained page and is the right
    answer when the read is something to publish and read. A UI that
    lets somebody *click a turn* cannot use that page -- it has to
    build the transcript itself -- and the moment it does, the
    template exists twice and the two drift. Four reads came out
    with four designs before the generator existed; the same thing
    would happen again one layer down.

    So the data is the contract. `render_review` builds its
    conversations from the same function this does, and anything
    that renders a read elsewhere eats this rather than
    reconstructing it. What is *not* shared is the visual template,
    which is the renderer's own business.

    :param source: A `Read`, or the path to a YAML spec.
    :return: The read's text, verdicts, measures and conversations,
        plus the note kinds and stamps a renderer needs to label
        them. JSON-serialisable throughout.
    """
    read = source if isinstance(source, Read) else load_read(source)
    return {
        "read_id": read.read_id,
        "title": read.title,
        "eyebrow": read.eyebrow,
        "standfirst": read.standfirst,
        "caveat": read.caveat,
        "pricing": read.pricing,
        "closing": list(read.closing),
        "note_kinds": dict(NOTE_KINDS),
        "stamps": dict(STAMPS),
        "verdicts": [
            {
                "claim": v.claim,
                "stamp": v.stamp,
                "label": v.wording,
                "detail": v.detail,
            }
            for v in read.verdicts
        ],
        "measures": dict(measure_read(read)),
        "comparison": [dict(row) for row in read.comparison],
        "conversations": [
            _conversation_payload(conv) for conv in read.conversations
        ],
    }


def _conversation_payload(conv: ReviewedConversation) -> dict[str, Any]:
    running = 0.0
    unpriced = False
    turns = []
    for turn in conv.recording.turns:
        cost = turn.cents
        if cost is None:
            unpriced = True
        else:
            running += cost
        prompt_tokens = int(turn.chatter.get("prompt_tokens") or 0)
        cached = int(turn.chatter.get("cached_prompt_tokens") or 0)
        turns.append(
            {
                "i": turn.index,
                "prompt": turn.prompt,
                "messages": list(turn.messages),
                "chatter_s": round(turn.chatter_seconds, 3),
                "guard_s": round(turn.guard_seconds, 3),
                "wall_s": round(turn.wall_seconds, 3),
                "gap_s": round(turn.gap_seconds, 3),
                "attempts": len(turn.guard),
                "rejected": len(turn.rejections),
                "chatter_c": turn.chatter_cents,
                "guard_c": turn.guard_cents,
                "running_c": None if unpriced else round(running, 5),
                "prompt_t": prompt_tokens,
                "cached_pct": (
                    round(100 * cached / prompt_tokens)
                    if prompt_tokens
                    else None
                ),
                "out_t": int(turn.chatter.get("tokens") or 0),
                "cached": bool(turn.chatter.get("cached")),
                "retries": int(turn.chatter.get("retries") or 0),
                "attempts_detail": [
                    {
                        "accepted": a.accepted,
                        "reason": a.reason or "no verdict",
                        "text": a.text,
                        "used_fix": a.used_fix,
                        "fix_repeats": a.fix_repeats_the_rejection,
                    }
                    for a in turn.guard
                ],
                "notes": [
                    {
                        "kind": n.kind,
                        "label": NOTE_KINDS[n.kind],
                        "text": n.text,
                    }
                    for n in conv.notes_for(turn.index)
                ],
            }
        )
    rec = conv.recording
    return {
        "id": rec.conversation_id,
        "title": conv.title,
        "tab": conv.tab or _tab_label(conv.title),
        "note": conv.note,
        "tag": _conversation_tag(rec),
        "context": _context_rows(rec),
        "turns": turns,
    }


def _context_rows(rec: Recording) -> list[dict[str, Any]]:
    """Everything the conversation ran with, as label/value rows.

    A transcript on its own does not say when it happened, what
    nickname the agent was wearing, what time of day it thought it
    was, or what it already believed about itself from an earlier
    conversation. All of that shapes the replies, and a reviewer
    judging whether a reply is any good needs it in front of them.

    Each row names the `group` it belongs to, so a renderer can lay
    the run, the context, the memory and the variants out apart
    without guessing from a label. The memory row also carries its
    facts one by one as `items`; `value` keeps them joined for a
    renderer that shows only text.

    :param rec: The recording to describe.
    :return: Rows in display order, each `label`, `value` and
        `group`, the memory row with `items` as well.
    """
    rows: list[dict[str, Any]] = [
        {
            "label": "When",
            "value": _format_recorded(rec.recorded),
            "group": "run",
        },
        {
            "label": "Agent",
            "value": f"{rec.agent} v{rec.agent_version}",
            "group": "run",
        },
    ]
    models = [
        ("Model", rec.chatter_model),
        ("Guard model", rec.guard_model),
    ]
    for label, name in models:
        if name:
            rows.append({"label": label, "value": name, "group": "run"})
    for name, value in rec.context.items():
        rows.append(
            {
                "label": name,
                "value": value or "(empty)",
                "group": "context",
            },
        )
    if not rec.context:
        # Distinct from "ran with no context": every recording made
        # before agents7 v9.17.0 has the field and cannot fill it.
        rows.append(
            {
                "label": "Context",
                "value": "not recorded",
                "group": "context",
            }
        )
    if rec.facts:
        rows.append(
            {
                "label": "Knew already",
                "value": " · ".join(rec.facts),
                "items": list(rec.facts),
                "group": "memory",
            }
        )
    for name, value in rec.variants.items():
        rows.append(
            {
                "label": f"variant: {name}",
                "value": value,
                "group": "variants",
            }
        )
    return rows


def _format_recorded(stamp: str) -> str:
    """An ISO timestamp as a person would write it.

    :param stamp: The recording's ISO-8601 UTC timestamp.
    :return: `20 September 2026, 12:51 UTC`, or the raw string
        when it will not parse, or a plain note when it is empty.
    """
    if not stamp:
        return "not recorded"
    try:
        when = datetime.fromisoformat(stamp)
    except ValueError:
        return stamp
    return when.strftime("%-d %B %Y, %H:%M UTC")


def _tab_label(title: str) -> str:
    """A tab-sized name out of a conversation's heading.

    Headings read `Cold -- she gives him nothing`, which is right
    above a transcript and far too long in a tab, so the tab takes
    the part before the dash.
    """
    return title.split("—")[0].split(" - ")[0].strip() or title


def _conversation_tag(rec: Recording) -> str:
    rejected = rec.rejections
    if rejected == 0:
        guard = "no guard rejections"
    elif rejected == 1:
        guard = "1 guard rejection"
    else:
        guard = f"{rejected} guard rejections"
    return f"{len(rec.turns)} replies · {guard}"


def _script(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return (
        "<script>\nconst PAGE = " + data + ";\n" + _RENDERER + "\n</script>"
    )


# The renderer. Kept as one string rather than a build step: the page
# has to be a single self-contained file, and there is no bundler in
# this project to make one.
_RENDERER = r"""
const esc = (s) => String(s).replace(/[&<>]/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

const cents = (v, places) =>
  v === null || v === undefined ? "unpriced" : v.toFixed(places) + "c";

function metaLine(t) {
  const hot = t.wall_s > 6 ? "hot" : "";
  const bits = [
    `<span><b>agent</b> ${t.chatter_s.toFixed(2)}s</span>`,
    `<span><b>guard</b> ${t.guard_s.toFixed(2)}s` +
      (t.attempts > 1 ? ` ×${t.attempts}` : "") + `</span>`,
    `<span><b>gap</b> ${t.gap_s.toFixed(2)}s</span>`,
    `<span class="${hot}"><b>total</b> ${t.wall_s.toFixed(2)}s</span>`,
    `<span><b>cost</b> ${cents(
      t.chatter_c === null || t.guard_c === null
        ? null : t.chatter_c + t.guard_c, 4)} ` +
      `<span style="opacity:.7">(${cents(t.chatter_c, 4)} + ` +
      `${cents(t.guard_c, 4)})</span></span>`,
    `<span><b>running</b> ${cents(t.running_c, 3)}</span>`
  ];
  if (t.prompt_t) {
    const cache = t.cached_pct === null ? "" :
      ` <span style="opacity:.7">(${t.cached_pct}% cached)</span>`;
    bits.push(`<span><b>tokens</b> ${t.prompt_t}p${cache} / ` +
      `${t.out_t}g</span>`);
  }
  if (t.cached) bits.push(`<span><b>served from cache</b></span>`);
  if (t.retries) bits.push(`<span class="hot"><b>retries</b> ` +
    `${t.retries}</span>`);
  return `<div class="meta">${bits.join("")}</div>`;
}

function attemptsBlock(t) {
  if (!t.rejected) return "";
  const items = t.attempts_detail.map(a => {
    const badge = a.accepted
      ? `<span class="ok">accepted</span>`
      : `<span class="rej">rejected · ${esc(a.reason)}</span>`;
    const shipped = a.used_fix ? " <em>(this is what shipped)</em>" : "";
    const same = a.fix_repeats
      ? `<br><em>Its suggested fix was the same text, character for ` +
        `character.</em>` : "";
    return `<li>${badge}<br>${esc(a.text)}${shipped}${same}</li>`;
  }).join("");
  return `<details class="attempts"><summary>Guard: ${t.rejected} ` +
    `rejected of ${t.attempts} checks — open</summary>` +
    `<ol>${items}</ol></details>`;
}

// Stays visible in Read mode. It is context for the chat, the way a
// messenger shows who you are talking to and when — not part of the
// review apparatus the toggle strips.
function contextBlock(conv) {
  if (!conv.context || !conv.context.length) return "";
  const rows = conv.context.map(r =>
    `<div class="ctx-row"><dt>${esc(r.label)}</dt>` +
    `<dd>${esc(r.value)}</dd></div>`).join("");
  return `<dl class="ctx">${rows}</dl>`;
}

function render() {
  const out = [];
  PAGE.conversations.forEach((conv, index) => {
    out.push(`<section class="panel" id="panel-conv-${index}" ` +
      `role="tabpanel" aria-labelledby="tab-conv-${index}" hidden>`);
    out.push(`<section class="conv"><div class="conv-head">` +
      `<h2>${esc(conv.title)}</h2>` +
      `<span class="tag">${esc(conv.tag)}</span></div>` +
      (conv.note ? `<p class="section-note">${conv.note}</p>` : "") +
      contextBlock(conv) +
      `<div class="thread">`);
    for (const t of conv.turns) {
      out.push(`<div class="turn">`);
      if (t.prompt !== null && t.prompt !== undefined) {
        out.push(`<div class="row mine"><div class="bubble">` +
          `${esc(t.prompt)}</div></div>`);
      }
      out.push(`<div class="turnno">turn ${t.i}</div>`);
      for (const m of t.messages) {
        out.push(`<div class="row his"><div class="bubble">` +
          `${esc(m)}</div></div>`);
      }
      out.push(metaLine(t));
      out.push(attemptsBlock(t));
      for (const n of t.notes) {
        out.push(`<div class="note ${n.kind}">` +
          `<span class="lbl">${esc(n.label)}</span>` +
          `<p>${n.text}</p></div>`);
      }
      out.push(`</div>`);
    }
    out.push(`</div></section>`);
    out.push(`</section>`);
  });
  document.getElementById("threads").innerHTML = out.join("");
}

const readBtn = document.getElementById("btn-read");
const reviewBtn = document.getElementById("btn-review");
const tabs = Array.from(document.querySelectorAll(".tabs button"));

function visibleTabs() {
  const reading = document.body.dataset.mode === "read";
  return tabs.filter(t => !(reading && t.dataset.review));
}

function showPanel(id) {
  for (const tab of tabs) {
    const on = tab.dataset.panel === id;
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
    const panel = document.getElementById("panel-" + tab.dataset.panel);
    if (panel) panel.hidden = !on;
  }
  try { localStorage.setItem("chatReview.panel", id); } catch (e) { }
  // In the address bar so one conversation can be linked to
  // directly. replaceState rather than a hash assignment, which
  // would jump the page to the panel.
  try {
    history.replaceState(null, "", "#" + id);
  } catch (e) { }
}

function currentPanel() {
  const on = tabs.find(t => t.getAttribute("aria-selected") === "true");
  return on ? on.dataset.panel : null;
}

for (const tab of tabs) {
  tab.addEventListener("click", () => showPanel(tab.dataset.panel));
  tab.addEventListener("keydown", (event) => {
    const order = visibleTabs();
    const at = order.indexOf(tab);
    let next = null;
    if (event.key === "ArrowRight") next = order[at + 1];
    else if (event.key === "ArrowLeft") next = order[at - 1];
    else if (event.key === "Home") next = order[0];
    else if (event.key === "End") next = order[order.length - 1];
    else return;
    event.preventDefault();
    if (next) { showPanel(next.dataset.panel); next.focus(); }
  });
}

function setMode(mode) {
  document.body.dataset.mode = mode;
  readBtn.setAttribute("aria-pressed", String(mode === "read"));
  reviewBtn.setAttribute("aria-pressed", String(mode === "review"));
  // Read mode drops every tab but the conversations, so a selected
  // Verdicts tab would leave the page showing nothing.
  const open = visibleTabs();
  if (open.length && !open.some(t => t.dataset.panel === currentPanel())) {
    showPanel(open[0].dataset.panel);
  }
  try { localStorage.setItem("chatReview.mode", mode); } catch (e) { }
}

readBtn.addEventListener("click", () => setMode("read"));
reviewBtn.addEventListener("click", () => setMode("review"));

render();

let saved = "review";
try { saved = localStorage.getItem("chatReview.mode") || "review"; }
catch (e) { saved = "review"; }

let panel = decodeURIComponent(location.hash.replace(/^#/, ""));
if (!panel) {
  try { panel = localStorage.getItem("chatReview.panel"); } catch (e) { }
}
if (!panel || !tabs.some(t => t.dataset.panel === panel)) {
  panel = tabs.length ? tabs[0].dataset.panel : null;
}
if (panel) showPanel(panel);
setMode(saved === "read" ? "read" : "review");
"""


def write_review(
    source: str | Path,
    out_path: str | Path,
    history: str | Path | None = None,
    read_id: str | None = None,
) -> Path:
    """Render a spec to a file, optionally recording its figures.

    :param source: The YAML spec.
    :param out_path: Where to write the page.
    :param history: When given, record this read's figures there so
        the next read's comparison column fills itself. A read
        already in the file is left alone, so re-rendering a page --
        which happens every time a note is reworded -- never files
        the same read twice.
    :param read_id: Name for the history entry. Defaults to the
        spec's own `read_id`.
    :return: The page's path.
    """
    read = load_read(source)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_review(read), encoding="utf-8")
    if history is not None:
        first = read.conversations[0].recording
        name = read_id or read.read_id
        already = any(
            record.read_id == name
            for record in load_read_history(history, first.agent)
        )
        if not already:
            append_read(
                ReadRecord(
                    read_id=name,
                    date=datetime.now(UTC).date().isoformat(),
                    agent=first.agent,
                    agent_version=first.agent_version,
                    counts=measure_read(read),
                ),
                history,
            )
    return target


def turns_where(
    read: Read,
    kind: str,
) -> tuple[tuple[str, TurnRecord], ...]:
    """Every turn carrying a note of one kind, with its conversation.

    For a caller writing the closing paragraphs, so a claim about how
    many defects a read found is derived rather than counted.

    :param read: The read to search.
    :param kind: One of `worked`, `call`, `defect`, `guard`.
    :return: `(conversation title, turn)` pairs, in page order.
    """
    found = []
    for conv in read.conversations:
        for turn in conv.recording.turns:
            if any(n.kind == kind for n in conv.notes_for(turn.index)):
                found.append((conv.title, turn))
    return tuple(found)


def dry_turns(read: Read, words: int = 3) -> Sequence[TurnRecord]:
    """Replies to a short message, which is where brevity rules bite.

    :param read: The read to search.
    :param words: How few words counts as short.
    :return: The turns.
    """
    return [
        t
        for t in read.turns
        if t.prompt is not None and t.prompt_words <= words
    ]


__all__ = [
    "NOTE_KINDS",
    "STAMPS",
    "ConversationRecorder",
    "GuardAttempt",
    "Note",
    "Read",
    "ReadRecord",
    "Recording",
    "ReviewedConversation",
    "ReadVerdict",
    "TurnRecord",
    "append_read",
    "dry_turns",
    "load_read",
    "load_read_history",
    "load_recording",
    "measure_read",
    "render_review",
    "save_recording",
    "turns_where",
    "write_review",
]
