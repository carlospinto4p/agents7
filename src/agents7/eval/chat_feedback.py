"""Standardised feedback on a chat agent, and what it feeds.

The third instrument. `style.py` asks *did a rule stop holding?* and
answers it mechanically. `chat_review.py` asks *is the agent good
company?* and puts a person in the best position to judge. Neither
produces the thing that actually improves the agent: **the
operator's correction of a specific line, and the rule it becomes.**

That correction has been made by hand for as long as the agent has
existed, in two files with two shapes and no way to read either
mechanically:

- `data/golden/…md` holds **rewrites** -- "this is what he should
  have said", with the original left unedited elsewhere.
- `docs/voice-feedback.md` holds **rules** -- "he is too verbose and
  does not mirror her" -- each traced to the config file and the
  version that shipped it, and deliberately including conversations
  that were *not* accepted as targets: store the corrections, not
  the conversations.

Both say the same thing about themselves: *not yet
machine-readable*. This module is that format. One entry carries a
rewrite, a rule, or both, because a real correction is usually both
-- conversation A's eight rewrites each became a config rule.

Three things are derived from one file, none of them a second
source of truth:

- `rule_changes` -- what to change in the config, grouped by the
  file it lands in, with the unshipped ones called out.
- `eval_rows` -- corrected turns as targets, and lines marked right
  as examples to imitate. The machine-readable half the golden set
  has been waiting for.
- `render_feedback` -- the operator's own report, grouped by theme.

Reading one::

    from agents7.eval.chat_feedback import load_feedback

    feedback = load_feedback("docs/reads/2026-09-20/feedback.yml")
    for change in feedback.rule_changes():
        print(change.lands_in, len(change.rules))

**Feedback is a separate file from the read's `read.yml`, on
purpose.** A read is the judgement of one version: it is published
as a page, it is argued over, and it is finished. Feedback outlives
it -- a correction taken from a conversation nobody accepted is
still binding on every later version. They share the turn anchors
and nothing else.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agents7.exceptions import AgentConfigurationError


#: What the operator thought of a line. `right` is not a neutral
#: absence of criticism -- it marks a line good enough to imitate,
#: which is how the recipes exchange became a style example.
VERDICTS: dict[str, str] = {
    "right": "Right",
    "weak": "Could be better",
    "wrong": "Wrong",
}

#: Verdicts whose entries are corrections rather than praise.
CORRECTIONS = ("weak", "wrong")


@dataclass(frozen=True)
class FeedbackEntry:
    """One correction, on one turn or on a whole conversation.

    Every field past the verdict is optional because the two shapes
    this replaces are both legitimate on their own: a rewrite with
    no rule stated yet, and a structural rule that no single line
    can carry. What is *not* optional is saying which it is.

    :param conversation: The recording's `conversation_id`. A read
        holds several, so a turn number alone does not locate a
        line.
    :param turn: Which turn, or None for a note about the whole
        conversation. "He is too verbose and does not mirror her"
        is the motivating case: it is the most valuable kind of
        feedback and it anchors to no single line.
    :param verdict: One of `VERDICTS`.
    :param he_said: The line as the agent wrote it. Kept beside the
        rewrite so the pair reads as a correction without having to
        open the recording, and so a rewrite cannot silently drift
        from what it was correcting.
    :param should_say: What it should have said. None for a
        structural note, or when the operator knows a line is wrong
        without wanting to write its replacement.
    :param why: The reasoning. This is the part that stops a
        settled question being reopened, so it outranks the rewrite
        in value even though it is optional.
    :param rule: The instruction the correction became, in the
        config's own voice.
    :param lands_in: The config file the rule belongs in -- the
        character's file when it is about voice, the goal's when it
        is about the conversation. Free text, because only the
        agent knows its own layout.
    :param shipped: The version that shipped the rule, once it has.
        Empty means outstanding, which is what `rule_changes`
        reports on.
    """

    conversation: str
    verdict: str
    turn: int | None = None
    he_said: str = ""
    should_say: str = ""
    why: str = ""
    rule: str = ""
    lands_in: str = ""
    shipped: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise AgentConfigurationError(
                f"verdict must be one of "
                f"{', '.join(sorted(VERDICTS))}, "
                f"not `{self.verdict}`."
            )
        if not (self.should_say or self.rule or self.why):
            raise AgentConfigurationError(
                f"feedback on `{self.conversation}` turn "
                f"{self.turn} says nothing: it needs a rewrite, a "
                f"rule, or at least a reason."
            )

    @property
    def is_correction(self) -> bool:
        """True when this asks for a change."""
        return self.verdict in CORRECTIONS

    @property
    def is_structural(self) -> bool:
        """True when it is about the conversation, not one line."""
        return self.turn is None

    @property
    def outstanding(self) -> bool:
        """True when it states a rule that has not shipped yet."""
        return bool(self.rule) and not self.shipped

    def to_dict(self) -> dict[str, Any]:
        """Serialise, dropping the fields left empty."""
        raw: dict[str, Any] = {
            "conversation": self.conversation,
            "verdict": self.verdict,
        }
        if self.turn is not None:
            raw["turn"] = self.turn
        for name in (
            "he_said",
            "should_say",
            "why",
            "rule",
            "lands_in",
            "shipped",
        ):
            value = getattr(self, name)
            if value:
                raw[name] = value
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FeedbackEntry":
        """Rebuild one entry from its YAML mapping."""
        try:
            conversation = raw["conversation"]
            verdict = raw["verdict"]
        except (KeyError, TypeError) as exc:
            raise AgentConfigurationError(
                f"a feedback entry needs `conversation` and "
                f"`verdict`; got {raw!r}."
            ) from exc
        turn = raw.get("turn")
        return cls(
            conversation=str(conversation),
            verdict=str(verdict),
            turn=None if turn is None else int(turn),
            he_said=_text(raw.get("he_said", "")),
            should_say=_text(raw.get("should_say", "")),
            why=_text(raw.get("why", "")),
            rule=_text(raw.get("rule", "")),
            lands_in=str(raw.get("lands_in", "")),
            shipped=str(raw.get("shipped", "")),
        )


@dataclass(frozen=True)
class RuleChange:
    """Every rule destined for one config file.

    :param lands_in: The config file, or `unassigned` when entries
        did not say. Unassigned is surfaced rather than hidden: a
        rule with nowhere to go is the one most likely to be lost.
    :param entries: The entries that produced those rules.
    """

    lands_in: str
    entries: tuple[FeedbackEntry, ...]

    @property
    def rules(self) -> tuple[str, ...]:
        """The rule text of each entry, in order."""
        return tuple(e.rule for e in self.entries)

    @property
    def outstanding(self) -> tuple[FeedbackEntry, ...]:
        """Those whose rule has not shipped."""
        return tuple(e for e in self.entries if e.outstanding)


@dataclass(frozen=True)
class Feedback:
    """One review's worth of standardised feedback.

    :param read_id: The read it was taken from.
    :param agent: The agent's name.
    :param agent_version: The version reviewed.
    :param reviewed: ISO date the operator judged it.
    :param golden: Conversation ids accepted as targets to imitate
        in full. Everything else contributes corrections only --
        the distinction `voice-feedback.md` makes when it says to
        store the corrections and not the conversations.
    :param entries: The feedback itself.
    """

    read_id: str
    agent: str
    agent_version: str = "unknown"
    reviewed: str = ""
    golden: tuple[str, ...] = ()
    entries: tuple[FeedbackEntry, ...] = ()

    def for_turn(
        self, conversation: str, turn: int
    ) -> tuple[FeedbackEntry, ...]:
        """Entries anchored to one turn of one conversation.

        :param conversation: The recording's id.
        :param turn: The turn index.
        :return: Its entries, in file order.
        """
        return tuple(
            e
            for e in self.entries
            if e.conversation == conversation and e.turn == turn
        )

    def structural(
        self, conversation: str | None = None
    ) -> tuple[FeedbackEntry, ...]:
        """Entries about a whole conversation rather than a line.

        :param conversation: Narrow to one conversation, or None
            for all of them.
        :return: The structural entries.
        """
        return tuple(
            e
            for e in self.entries
            if e.is_structural
            and (conversation is None or e.conversation == conversation)
        )

    def rule_changes(self) -> tuple[RuleChange, ...]:
        """What to change in the config, grouped by file.

        The actionable output: `voice-feedback.md`'s job, derived
        rather than hand-written.

        :return: One `RuleChange` per destination file, files named
            first and `unassigned` last.
        """
        buckets: dict[str, list[FeedbackEntry]] = {}
        for entry in self.entries:
            if not entry.rule:
                continue
            buckets.setdefault(entry.lands_in or "unassigned", []).append(
                entry
            )
        ordered = sorted(
            buckets, key=lambda name: (name == "unassigned", name)
        )
        return tuple(
            RuleChange(lands_in=name, entries=tuple(buckets[name]))
            for name in ordered
        )

    def eval_rows(self) -> tuple[dict[str, Any], ...]:
        """Corrected and exemplary turns, as machine-readable rows.

        The half the golden set has been waiting for. A row is a
        target for how the agent should write at that point in that
        conversation: `target` is the corrected line for a
        correction, and the agent's own line for one marked right,
        because a line good enough to imitate is a target already.

        Only turns yield rows. A structural note has no line to
        score against, so it is left to `rule_changes`.

        :return: One row per usable entry.
        """
        rows = []
        for entry in self.entries:
            if entry.is_structural:
                continue
            target = entry.should_say or (
                entry.he_said if entry.verdict == "right" else ""
            )
            if not target:
                continue
            rows.append(
                {
                    "agent": self.agent,
                    "agent_version": self.agent_version,
                    "read_id": self.read_id,
                    "conversation": entry.conversation,
                    "turn": entry.turn,
                    "verdict": entry.verdict,
                    "golden": entry.conversation in self.golden,
                    "original": entry.he_said,
                    "target": target,
                    "why": entry.why,
                }
            )
        return tuple(rows)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the whole file."""
        return {
            "read_id": self.read_id,
            "agent": self.agent,
            "agent_version": self.agent_version,
            "reviewed": self.reviewed,
            "golden": list(self.golden),
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Feedback":
        """Rebuild from a parsed YAML mapping."""
        if not isinstance(raw, dict):
            raise AgentConfigurationError(
                f"feedback is a mapping, not {type(raw).__name__}."
            )
        try:
            read_id = raw["read_id"]
            agent = raw["agent"]
        except KeyError as exc:
            raise AgentConfigurationError(
                f"feedback needs `read_id` and `agent` ({exc})."
            ) from exc
        return cls(
            read_id=str(read_id),
            agent=str(agent),
            agent_version=str(raw.get("agent_version", "unknown")),
            reviewed=str(raw.get("reviewed", "")),
            golden=tuple(str(g) for g in raw.get("golden") or ()),
            entries=tuple(
                FeedbackEntry.from_dict(e) for e in raw.get("entries") or ()
            ),
        )


def _text(value: Any) -> str:
    """Collapse YAML's folded whitespace into one paragraph."""
    return " ".join(str(value).split()) if value else ""


def load_feedback(source: str | Path) -> Feedback:
    """Read a feedback file.

    :param source: Path to the YAML.
    :return: The feedback.
    :raises AgentConfigurationError: If it is unusable.
    """
    path = Path(source)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise AgentConfigurationError(
            f"{path} is not a readable feedback file ({exc})."
        ) from exc
    return Feedback.from_dict(raw)


def save_feedback(feedback: Feedback, path: str | Path) -> Path:
    """Write a feedback file, creating its directory.

    Round-trips with `load_feedback`, which is what lets an editing
    UI read a file, add an entry and write it back without the
    operator losing anything they wrote by hand.

    :param feedback: What to write.
    :param path: Where to write it.
    :return: The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(
            feedback.to_dict(),
            allow_unicode=True,
            sort_keys=False,
            width=72,
        ),
        encoding="utf-8",
    )
    return target


def write_eval_rows(feedback: Feedback, path: str | Path) -> Path:
    """Write `eval_rows` as JSONL.

    :param feedback: The feedback to derive rows from.
    :param path: Where to write them.
    :return: The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in feedback.eval_rows()
        ),
        encoding="utf-8",
    )
    return target


def render_feedback(feedback: Feedback) -> str:
    """The operator's report, as Markdown.

    Grouped the way the decisions are actually made: what has to
    change in each config file first, then the corrections that
    produced them, then the lines worth imitating.

    :param feedback: The feedback to report on.
    :return: Markdown, ready to read or to append to a running
        record.
    """
    lines = [
        f"# Feedback — {feedback.agent} v{feedback.agent_version}",
        "",
        f"From the read `{feedback.read_id}`"
        + (f", reviewed {feedback.reviewed}." if feedback.reviewed else "."),
        "",
    ]
    if feedback.golden:
        lines += [
            "Accepted as targets to imitate in full: "
            + ", ".join(f"`{name}`" for name in feedback.golden)
            + ". Every other conversation here contributes "
            "corrections only.",
            "",
        ]
    else:
        lines += [
            "No conversation was accepted as a target. Store the "
            "corrections, not the conversations.",
            "",
        ]

    lines += ["## Rules to change", ""]
    changes = feedback.rule_changes()
    if not changes:
        lines += ["No rule changes.", ""]
    for change in changes:
        outstanding = len(change.outstanding)
        lines.append(
            f"### `{change.lands_in}` — {len(change.entries)} rule(s)"
            + (f", {outstanding} not shipped" if outstanding else "")
        )
        lines.append("")
        for entry in change.entries:
            where = (
                "structural" if entry.is_structural else f"turn {entry.turn}"
            )
            mark = "" if entry.shipped else " **(not shipped)**"
            lines.append(
                f"- {entry.rule} _({entry.conversation}, {where})_{mark}"
            )
            if entry.why:
                lines.append(f"  - Why: {entry.why}")
        lines.append("")

    corrections = [
        e for e in feedback.entries if e.is_correction and e.should_say
    ]
    lines += ["## Corrected lines", ""]
    if not corrections:
        lines += ["No line rewrites.", ""]
    for entry in corrections:
        lines.append(
            f"- **{entry.conversation}, turn {entry.turn}** "
            f"({VERDICTS[entry.verdict].lower()})"
        )
        if entry.he_said:
            lines.append(f"  - He said: _{entry.he_said}_")
        lines.append(f"  - Should say: _{entry.should_say}_")
        if entry.why:
            lines.append(f"  - Why: {entry.why}")
    lines.append("")

    good = [e for e in feedback.entries if e.verdict == "right"]
    if good:
        lines += ["## Worth imitating", ""]
        for entry in good:
            where = (
                "structural" if entry.is_structural else f"turn {entry.turn}"
            )
            lines.append(
                f"- **{entry.conversation}, {where}**"
                + (f": _{entry.he_said}_" if entry.he_said else "")
            )
            if entry.why:
                lines.append(f"  - {entry.why}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_feedback_report(feedback: Feedback, path: str | Path) -> Path:
    """Render the report and write it.

    :param feedback: The feedback to report on.
    :param path: Where to write it.
    :return: The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_feedback(feedback), encoding="utf-8")
    return target


def merge_entries(
    feedback: Feedback, entries: Iterable[FeedbackEntry]
) -> Feedback:
    """Add entries to a feedback file, replacing matches.

    An editing UI re-sends whatever the operator has on screen, and
    a turn can carry more than one entry, so "replace" is keyed on
    the turn rather than on the entry: every existing entry for a
    touched (conversation, turn) is dropped and the incoming ones
    take its place. Turns nobody touched are left exactly as they
    were, including anything hand-written.

    :param feedback: The file as it stands.
    :param entries: What the operator just wrote.
    :return: A new `Feedback`; the original is untouched.
    """
    incoming = tuple(entries)
    touched = {(e.conversation, e.turn) for e in incoming}
    kept = tuple(
        e for e in feedback.entries if (e.conversation, e.turn) not in touched
    )
    return Feedback(
        read_id=feedback.read_id,
        agent=feedback.agent,
        agent_version=feedback.agent_version,
        reviewed=feedback.reviewed,
        golden=feedback.golden,
        entries=kept + incoming,
    )


__all__ = [
    "CORRECTIONS",
    "VERDICTS",
    "Feedback",
    "FeedbackEntry",
    "RuleChange",
    "load_feedback",
    "merge_entries",
    "render_feedback",
    "save_feedback",
    "write_eval_rows",
    "write_feedback_report",
]
