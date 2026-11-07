"""Unit tests for recording and rendering a chat agent's read."""

import json
from dataclasses import replace
from importlib.metadata import version as distribution_version

import pytest

from agents7.agents.context import ContextParam
from agents7.exceptions import AgentConfigurationError
from agents7.eval.chat_review import (
    HEADLINE_MEASURES,
    NOTE_KINDS,
    STAMPS,
    ConversationRecorder,
    GuardAttempt,
    Note,
    Read,
    ReadRecord,
    ReadVerdict,
    Recording,
    ReviewedConversation,
    TurnRecord,
    append_read,
    dry_turns,
    load_read,
    load_read_history,
    load_recording,
    measure_read,
    read_payload,
    render_review,
    save_recording,
    turns_where,
    write_review,
)

from helpers import make_guarded_chatter


# ── Fixtures ───────────────────────────────────────────────────────


def metrics(
    duration: float = 1.0,
    cost: float | None = 0.01,
    prompt: int = 100,
    cached: int = 90,
    tokens: int = 10,
) -> dict:
    """One call's metrics, in the shape `LLMMetrics` dumps."""
    return {
        "tokens": tokens,
        "prompt_tokens": prompt,
        "cached_prompt_tokens": cached,
        "total_duration": duration,
        "cost_cents": cost,
        "cached": False,
        "model": "openai_compat::demo-1",
        "retries": 0,
        "retry_duration": 0.0,
    }


def turn(
    index: int = 0,
    prompt: str | None = "hola",
    reply: str = "qué tal",
    guard: tuple[GuardAttempt, ...] = (),
    wall: float = 2.0,
) -> TurnRecord:
    """A turn with one accepted guard check unless told otherwise."""
    if not guard:
        guard = (
            GuardAttempt(
                attempt=0,
                accepted=True,
                reason="in character",
                text=reply,
                fix=None,
                metrics=metrics(duration=0.8, cost=0.005),
            ),
        )
    return TurnRecord(
        index=index,
        prompt=prompt,
        reply=reply,
        chatter=metrics(),
        guard=guard,
        wall_seconds=wall,
    )


@pytest.fixture
def recording() -> Recording:
    return Recording(
        conversation_id="cold",
        agent="demo",
        agent_version="1.2.3",
        recorded="2026-09-06T10:00:00+00:00",
        turns=(
            turn(0, None, "buenas"),
            turn(1, "bien", "vale\n\n¿y tú?"),
        ),
    )


# ── TurnRecord ─────────────────────────────────────────────────────


def test_messages_splits_on_blank_lines():
    record = turn(reply="una\n\ndos\n\ntres")
    assert record.messages == ("una", "dos", "tres")


def test_messages_drops_empty_lines():
    assert turn(reply="\n\nuna\n\n").messages == ("una",)


def test_prompt_words_counts_the_counterpart():
    assert turn(prompt="ya te lo he dicho").prompt_words == 5


def test_prompt_words_is_zero_when_the_agent_opened():
    assert turn(prompt=None).prompt_words == 0


def test_guard_seconds_sums_every_attempt():
    record = turn(
        guard=(
            GuardAttempt(
                0,
                False,
                "off topic",
                "no",
                None,
                metrics=metrics(duration=0.5, cost=0.001),
            ),
            GuardAttempt(
                1,
                True,
                "in character",
                "sí",
                None,
                metrics=metrics(duration=0.7, cost=0.002),
            ),
        )
    )
    assert record.guard_seconds == pytest.approx(1.2)


def test_gap_is_wall_clock_the_two_halves_do_not_explain():
    record = turn(wall=2.5)
    assert record.gap_seconds == pytest.approx(2.5 - 1.0 - 0.8)


def test_turn_cost_adds_agent_and_guard():
    assert turn().cents == pytest.approx(0.015)


def test_an_unpriced_guard_call_makes_the_turn_unpriced():
    record = turn(
        guard=(
            GuardAttempt(
                0,
                True,
                "in character",
                "sí",
                None,
                metrics=metrics(cost=None),
            ),
        )
    )
    assert record.guard_cents is None
    assert record.cents is None


def test_an_unpriced_agent_call_makes_the_turn_unpriced():
    record = TurnRecord(
        index=0,
        prompt="hola",
        reply="hey",
        chatter=metrics(cost=None),
    )
    assert record.cents is None


def test_a_turn_with_no_guard_costs_the_guard_nothing():
    record = TurnRecord(
        index=0, prompt="hola", reply="hey", chatter=metrics()
    )
    assert record.guard_cents == 0.0
    assert record.cents == pytest.approx(0.01)


def test_rejections_are_the_attempts_that_said_no():
    record = turn(
        guard=(
            GuardAttempt(0, False, "off topic", "no", "otra"),
            GuardAttempt(1, True, "in character", "sí", None),
        )
    )
    assert len(record.rejections) == 1
    assert record.rejections[0].attempt == 0


# ── GuardAttempt ───────────────────────────────────────────────────


def test_a_fix_identical_to_the_rejection_is_flagged():
    attempt = GuardAttempt(0, False, "off topic", " hola ", "hola")
    assert attempt.fix_repeats_the_rejection


def test_a_genuine_fix_is_not_flagged():
    attempt = GuardAttempt(0, False, "off topic", "hola", "adiós")
    assert not attempt.fix_repeats_the_rejection


def test_an_accepted_attempt_never_flags_a_repeated_fix():
    attempt = GuardAttempt(0, True, "in character", "hola", "hola")
    assert not attempt.fix_repeats_the_rejection


# ── Recording ──────────────────────────────────────────────────────


def test_model_is_read_from_the_replies(recording):
    assert recording.chatter_model == "openai_compat::demo-1"
    assert recording.guard_model == "openai_compat::demo-1"


def test_model_is_none_when_the_calls_disagree():
    mixed = Recording(
        conversation_id="c",
        agent="demo",
        agent_version="1",
        recorded="",
        turns=(
            TurnRecord(0, None, "a", metrics()),
            TurnRecord(
                1,
                "b",
                "c",
                {**metrics(), "model": "openai_compat::demo-2"},
            ),
        ),
    )
    assert mixed.chatter_model is None


def test_totals_add_across_the_conversation(recording):
    assert recording.chatter_seconds == pytest.approx(2.0)
    assert recording.guard_seconds == pytest.approx(1.6)
    assert recording.wall_seconds == pytest.approx(4.0)
    assert recording.cents == pytest.approx(0.03)
    assert recording.guard_calls == 2
    assert recording.rejections == 0


def test_recording_survives_a_round_trip(tmp_path, recording):
    path = save_recording(recording, tmp_path / "a" / "cold.json")
    assert load_recording(path) == recording


def test_context_and_facts_survive_a_round_trip(tmp_path, recording):
    full = replace(
        recording,
        context={"nick": "Marta_87", "mi_nick": "cronista_del_barrio"},
        facts=("Le gusta ser el último en irse.",),
    )
    path = save_recording(full, tmp_path / "cold.json")
    assert load_recording(path) == full


def test_a_recording_from_before_the_field_loads_empty(tmp_path):
    """A pre-v9.17.0 file has no `context` key and must still load."""
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "conversation_id": "cold",
                "agent": "demo",
                "agent_version": "0.1.0",
                "recorded": "2026-09-06T10:00:00+00:00",
                "turns": [],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_recording(path)
    assert loaded.context == {}
    assert loaded.facts == ()


def test_loading_something_that_is_not_a_recording_explains_itself(
    tmp_path,
):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(AgentConfigurationError, match="not a chat"):
        load_recording(bad)


# ── ConversationRecorder ───────────────────────────────────────────


class _Response:
    def __init__(self, text: str) -> None:
        self.response = text
        self.metrics = _Metrics()


class _Metrics:
    @staticmethod
    def model_dump() -> dict:
        return metrics()


class _Agent:
    """A chatter with no guard: the plainest thing to record."""

    name = "demo"
    version = 3
    __dataclass_fields__: dict = {}

    def __init__(self) -> None:
        self.said: list[str | None] = []

    def open_chat(self) -> _Response:
        self.said.append(None)
        return _Response("buenas")

    def chat(self, message: str) -> _Response:
        self.said.append(message)
        return _Response(f"eco: {message}")


def test_the_recorder_writes_every_turn_as_it_happens(tmp_path):
    path = tmp_path / "warm.json"
    rec = ConversationRecorder(_Agent(), "warm", path, agent_version="0.1")
    assert rec.open() == "buenas"
    assert load_recording(path).turns[0].prompt is None
    assert rec.say("hola") == "eco: hola"
    saved = load_recording(path)
    assert [t.prompt for t in saved.turns] == [None, "hola"]
    assert [t.index for t in saved.turns] == [0, 1]


class _PackagedAgent(_Agent):
    """An agent named after a distribution that is installed."""

    name = "agents7"


def test_the_recorder_stamps_the_agent_and_its_version(tmp_path):
    rec = ConversationRecorder(
        _Agent(), "warm", tmp_path / "w.json", agent_version="0.1"
    )
    rec.open()
    saved = load_recording(tmp_path / "w.json")
    assert saved.agent == "demo"
    assert saved.agent_version == "0.1"


def test_the_version_comes_from_the_installed_package(tmp_path):
    rec = ConversationRecorder(_PackagedAgent(), "warm", tmp_path / "w.json")
    rec.open()
    saved = load_recording(tmp_path / "w.json")
    assert saved.agent_version == distribution_version("agents7")


def test_the_config_schema_version_is_never_stamped(tmp_path):
    """`_PackagedAgent.version` is 3, the config schema version.

    A recording filed under `3` is filed under the schema, not the
    release, and a recording's version is what tells two reads
    apart.
    """
    rec = ConversationRecorder(_PackagedAgent(), "warm", tmp_path / "w.json")
    rec.open()
    assert load_recording(tmp_path / "w.json").agent_version != "3"


def test_an_unresolvable_name_raises_rather_than_guessing(tmp_path):
    with pytest.raises(AgentConfigurationError, match="demo"):
        ConversationRecorder(_Agent(), "warm", tmp_path / "w.json")


def test_an_explicit_version_wins(tmp_path):
    rec = ConversationRecorder(
        _Agent(), "warm", tmp_path / "w.json", agent_version="0.1.47"
    )
    rec.open()
    assert load_recording(tmp_path / "w.json").agent_version == "0.1.47"


def test_wall_clock_is_measured_around_the_call(tmp_path):
    rec = ConversationRecorder(
        _Agent(), "warm", tmp_path / "w.json", agent_version="0.1"
    )
    rec.open()
    assert rec.recording.turns[0].wall_seconds >= 0.0


class _Fact:
    def __init__(self, text: str) -> None:
        self.text = text


class _ContextAgent(_Agent):
    """An agent carrying resolved context and a fact store."""

    resolved_context = {"nick": "Marta_87", "moment": "por la mañana"}
    character_facts = (_Fact("Le gusta llegar el último."),)


def test_the_recorder_captures_the_resolved_context(tmp_path):
    rec = ConversationRecorder(
        _ContextAgent(), "warm", tmp_path / "w.json", agent_version="0.1"
    )
    rec.open()
    saved = load_recording(tmp_path / "w.json")
    assert saved.context == {
        "nick": "Marta_87",
        "moment": "por la mañana",
    }


def test_the_recorder_talks_to_an_agent_that_kept_its_context(tmp_path):
    """The instrumented copy is the one that talks.

    A guarded agent is copied to attach the guard callback. On
    20th September 2026 that copy had lost what `set_context` was
    given, so a read ran on the defaults while the caller's own
    agent still showed the supplied values.
    """
    agent = make_guarded_chatter(
        system_prompt="Hablas con $nick.",
        context_params=(ContextParam(name="nick", default="ella"),),
    )
    agent.set_context(nick="Marta_87")

    rec = ConversationRecorder(
        agent, "warm", tmp_path / "w.json", agent_version="0.1"
    )

    assert rec.agent.resolved_system_prompt == "Hablas con Marta_87."
    assert rec.recording.context == {"nick": "Marta_87"}


def test_the_recorder_captures_the_facts_it_started_with(tmp_path):
    rec = ConversationRecorder(
        _ContextAgent(), "warm", tmp_path / "w.json", agent_version="0.1"
    )
    rec.open()
    saved = load_recording(tmp_path / "w.json")
    assert saved.facts == ("Le gusta llegar el último.",)


def test_an_agent_without_context_records_none(tmp_path):
    rec = ConversationRecorder(
        _Agent(), "warm", tmp_path / "w.json", agent_version="0.1"
    )
    rec.open()
    saved = load_recording(tmp_path / "w.json")
    assert saved.context == {}
    assert saved.facts == ()


# ── Notes and verdicts ─────────────────────────────────────────────


def test_a_note_kind_outside_the_four_is_refused():
    with pytest.raises(AgentConfigurationError, match="kind must be"):
        Note(turn=1, kind="interesting", text="…")


def test_every_note_kind_has_a_label():
    for kind in NOTE_KINDS:
        assert Note(turn=0, kind=kind, text="x").kind in NOTE_KINDS


def test_a_verdict_stamp_outside_the_three_is_refused():
    with pytest.raises(AgentConfigurationError, match="stamp must be"):
        ReadVerdict(claim="x", stamp="maybe", detail="…")


def test_a_verdict_label_overrides_the_stamp_wording():
    verdict = ReadVerdict(
        claim="x", stamp="new", detail="…", label="Hold, with a hole"
    )
    assert verdict.wording == "Hold, with a hole"


def test_a_verdict_falls_back_to_its_stamp_wording():
    assert ReadVerdict(claim="x", stamp="fail", detail="…").wording


# ── Read ───────────────────────────────────────────────────────────


def _read(recording: Recording, notes: tuple[Note, ...] = ()) -> Read:
    return Read(
        title="A read",
        eyebrow="demo v1 · today",
        standfirst="…",
        caveat="…",
        conversations=(
            ReviewedConversation(
                recording=recording, title="Cold", notes=notes
            ),
        ),
    )


def test_read_cost_is_none_when_any_call_was_unpriced(recording):
    unpriced = Recording(
        conversation_id="c",
        agent="demo",
        agent_version="1",
        recorded="",
        turns=(TurnRecord(0, None, "a", metrics(cost=None)),),
    )
    assert _read(unpriced).cents is None


def test_guard_share_is_guard_time_over_wall_clock(recording):
    assert _read(recording).guard_share == pytest.approx(1.6 / 4.0)


def test_turns_where_finds_the_notes_of_one_kind(recording):
    read = _read(
        recording,
        notes=(
            Note(turn=0, kind="defect", text="a"),
            Note(turn=1, kind="worked", text="b"),
        ),
    )
    found = turns_where(read, "defect")
    assert [t.index for _, t in found] == [0]


def test_dry_turns_are_replies_to_a_short_message(recording):
    assert [t.index for t in dry_turns(_read(recording))] == [1]


def test_the_opening_turn_is_never_a_dry_turn(recording):
    assert all(t.prompt is not None for t in dry_turns(_read(recording)))


# ── measure_read ───────────────────────────────────────────────────


def test_measure_read_counts_the_headline_figures(recording):
    counts = measure_read(_read(recording))
    assert counts["Replies"] == "2"
    assert counts["Replies containing a question"] == "50%"
    assert counts["Guard rejections"] == "0"
    assert counts["Cost"] == "0.03c"


def test_measure_read_says_unpriced_rather_than_zero():
    unpriced = Recording(
        conversation_id="c",
        agent="demo",
        agent_version="1",
        recorded="",
        turns=(TurnRecord(0, None, "a", metrics(cost=None)),),
    )
    assert measure_read(_read(unpriced))["Cost"] == "unpriced"


def test_every_headline_measure_is_produced(recording):
    counts = measure_read(_read(recording))
    assert set(HEADLINE_MEASURES) <= set(counts)


# ── History ────────────────────────────────────────────────────────


def test_history_round_trips(tmp_path):
    record = ReadRecord(
        read_id="first",
        date="2026-09-06",
        agent="demo",
        agent_version="1.0.0",
        counts={"Replies": "40"},
    )
    path = append_read(record, tmp_path / "h" / "reads.jsonl")
    assert load_read_history(path) == (record,)


def test_history_is_append_only(tmp_path):
    path = tmp_path / "reads.jsonl"
    append_read(ReadRecord("a", "2026-09-01", "demo", "1", {}), path)
    append_read(ReadRecord("b", "2026-09-06", "demo", "2", {}), path)
    assert [r.read_id for r in load_read_history(path)] == ["a", "b"]


def test_a_missing_history_is_empty_not_an_error(tmp_path):
    assert load_read_history(tmp_path / "nothing.jsonl") == ()


def test_history_filters_by_agent(tmp_path):
    path = tmp_path / "reads.jsonl"
    append_read(ReadRecord("a", "", "demo", "1", {}), path)
    append_read(ReadRecord("b", "", "other", "1", {}), path)
    assert [r.read_id for r in load_read_history(path, "demo")] == ["a"]


def test_a_broken_history_line_names_its_number(tmp_path):
    path = tmp_path / "reads.jsonl"
    path.write_text('{"read_id": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(AgentConfigurationError, match="line 2"):
        load_read_history(path)


# ── The spec ───────────────────────────────────────────────────────


SPEC = """
title: A read
eyebrow: demo v1
standfirst: What this asked.
conversations:
    - recording: cold.json
      title: Cold
      note: She gives nothing.
      notes:
          - turn: 1
            kind: defect
            text: He rambled.
verdicts:
    - claim: He rambles
      stamp: fail
      detail: Still true.
closing:
    - One thing to do next.
pricing: Rates as published.
"""


@pytest.fixture
def spec_dir(tmp_path, recording):
    save_recording(recording, tmp_path / "cold.json")
    (tmp_path / "read.yml").write_text(SPEC, encoding="utf-8")
    return tmp_path


def test_a_spec_loads_its_recordings(spec_dir):
    read = load_read(spec_dir / "read.yml")
    assert read.title == "A read"
    assert read.conversations[0].recording.conversation_id == "cold"
    assert read.conversations[0].title == "Cold"


def test_a_spec_anchors_notes_to_their_turn(spec_dir):
    read = load_read(spec_dir / "read.yml")
    conv = read.conversations[0]
    assert conv.notes_for(1)[0].kind == "defect"
    assert conv.notes_for(0) == ()


def test_a_spec_without_conversations_is_refused(tmp_path):
    path = tmp_path / "read.yml"
    path.write_text("title: x\n", encoding="utf-8")
    with pytest.raises(AgentConfigurationError, match="at least one"):
        load_read(path)


def test_a_spec_that_is_not_a_mapping_is_refused(tmp_path):
    path = tmp_path / "read.yml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(AgentConfigurationError, match="a mapping"):
        load_read(path)


def test_a_spec_supplies_the_standing_caveat_when_silent(spec_dir):
    read = load_read(spec_dir / "read.yml")
    assert "invented" in read.caveat


# ── Rendering ──────────────────────────────────────────────────────


def test_the_page_carries_the_title_and_the_transcript(spec_dir):
    html = render_review(spec_dir / "read.yml")
    assert "<title>A read</title>" in html
    assert "buenas" in html
    assert "He rambled." in html


def test_the_page_carries_the_verdict_and_its_stamp(spec_dir):
    html = render_review(spec_dir / "read.yml")
    assert "He rambles" in html
    assert 'class="stamp fail"' in html


def test_the_page_offers_both_display_modes(spec_dir):
    html = render_review(spec_dir / "read.yml")
    assert 'id="btn-read"' in html
    assert 'id="btn-review"' in html


def test_the_page_dates_every_conversation(spec_dir):
    html = render_review(spec_dir / "read.yml")
    assert "6 September 2026, 10:00 UTC" in html


def test_the_page_shows_the_context_it_ran_with(tmp_path, recording):
    full = replace(
        recording,
        context={"nick": "Marta_87", "mi_nick": "cronista_del_barrio"},
        facts=("Le gusta ser el último en irse.",),
    )
    save_recording(full, tmp_path / "cold.json")
    (tmp_path / "read.yml").write_text(
        "title: A read\nconversations:\n  - recording: cold.json\n",
        encoding="utf-8",
    )
    html = render_review(tmp_path / "read.yml")
    assert "Marta_87" in html
    assert "cronista_del_barrio" in html
    assert "Le gusta ser el último en irse." in html


def test_a_conversation_with_no_context_says_so(spec_dir):
    """Blank would read as "it ran with none", which is different."""
    html = render_review(spec_dir / "read.yml")
    assert "not recorded" in html


# ── read_payload ───────────────────────────────────────────────────


def test_the_payload_carries_the_whole_read(spec_dir):
    payload = read_payload(spec_dir / "read.yml")
    assert payload["title"] == "A read"
    assert payload["verdicts"][0]["stamp"] == "fail"
    assert payload["conversations"][0]["turns"]


def test_the_payload_carries_the_labels_a_renderer_needs(spec_dir):
    """A renderer elsewhere must not hardcode its own."""
    payload = read_payload(spec_dir / "read.yml")
    assert payload["note_kinds"] == NOTE_KINDS
    assert payload["stamps"] == STAMPS


def test_the_payload_groups_the_context_rows(tmp_path, recording):
    """A renderer lays the groups out apart without reading labels."""
    full = replace(
        recording,
        context={"nick": "Marta_87"},
        facts=("Es periodista.", "Es deportista."),
        variants={"bien_reply": "un bien de manual"},
    )
    save_recording(full, tmp_path / "cold.json")
    (tmp_path / "read.yml").write_text(
        "title: A read\nconversations:\n  - recording: cold.json\n",
        encoding="utf-8",
    )
    rows = read_payload(tmp_path / "read.yml")["conversations"][0]["context"]
    groups = {row["label"]: row["group"] for row in rows}
    assert groups["When"] == "run"
    assert groups["Agent"] == "run"
    assert groups["nick"] == "context"
    assert groups["Knew already"] == "memory"
    assert groups["variant: bien_reply"] == "variants"
    assert all("group" in row for row in rows)


def test_the_payload_lists_the_facts_one_by_one(tmp_path, recording):
    """Joined into one string, a renderer could only wrap them."""
    full = replace(
        recording,
        facts=("Es periodista.", "Es deportista."),
    )
    save_recording(full, tmp_path / "cold.json")
    (tmp_path / "read.yml").write_text(
        "title: A read\nconversations:\n  - recording: cold.json\n",
        encoding="utf-8",
    )
    rows = read_payload(tmp_path / "read.yml")["conversations"][0]["context"]
    memory = next(row for row in rows if row["group"] == "memory")
    assert memory["items"] == ["Es periodista.", "Es deportista."]
    assert memory["value"] == "Es periodista. · Es deportista."


def test_the_payload_is_json_serialisable(spec_dir):
    """It is served over HTTP, so a stray tuple would break it."""
    json.dumps(read_payload(spec_dir / "read.yml"))


def test_the_payload_and_the_page_show_the_same_conversations(
    spec_dir,
):
    """The whole point: one data contract, so they cannot drift."""
    payload = read_payload(spec_dir / "read.yml")
    html = render_review(spec_dir / "read.yml")
    for conv in payload["conversations"]:
        assert conv["title"] in html


def test_the_page_carries_all_four_note_colours(spec_dir):
    html = render_review(spec_dir / "read.yml")
    for label in NOTE_KINDS.values():
        assert label in html


def test_transcript_text_is_escaped_but_a_note_is_not(tmp_path):
    save_recording(
        Recording(
            conversation_id="c",
            agent="demo",
            agent_version="1",
            recorded="",
            turns=(TurnRecord(0, "<b>hola</b>", "a & b", metrics()),),
        ),
        tmp_path / "cold.json",
    )
    (tmp_path / "read.yml").write_text(
        "title: t\nconversations:\n"
        "    - recording: cold.json\n"
        "      title: C\n"
        "      notes:\n"
        "          - turn: 0\n"
        "            kind: worked\n"
        "            text: <strong>kept</strong>\n",
        encoding="utf-8",
    )
    html = render_review(tmp_path / "read.yml")
    payload = html.split("const PAGE = ")[1].split(";\n")[0]
    data = json.loads(payload)
    assert data["conversations"][0]["turns"][0]["prompt"] == ("<b>hola</b>")
    assert "<strong>kept</strong>" in html


def test_write_review_records_the_read_when_asked(spec_dir):
    out = spec_dir / "page.html"
    history = spec_dir / "reads.jsonl"
    write_review(spec_dir / "read.yml", out, history=history, read_id="first")
    assert out.is_file()
    stored = load_read_history(history)
    assert stored[0].read_id == "first"
    assert stored[0].agent == "demo"
    assert stored[0].counts["Replies"] == "2"


def test_write_review_leaves_no_history_by_default(spec_dir):
    write_review(spec_dir / "read.yml", spec_dir / "page.html")
    assert not (spec_dir / "reads.jsonl").exists()


def test_the_comparison_column_fills_from_the_history(spec_dir):
    append_read(
        ReadRecord(
            read_id="earlier",
            date="2026-09-01",
            agent="demo",
            agent_version="0.9",
            counts={"Replies": "40"},
        ),
        spec_dir / "reads.jsonl",
    )
    (spec_dir / "read.yml").write_text(
        SPEC + "history: reads.jsonl\n", encoding="utf-8"
    )
    read = load_read(spec_dir / "read.yml")
    row = next(r for r in read.comparison if r["measure"] == "Replies")
    assert row["previous"] == "40"
    assert row["current"] == "2"


# ── The theme ──────────────────────────────────────────────────────
#
# A colour defined in one theme's block and forgotten in another
# renders one theme's text on the other theme's ground, and nothing
# fails when it happens: the page just looks wrong for whoever has
# that setting. It went wrong once, on the agent's own chat bubble,
# so it is checked rather than watched for.

#: Tokens that are deliberately the same in every theme.
THEME_FREE = {"--radius"}


def _block(css: str, selector: str) -> dict[str, str]:
    """The custom properties one selector's block declares."""
    start = css.index(selector + " {") + len(selector) + 2
    depth = 1
    end = start
    while depth:
        if css[end] == "{":
            depth += 1
        elif css[end] == "}":
            depth -= 1
        end += 1
    body = css[start : end - 1]
    return dict(
        line.split(":", 1)[0].strip()
        and (
            line.split(":", 1)[0].strip(),
            line.split(":", 1)[1].strip().rstrip(";"),
        )
        for line in body.splitlines()
        if line.strip().startswith("--")
    )


@pytest.fixture
def themes() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    from agents7.eval._review_theme import REVIEW_CSS

    return (
        _block(REVIEW_CSS, ":root"),
        _block(REVIEW_CSS, ':root:not([data-theme="light"])'),
        _block(REVIEW_CSS, ':root[data-theme="dark"]'),
    )


def test_every_theme_variable_ships_a_dark_value(themes):
    light, system_dark, chosen_dark = themes
    expected = set(light) - THEME_FREE
    assert expected - set(system_dark) == set()
    assert expected - set(chosen_dark) == set()


def test_the_two_dark_blocks_agree(themes):
    """A viewer on the system default and one who chose dark must
    not get different pages. Both blocks are written from `_DARK`,
    so this checks the generation rather than a hand-kept copy."""
    _, system_dark, chosen_dark = themes
    assert system_dark == chosen_dark


def test_the_dark_blocks_are_the_declared_palette(themes):
    from agents7.eval._review_theme import _DARK

    _, system_dark, _ = themes
    assert system_dark == _DARK


def test_a_dark_block_defines_nothing_light_does_not(themes):
    light, system_dark, _ = themes
    assert set(system_dark) - set(light) == set()


def test_every_note_kind_has_a_colour_in_every_theme(themes):
    for block in themes:
        for kind in NOTE_KINDS:
            assert f"--{kind}" in block
            assert f"--{kind}-wash" in block


def test_the_page_never_hardcodes_a_colour_outside_the_tokens():
    """Structure is ink and slate, and both come from tokens. A
    literal colour in a rule is one that cannot follow the theme."""
    from agents7.eval._review_theme import REVIEW_CSS

    body = REVIEW_CSS.split("--radius: 14px;", 1)[1]
    literals = [
        line
        for line in body.splitlines()
        if "#" in line and not line.strip().startswith("--")
    ]
    assert literals == []


def test_a_read_is_never_its_own_previous_read(spec_dir):
    """Recording a read then re-rendering it must not compare it
    against itself, which is what happens every time a note is
    reworded."""
    (spec_dir / "read.yml").write_text(
        SPEC + "history: reads.jsonl\nread_id: fourth\n",
        encoding="utf-8",
    )
    write_review(
        spec_dir / "read.yml",
        spec_dir / "page.html",
        history=spec_dir / "reads.jsonl",
    )
    read = load_read(spec_dir / "read.yml")
    assert all(r["previous"] == "—" for r in read.comparison)


def test_recording_the_same_read_twice_files_it_once(spec_dir):
    (spec_dir / "read.yml").write_text(
        SPEC + "history: reads.jsonl\nread_id: fourth\n",
        encoding="utf-8",
    )
    for _ in range(3):
        write_review(
            spec_dir / "read.yml",
            spec_dir / "page.html",
            history=spec_dir / "reads.jsonl",
        )
    assert len(load_read_history(spec_dir / "reads.jsonl")) == 1


def test_a_later_read_does_compare_against_the_earlier_one(spec_dir):
    append_read(
        ReadRecord(
            read_id="third",
            date="2026-09-01",
            agent="demo",
            agent_version="0.9",
            counts={"Replies": "40"},
        ),
        spec_dir / "reads.jsonl",
    )
    (spec_dir / "read.yml").write_text(
        SPEC + "history: reads.jsonl\nread_id: fourth\n",
        encoding="utf-8",
    )
    read = load_read(spec_dir / "read.yml")
    row = next(r for r in read.comparison if r["measure"] == "Replies")
    assert row["previous"] == "40"


def test_the_read_id_defaults_to_the_folder_name(spec_dir):
    assert load_read(spec_dir / "read.yml").read_id == spec_dir.name


def test_a_first_read_shows_no_previous_figures(spec_dir):
    (spec_dir / "read.yml").write_text(
        SPEC + "history: reads.jsonl\n", encoding="utf-8"
    )
    read = load_read(spec_dir / "read.yml")
    assert all(r["previous"] == "—" for r in read.comparison)
