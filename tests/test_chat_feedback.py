"""Unit tests for standardised feedback on a chat agent."""

import json

import pytest

from agents7.eval.chat_feedback import (
    VERDICTS,
    Feedback,
    FeedbackEntry,
    load_feedback,
    merge_entries,
    render_feedback,
    save_feedback,
    write_eval_rows,
    write_feedback_report,
)
from agents7.exceptions import AgentConfigurationError


# ── Fixtures ───────────────────────────────────────────────────────


def entry(**over) -> FeedbackEntry:
    """A correction with a rewrite and the rule it became."""
    base = {
        "conversation": "warm",
        "turn": 3,
        "verdict": "wrong",
        "he_said": "mucho mejor que corregir textos",
        "should_say": "esto es más entretenido que corregir textos",
        "why": "overstates the investment three messages in",
        "rule": "no overstating his interest early on",
        "lands_in": "characters/david.yml",
    }
    base.update(over)
    return FeedbackEntry(**base)


@pytest.fixture
def feedback() -> Feedback:
    return Feedback(
        read_id="2026-09-20",
        agent="david-chat",
        agent_version="0.4.0",
        reviewed="2026-09-20",
        golden=("warm",),
        entries=(
            entry(),
            entry(
                turn=7,
                verdict="right",
                he_said="jajaja eso debería ser un delito",
                should_say="",
                why="the recipes exchange is the register to copy",
                rule="",
                lands_in="",
            ),
            entry(
                conversation="cold",
                turn=None,
                verdict="wrong",
                he_said="",
                should_say="",
                why="he writes more the less the other person gives him",
                rule="one message, fairly short, to a cold turn",
                lands_in="goals/cafe_chat.yml",
            ),
        ),
    )


# ── FeedbackEntry ──────────────────────────────────────────────────


def test_a_verdict_outside_the_three_is_refused():
    with pytest.raises(AgentConfigurationError, match="verdict must"):
        entry(verdict="terrible")


def test_an_entry_saying_nothing_is_refused():
    """A verdict alone is not feedback."""
    with pytest.raises(AgentConfigurationError, match="says nothing"):
        FeedbackEntry(conversation="warm", turn=1, verdict="wrong")


def test_a_reason_alone_is_enough():
    """Knowing a line is wrong without writing its replacement."""
    only_why = FeedbackEntry(
        conversation="warm",
        turn=1,
        verdict="weak",
        why="flat, but I am not sure what would be better",
    )
    assert only_why.is_correction


def test_an_entry_without_a_turn_is_structural():
    assert entry(turn=None).is_structural
    assert not entry(turn=3).is_structural


def test_right_is_not_a_correction():
    assert not entry(verdict="right").is_correction
    assert entry(verdict="weak").is_correction
    assert entry(verdict="wrong").is_correction


def test_a_rule_is_outstanding_until_it_ships():
    assert entry().outstanding
    assert not entry(shipped="v0.1.5").outstanding


def test_an_entry_with_no_rule_is_never_outstanding():
    """Nothing to ship, so nothing to chase."""
    assert not entry(rule="", why="just flat").outstanding


def test_empty_fields_are_left_out_of_the_file(feedback):
    raw = feedback.entries[1].to_dict()
    assert "rule" not in raw
    assert "should_say" not in raw
    assert raw["verdict"] == "right"


def test_a_structural_entry_keeps_no_turn_key():
    assert "turn" not in entry(turn=None).to_dict()


def test_an_entry_missing_its_anchor_explains_itself():
    with pytest.raises(AgentConfigurationError, match="conversation"):
        FeedbackEntry.from_dict({"verdict": "wrong"})


# ── Rule changes ───────────────────────────────────────────────────


def test_rules_group_by_the_file_they_land_in(feedback):
    changes = feedback.rule_changes()
    assert [c.lands_in for c in changes] == [
        "characters/david.yml",
        "goals/cafe_chat.yml",
    ]
    assert changes[0].rules == ("no overstating his interest early on",)


def test_an_entry_with_no_rule_produces_no_change(feedback):
    """The `right` entry states no rule, so it changes nothing."""
    assert sum(len(c.entries) for c in feedback.rule_changes()) == 2


def test_a_rule_with_no_destination_is_surfaced_last():
    stray = Feedback(
        read_id="r",
        agent="a",
        entries=(
            entry(lands_in=""),
            entry(turn=4, lands_in="characters/david.yml"),
        ),
    )
    assert [c.lands_in for c in stray.rule_changes()] == [
        "characters/david.yml",
        "unassigned",
    ]


def test_outstanding_rules_are_separable(feedback):
    change = feedback.rule_changes()[0]
    assert len(change.outstanding) == 1
    shipped = Feedback(
        read_id="r", agent="a", entries=(entry(shipped="v0.1.5"),)
    )
    assert shipped.rule_changes()[0].outstanding == ()


# ── Lookups ────────────────────────────────────────────────────────


def test_entries_are_found_by_turn(feedback):
    found = feedback.for_turn("warm", 3)
    assert len(found) == 1
    assert found[0].verdict == "wrong"


def test_a_turn_number_alone_does_not_match_across_conversations():
    """Two conversations both have a turn 3."""
    both = Feedback(
        read_id="r",
        agent="a",
        entries=(
            entry(conversation="warm"),
            entry(conversation="cold"),
        ),
    )
    assert len(both.for_turn("warm", 3)) == 1


def test_structural_entries_can_be_narrowed(feedback):
    assert len(feedback.structural()) == 1
    assert len(feedback.structural("cold")) == 1
    assert feedback.structural("warm") == ()


# ── Eval rows ──────────────────────────────────────────────────────


def test_a_correction_becomes_a_row_targeting_the_rewrite(feedback):
    rows = feedback.eval_rows()
    wrong = [r for r in rows if r["verdict"] == "wrong"][0]
    assert wrong["original"] == "mucho mejor que corregir textos"
    assert wrong["target"] == "esto es más entretenido que corregir textos"


def test_a_line_marked_right_targets_itself(feedback):
    """Good enough to imitate is already a target."""
    rows = feedback.eval_rows()
    good = [r for r in rows if r["verdict"] == "right"][0]
    assert good["target"] == good["original"]


def test_a_structural_note_yields_no_row(feedback):
    """Nothing to score a whole-conversation note against."""
    assert all(r["turn"] is not None for r in feedback.eval_rows())
    assert len(feedback.eval_rows()) == 2


def test_a_correction_with_no_rewrite_yields_no_row():
    vague = Feedback(
        read_id="r",
        agent="a",
        entries=(entry(should_say="", rule="", why="flat"),),
    )
    assert vague.eval_rows() == ()


def test_rows_say_whether_the_conversation_was_golden(feedback):
    rows = {r["conversation"]: r for r in feedback.eval_rows()}
    assert rows["warm"]["golden"] is True


def test_rows_write_as_jsonl(tmp_path, feedback):
    path = write_eval_rows(feedback, tmp_path / "a" / "rows.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["agent"] == "david-chat"


# ── Round trip ─────────────────────────────────────────────────────


def test_feedback_survives_a_round_trip(tmp_path, feedback):
    path = save_feedback(feedback, tmp_path / "a" / "feedback.yml")
    assert load_feedback(path) == feedback


def test_a_file_missing_its_header_explains_itself(tmp_path):
    path = tmp_path / "feedback.yml"
    path.write_text("entries: []\n", encoding="utf-8")
    with pytest.raises(AgentConfigurationError, match="read_id"):
        load_feedback(path)


def test_a_file_that_is_not_a_mapping_is_refused(tmp_path):
    path = tmp_path / "feedback.yml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(AgentConfigurationError, match="a mapping"):
        load_feedback(path)


# ── Merging ────────────────────────────────────────────────────────


def test_merging_replaces_every_entry_on_a_touched_turn(feedback):
    fresh = entry(turn=3, rule="a different rule")
    merged = merge_entries(feedback, [fresh])
    assert len(merged.for_turn("warm", 3)) == 1
    assert merged.for_turn("warm", 3)[0].rule == "a different rule"


def test_merging_keeps_turns_nobody_touched(feedback):
    merged = merge_entries(feedback, [entry(turn=3)])
    assert len(merged.for_turn("warm", 7)) == 1
    assert len(merged.structural("cold")) == 1


def test_merging_can_add_several_entries_to_one_turn(feedback):
    merged = merge_entries(
        feedback,
        [entry(turn=3), entry(turn=3, rule="a second rule")],
    )
    assert len(merged.for_turn("warm", 3)) == 2


def test_merging_leaves_the_original_alone(feedback):
    merge_entries(feedback, [entry(turn=3, rule="x")])
    assert feedback.for_turn("warm", 3)[0].rule.startswith("no overstating")


# ── The report ─────────────────────────────────────────────────────


def test_the_report_groups_rules_by_file(feedback):
    report = render_feedback(feedback)
    assert "`characters/david.yml`" in report
    assert "`goals/cafe_chat.yml`" in report


def test_the_report_marks_what_has_not_shipped(feedback):
    assert "not shipped" in render_feedback(feedback)


def test_the_report_carries_the_rewrite_and_the_reason(feedback):
    report = render_feedback(feedback)
    assert "esto es más entretenido que corregir textos" in report
    assert "overstates the investment" in report


def test_the_report_names_the_golden_conversations(feedback):
    assert "`warm`" in render_feedback(feedback)


def test_the_report_says_when_nothing_was_accepted():
    plain = Feedback(read_id="r", agent="a", entries=(entry(),))
    assert "Store the corrections" in render_feedback(plain)


def test_the_report_lists_what_is_worth_imitating(feedback):
    report = render_feedback(feedback)
    assert "Worth imitating" in report
    assert "delito" in report


def test_a_report_with_no_feedback_still_renders():
    empty = Feedback(read_id="r", agent="a")
    report = render_feedback(empty)
    assert "No rule changes." in report
    assert "No line rewrites." in report


def test_the_report_writes_to_a_file(tmp_path, feedback):
    path = write_feedback_report(feedback, tmp_path / "a" / "r.md")
    assert path.read_text(encoding="utf-8").startswith("# Feedback")


def test_every_verdict_has_a_label():
    assert set(VERDICTS) == {"right", "weak", "wrong"}
