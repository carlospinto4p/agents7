"""Unit tests for persisting and comparing style-contract runs."""

import hashlib
import json
import math
from dataclasses import replace
from math import log
from statistics import NormalDist, stdev

import pytest

from agents7.eval.style import (
    ConversationScript,
    LostRun,
    RunScore,
    StyleContract,
    StyleRule,
    ends_with_question,
    score_transcripts,
)
from agents7.eval.style_history import (
    MIN_BASELINE_RUNS,
    MIN_SPREAD_PRIORS,
    MarginSource,
    RuleRecord,
    ShiftEstimate,
    StyleRunRecord,
    Verdict,
    _chi2_quantile,
    _expected_max,
    _spread_upper_bound,
    append_style_run,
    compare_to_history,
    configuration_key,
    detectable_shift,
    fingerprint_run,
    load_style_history,
    record_style_run,
    relabel,
    runs_for_shift,
    sampling_margin,
)
from agents7.exceptions import AgentConfigurationError

from agents7.call_log import CallRecord
from agents7.llm import LLMEngine

from helpers import make_chatter, make_guard, make_guarded_chatter


# --- Helpers ---


def contract(*, budget=None, lines=("a",)):
    return StyleContract(
        rules=(
            StyleRule(
                id="ends_q",
                description="d",
                check=ends_with_question(),
                budget=budget,
            ),
        ),
        scripts=(ConversationScript(id="s", lines=lines),),
    )


def record_at(
    rate_pairs,
    *,
    fingerprint="fp",
    when="2026-08-30T10:00:00+00:00",
    answered_by=(),
    config_key=None,
    definitions=None,
    guard_answered_by=(),
):
    """Build a record directly, for history that was never run.

    Each entry is ``(rule id, violations, applicable)``, optionally
    with a fourth item holding that rule's per-run rates.
    *definitions* maps a rule id to its `RuleRecord.definition`.
    """
    definitions = definitions or {}
    return StyleRunRecord(
        recorded_at=when,
        subject="test",
        fingerprint=fingerprint,
        agent="a",
        agent_version=1,
        model_id="ollama::x",
        repeats=1,
        replies=10,
        calls=20,
        cost_cents=None,
        rules=tuple(
            RuleRecord(
                id=entry[0],
                budget=None,
                violations=entry[1],
                applicable=entry[2],
                run_rates=tuple(entry[3]) if len(entry) > 3 else (),
                definition=definitions.get(entry[0]),
            )
            for entry in rate_pairs
        ),
        answered_by=tuple(answered_by),
        config_key=config_key,
        guard_answered_by=tuple(guard_answered_by),
    )


def answered_report(report, agent, models):
    """Add call records for *agent*, each answered by one model."""
    return replace(
        report,
        calls=report.calls
        + tuple(
            CallRecord(
                trace_id=f"a{i}",
                agent_name=agent.name,
                agent_version=agent.version,
                model_id=agent.engine.model_id,
                success=True,
                model=model,
            )
            for i, model in enumerate(models)
        ),
    )


def guarded_report(report, guard, calls=1, model=None):
    """Add the guard's own call records to a scored report.

    `score_transcripts` scores text and makes no calls, so a test
    that needs a run the guard took part in has to say so the way
    a live run does — with the guard's `CallRecord`s. *model* is
    what each of those calls reports as having answered.
    """
    return replace(
        report,
        calls=tuple(
            CallRecord(
                trace_id=f"t{i}",
                agent_name=guard.name,
                agent_version=guard.version,
                model_id=guard.engine.model_id,
                success=True,
                model=model,
            )
            for i in range(calls)
        ),
    )


# ── Recording ──────────────────────────────────────────────


def test_record_carries_the_pooled_and_per_run_figures():
    """A record keeps the spread, not only the pooled rate."""
    report = score_transcripts(
        contract(),
        [[("a", "¿y tú?")], [("a", "pues eso")]],
    )
    made = record_style_run(report, contract(), make_chatter(), "v1")

    rule = made.rule("ends_q")
    assert (rule.violations, rule.applicable) == (1, 2)
    assert rule.rate == 0.5
    assert rule.run_rates == (1.0, 0.0)


def test_record_keeps_the_budget_of_its_own_time():
    """A budget tightened later must not fail an old record.

    Without this, re-reading history after raising the bar makes
    every earlier invocation look like a failure it never was.
    """
    report = score_transcripts(contract(budget=0.5), [[("a", "¿y tú?")]])
    made = record_style_run(
        report, contract(budget=0.5), make_chatter(), "v1"
    )
    assert made.rule("ends_q").budget == 0.5


def test_record_names_the_model_it_ran_on():
    """A model change explains a moved number as readily as a
    config edit, so it is recorded rather than assumed."""
    agent = make_chatter()
    report = score_transcripts(contract(), [[("a", "¿y tú?")]])
    made = record_style_run(report, contract(), agent, "v1")
    assert made.model_id == agent.engine.model_id


def test_record_names_the_model_that_answered_not_the_configured_one():
    """DeepSeek moved `deepseek-chat` to a new model on 10th
    September 2026 and every record kept saying `deepseek-chat`."""
    agent = make_chatter()
    report = answered_report(
        score_transcripts(contract(), [[("a", "¿y tú?")]]),
        agent,
        ["openai_compat::deepseek-flash"] * 2,
    )
    made = record_style_run(report, contract(), agent, "v1")
    assert made.answered_by == ("openai_compat::deepseek-flash",)


def test_answered_by_ignores_the_guard_and_calls_with_no_model():
    """The guard is apparatus, and a call that reported no model
    says nothing about which one answered."""
    guard = make_guard(
        engine=LLMEngine(provider="ollama", model_name="gemma3:12b")
    )
    agent = make_guarded_chatter(guard=guard)
    scored = score_transcripts(contract(), [[("a", "¿y tú?")]])
    report = answered_report(
        guarded_report(scored, guard), agent, ["ollama::b", None, "ollama::a"]
    )
    made = record_style_run(report, contract(), agent, "v1")
    assert made.answered_by == ("ollama::a", "ollama::b")


def test_a_run_with_no_reported_model_records_none():
    report = score_transcripts(contract(), [[("a", "¿y tú?")]])
    made = record_style_run(report, contract(), make_chatter(), "v1")
    assert made.answered_by == ()


def test_answered_by_survives_being_written_and_read(tmp_path):
    path = tmp_path / "history.jsonl"
    append_style_run(record_at([("r", 1, 2)], answered_by=["x::y"]), path)
    (back,) = load_style_history(path)
    assert back.answered_by == ("x::y",)


def test_an_unpriced_run_records_no_cost():
    """None, never 0.0 — an unpriced run is not a free one."""
    report = score_transcripts(contract(), [[("a", "¿y tú?")]])
    made = record_style_run(report, contract(), make_chatter(), "v1")
    assert made.cost_cents is None


# ── Fingerprints ───────────────────────────────────────────


def test_the_same_agent_and_contract_fingerprint_alike():
    agent = make_chatter()
    assert fingerprint_run(contract(), agent) == fingerprint_run(
        contract(), agent
    )


def test_a_changed_system_prompt_changes_the_fingerprint():
    """Editing the persona must start a fresh baseline.

    A band drawn from a different agent describes a different
    agent, and comparing against it would invent a regression.
    """
    before = fingerprint_run(contract(), make_chatter())
    after = fingerprint_run(
        contract(), make_chatter(system_prompt="You are someone else.")
    )
    assert before != after


def test_a_changed_script_changes_the_fingerprint():
    """A different stimulus is a different measurement."""
    agent = make_chatter()
    assert fingerprint_run(contract(lines=("a",)), agent) != fingerprint_run(
        contract(lines=("b",)), agent
    )


def test_a_script_context_changes_the_fingerprint():
    """What a script hands the agent is part of the stimulus."""
    agent = make_chatter()
    plain = contract()
    script = ConversationScript(id="s", lines=("a",), context={"nick": "x"})
    with_context = StyleContract(rules=plain.rules, scripts=(script,))
    assert fingerprint_run(plain, agent) != fingerprint_run(
        with_context, agent
    )
    assert configuration_key(plain, agent) != configuration_key(
        with_context, agent
    )


def test_a_script_without_context_keeps_its_stored_fingerprint():
    """The hash of a context-free script is what it always was."""
    agent = make_chatter()
    parts = [agent.system_prompt_identity, "ends_q", "s|False|a"]
    expected = hashlib.sha256("\n".join(parts).encode("utf-8"))
    assert fingerprint_run(contract(), agent) == expected.hexdigest()[:12]


def test_a_changed_budget_leaves_the_fingerprint_alone():
    """A threshold is not a measurement, so it is not an identity.

    A budget is read once, in `StyleRuleResult.passed`, against a
    rate that has already been measured. Keying on it meant that
    writing down the threshold a baseline had just justified threw
    that baseline away — the one edit that provably cannot move a
    number also reset every band. Changed in v8.0.0.
    """
    agent = make_chatter()
    assert fingerprint_run(contract(budget=None), agent) == fingerprint_run(
        contract(budget=0.5), agent
    )


def test_a_changed_rule_set_still_changes_the_fingerprint():
    """Dropping the budget must not make rule edits invisible.

    Adding or removing a rule changes what is measured, so it has
    to keep starting a fresh band. This is also what kept
    one agent's two bands apart when the budgets left the hash:
    they differ by a rule that was added, not only by a threshold,
    so correcting the key could not merge them.
    """
    agent = make_chatter()
    two = StyleContract(
        rules=(
            *contract().rules,
            StyleRule(
                id="other",
                description="d",
                check=ends_with_question(),
                budget=None,
            ),
        ),
        scripts=contract().scripts,
    )
    assert fingerprint_run(two, agent) != fingerprint_run(contract(), agent)


def test_a_plain_agent_hashes_to_its_pinned_value():
    """A fingerprint must never change by accident.

    Folding the guard into the hash would have changed every
    fingerprint, including the eleven unguarded records of one agent
    that were already a baseline, so this pins the unguarded hash
    and fails when an edit would reset a band.

    The pin moved once, deliberately, in v8.0.0, when the budgets
    left the hash. Every stored key changed with it; one agent's
    sixteen reproducible records were re-filed under their new
    keys in the same release, and its eight older ones were not,
    because that configuration could no longer be rebuilt to prove
    what they were.
    """
    assert fingerprint_run(contract(), make_chatter()) == "e48bf6291466"


def test_a_guarded_run_and_an_unguarded_one_differ():
    """The bug this fixes.

    Five guarded and five unguarded invocations of one agent all
    carried the fingerprint `cfcfa8d5d117` on 30th August 2026,
    and were one file away from sharing a band — while dropping
    the guard moved one rule from 50% to 56%.
    """
    agent = make_guarded_chatter()
    report = score_transcripts(contract(), [[("a", "¿y tú?")]])

    with_guard = fingerprint_run(
        contract(), agent, guarded_report(report, agent.guard)
    )
    without = fingerprint_run(contract(), agent, report)
    assert with_guard != without


def test_a_configured_guard_that_never_ran_does_not_count():
    """Participation is read from the run, not from the agent.

    An unguarded run is made by replacing the guard's check, so
    the agent still carries a fully configured guard. Such a run
    must hash as the plain agent does, or the unguarded history
    it belongs to would split in two.
    """
    report = score_transcripts(contract(), [[("a", "¿y tú?")]])
    bypassed = fingerprint_run(contract(), make_guarded_chatter(), report)
    assert bypassed == fingerprint_run(contract(), make_chatter())


def test_a_changed_guard_prompt_changes_the_fingerprint():
    """The guard rewrites replies before they are scored."""
    report = score_transcripts(contract(), [[("a", "¿y tú?")]])
    prints = []
    for prompt in ("(auto-generated)", "Reject everything."):
        agent = make_guarded_chatter(guard=make_guard(system_prompt=prompt))
        prints.append(
            fingerprint_run(
                contract(), agent, guarded_report(report, agent.guard)
            )
        )
    assert prints[0] != prints[1]


def test_a_changed_guard_model_changes_the_fingerprint():
    """One agent has run its guard on a local model while the
    chatter was on DeepSeek; swapping it back moved nothing."""
    report = score_transcripts(contract(), [[("a", "¿y tú?")]])
    prints = []
    for model in ("qwen3", "gemma3:12b"):
        guard = make_guard(
            engine=LLMEngine(provider="ollama", model_name=model)
        )
        agent = make_guarded_chatter(guard=guard)
        prints.append(
            fingerprint_run(contract(), agent, guarded_report(report, guard))
        )
    assert prints[0] != prints[1]


# ── The guard on the record ────────────────────────────────


def test_a_record_names_the_guard_model_that_ran():
    """The guard is 59% of an invocation's bill on one agent, and
    `model_id` alone cannot say which model produced that half."""
    guard = make_guard(
        engine=LLMEngine(provider="ollama", model_name="gemma3:12b")
    )
    agent = make_guarded_chatter(guard=guard)
    report = guarded_report(
        score_transcripts(contract(), [[("a", "¿y tú?")]]), guard
    )
    made = record_style_run(report, contract(), agent, "v1")

    assert made.guard_model_id == "ollama::gemma3:12b"
    assert made.model_id == agent.engine.model_id


def test_a_bypassed_guard_records_no_model():
    """None, never the configured model — a guard that did not
    run did not produce the numbers."""
    agent = make_guarded_chatter()
    report = score_transcripts(contract(), [[("a", "¿y tú?")]])
    made = record_style_run(report, contract(), agent, "v1")
    assert made.guard_model_id is None


def test_the_guard_model_survives_being_written_and_read(tmp_path):
    guard = make_guard(
        engine=LLMEngine(provider="ollama", model_name="gemma3:12b")
    )
    agent = make_guarded_chatter(guard=guard)
    report = guarded_report(
        score_transcripts(contract(), [[("a", "¿y tú?")]]), guard
    )
    path = tmp_path / "history.jsonl"
    append_style_run(record_style_run(report, contract(), agent, "v1"), path)

    (back,) = load_style_history(path)
    assert back.guard_model_id == "ollama::gemma3:12b"


def test_a_record_written_before_the_field_reads_none(tmp_path):
    """An old line must load, not raise, and must not claim a
    guard model it never recorded."""
    path = tmp_path / "history.jsonl"
    raw = record_at([("ends_q", 1, 2)]).to_dict()
    del raw["guard_model_id"]
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    (back,) = load_style_history(path)
    assert back.guard_model_id is None


def test_a_record_names_the_model_that_answered_the_guard():
    """One agent's guard asks for `deepseek-chat` too, so the
    configured name cannot say which model checked the replies."""
    guard = make_guard(
        engine=LLMEngine(provider="ollama", model_name="gemma3:12b")
    )
    agent = make_guarded_chatter(guard=guard)
    report = guarded_report(
        score_transcripts(contract(), [[("a", "¿y tú?")]]),
        guard,
        calls=2,
        model="ollama::gemma3:12b-it-qat",
    )
    made = record_style_run(report, contract(), agent, "v1")

    assert made.guard_answered_by == ("ollama::gemma3:12b-it-qat",)
    assert made.guard_model_id == "ollama::gemma3:12b"
    assert made.answered_by == ()


def test_a_bypassed_guard_records_no_answering_model():
    agent = make_guarded_chatter()
    report = score_transcripts(contract(), [[("a", "¿y tú?")]])
    made = record_style_run(report, contract(), agent, "v1")
    assert made.guard_answered_by == ()


def test_the_guard_answering_model_survives_a_round_trip(tmp_path):
    path = tmp_path / "history.jsonl"
    append_style_run(
        record_at([("r", 1, 2)], guard_answered_by=["x::g"]), path
    )
    (back,) = load_style_history(path)
    assert back.guard_answered_by == ("x::g",)


def test_a_record_written_before_the_guard_answer_reads_empty(tmp_path):
    path = tmp_path / "history.jsonl"
    raw = record_at([("ends_q", 1, 2)]).to_dict()
    del raw["guard_answered_by"]
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    (back,) = load_style_history(path)
    assert back.guard_answered_by == ()


# ── Round-tripping ─────────────────────────────────────────


def test_a_record_survives_being_written_and_read(tmp_path):
    path = tmp_path / "history.jsonl"
    made = record_at([("ends_q", 1, 2)])

    append_style_run(made, path)
    loaded = load_style_history(path)

    assert len(loaded) == 1
    assert loaded[0] == made


def test_history_appends_rather_than_replaces(tmp_path):
    """Append-only on purpose.

    The history is the only thing that can say how much a rule
    moves when nothing changed; rewriting it destroys that.
    """
    path = tmp_path / "history.jsonl"
    append_style_run(record_at([("ends_q", 1, 2)], when="t1"), path)
    append_style_run(record_at([("ends_q", 2, 2)], when="t2"), path)

    assert [r.recorded_at for r in load_style_history(path)] == ["t1", "t2"]


def test_a_missing_history_is_empty_not_an_error(tmp_path):
    """The first run of a new contract has no history."""
    assert load_style_history(tmp_path / "nope.jsonl") == ()


def test_history_can_be_filtered_by_fingerprint(tmp_path):
    path = tmp_path / "history.jsonl"
    append_style_run(record_at([("ends_q", 1, 2)], fingerprint="aaa"), path)
    append_style_run(record_at([("ends_q", 1, 2)], fingerprint="bbb"), path)

    assert len(load_style_history(path, fingerprint="aaa")) == 1


def test_a_corrupt_line_names_itself(tmp_path):
    path = tmp_path / "history.jsonl"
    append_style_run(record_at([("ends_q", 1, 2)]), path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")

    with pytest.raises(AgentConfigurationError, match="line 2"):
        load_style_history(path)


def test_records_are_written_one_json_object_per_line(tmp_path):
    path = tmp_path / "history.jsonl"
    append_style_run(record_at([("ends_q", 1, 2)]), path)
    append_style_run(record_at([("ends_q", 1, 2)]), path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["fingerprint"] == "fp" for line in lines)


def test_relabel_leaves_the_fingerprint_alone():
    """The label is for reading; the fingerprint is the identity."""
    made = record_at([("ends_q", 1, 2)])
    again = relabel(made, "my_agent v0.1.19")
    assert again.subject == "my_agent v0.1.19"
    assert again.fingerprint == made.fingerprint


# ── Comparing ──────────────────────────────────────────────


def test_one_prior_invocation_is_not_a_baseline():
    """A single earlier run is a point, not a band.

    This is the whole lesson: two invocations of an unchanged
    chat agent config differed by 29 points, so a before-and-after of
    one run each cannot tell a change from that variation.
    """
    current = record_at([("ends_q", 5, 10)], when="t2")
    history = [record_at([("ends_q", 1, 10)], when="t1")]

    result = compare_to_history(current, history)

    assert result.rules[0].verdict is Verdict.NO_BASELINE
    assert result.rules[0].band is None
    assert result.unbaselined
    assert "No baseline yet" in result.summary()


def test_a_rate_inside_the_band_is_not_a_change():
    """The band from today's real numbers from one agent.

    Two invocations of the unchanged config gave 23% and 70%. A
    third gave 41%. Anything that calls that a regression is
    measuring the provider, not the config.
    """
    history = [
        record_at([("dry", 11, 48)], when="t1"),
        record_at([("dry", 56, 80)], when="t2"),
    ]
    current = record_at([("dry", 33, 80)], when="t3")

    result = compare_to_history(current, history)

    assert result.rules[0].verdict is Verdict.WITHIN_BASELINE
    assert not result.worse
    assert not result.better


def test_a_rate_above_the_band_is_worse():
    history = [
        record_at([("dry", 2, 10)], when="t1"),
        record_at([("dry", 3, 10)], when="t2"),
    ]
    current = record_at([("dry", 9, 10)], when="t3")

    result = compare_to_history(current, history)

    assert result.rules[0].verdict is Verdict.ABOVE_BASELINE
    assert result.rules[0].changed
    assert [r.rule_id for r in result.worse] == ["dry"]
    assert "WORSE" in result.summary()


def test_a_rate_below_the_band_is_better():
    history = [
        record_at([("dry", 8, 10)], when="t1"),
        record_at([("dry", 9, 10)], when="t2"),
    ]
    current = record_at([("dry", 1, 10)], when="t3")

    result = compare_to_history(current, history)

    assert result.rules[0].verdict is Verdict.BELOW_BASELINE
    assert [r.rule_id for r in result.better] == ["dry"]


def test_only_the_same_configuration_forms_a_baseline():
    """A band from a different agent describes a different agent."""
    history = [
        record_at([("dry", 1, 10)], fingerprint="other", when="t1"),
        record_at([("dry", 1, 10)], fingerprint="other", when="t2"),
    ]
    current = record_at([("dry", 9, 10)], fingerprint="mine", when="t3")

    result = compare_to_history(current, history)

    assert result.prior == ()
    assert result.rules[0].verdict is Verdict.NO_BASELINE


def test_a_baseline_can_be_borrowed_from_a_named_configuration():
    """Comparing a changed config against the old one is explicit.

    The default refuses to cross configurations; asking for it by
    fingerprint is how a real before-and-after is requested.
    """
    history = [
        record_at([("dry", 1, 10)], fingerprint="old", when="t1"),
        record_at([("dry", 2, 10)], fingerprint="old", when="t2"),
    ]
    current = record_at([("dry", 9, 10)], fingerprint="new", when="t3")

    result = compare_to_history(current, history, fingerprint="old")

    assert result.rules[0].verdict is Verdict.ABOVE_BASELINE
    assert result.rules[0].band == (0.1, 0.2)


def test_the_record_being_judged_is_not_its_own_baseline():
    """A record already in the history must not compare to itself."""
    current = record_at([("dry", 9, 10)], when="t3")
    history = [
        record_at([("dry", 1, 10)], when="t1"),
        record_at([("dry", 2, 10)], when="t2"),
        current,
    ]

    result = compare_to_history(current, history)

    assert len(result.prior) == 2
    assert result.rules[0].verdict is Verdict.ABOVE_BASELINE


def test_an_unexercised_rule_gets_no_verdict():
    """Nothing triggered it, so there is nothing to compare."""
    history = [
        record_at([("dry", 1, 10)], when="t1"),
        record_at([("dry", 2, 10)], when="t2"),
    ]
    current = record_at([("dry", 0, 0)], when="t3")

    result = compare_to_history(current, history)

    assert result.rules[0].verdict is Verdict.UNEXERCISED
    assert result.rules[0].current is None


def test_a_rule_absent_from_history_has_no_baseline():
    """A newly added rule starts with nothing behind it."""
    history = [
        record_at([("old_rule", 1, 10)], when="t1"),
        record_at([("old_rule", 2, 10)], when="t2"),
    ]
    current = record_at([("new_rule", 9, 10)], when="t3")

    result = compare_to_history(current, history)

    assert result.rules[0].verdict is Verdict.NO_BASELINE


# ── A rule's history is its own ────────────────────────────


def _with_rule(base, rule_id):
    """*base* with one more rule, scored by the same check."""
    return StyleContract(
        rules=(
            *base.rules,
            StyleRule(
                id=rule_id,
                description="d",
                check=ends_with_question(),
                budget=None,
            ),
        ),
        scripts=base.scripts,
    )


def test_the_configuration_key_ignores_the_rule_set():
    """A rule added scores the replies; it does not change them."""
    agent = make_chatter()
    assert configuration_key(
        _with_rule(contract(), "other"), agent
    ) == configuration_key(contract(), agent)


def test_the_configuration_key_still_follows_agent_and_scripts():
    """Only the rule ids left; everything that shapes a reply stays."""
    before = configuration_key(contract(), make_chatter())
    other_agent = make_chatter(system_prompt="You are someone else.")
    assert configuration_key(contract(), other_agent) != before
    assert configuration_key(contract(lines=("b",)), make_chatter()) != before


def test_adding_a_rule_keeps_the_other_rules_baselines():
    """The case that cost a baseline every time a rule was added.

    The fingerprints differ because the rule sets do; the key does
    not, so `dry` is still judged against its band and only the new
    rule starts from nothing.
    """
    history = [
        record_at(
            [("dry", 1, 10)], fingerprint="a", config_key="k", when="t1"
        ),
        record_at(
            [("dry", 2, 10)], fingerprint="a", config_key="k", when="t2"
        ),
    ]
    current = record_at(
        [("dry", 9, 10), ("new", 3, 10)],
        fingerprint="b",
        config_key="k",
        when="t3",
    )

    result = compare_to_history(current, history)

    assert {r.rule_id: r.verdict for r in result.rules} == {
        "dry": Verdict.ABOVE_BASELINE,
        "new": Verdict.NO_BASELINE,
    }


def test_a_different_configuration_key_is_not_a_baseline():
    """When both records carry a key, the key decides."""
    history = [
        record_at([("dry", 1, 10)], config_key="other", when="t1"),
        record_at([("dry", 2, 10)], config_key="other", when="t2"),
    ]
    current = record_at([("dry", 9, 10)], config_key="k", when="t3")

    assert compare_to_history(current, history).prior == ()


def test_a_record_without_a_key_is_matched_on_its_fingerprint():
    """History written before v9.1.0 keeps comparing as it did."""
    history = [
        record_at([("dry", 1, 10)], when="t1"),
        record_at([("dry", 2, 10)], when="t2"),
    ]
    current = record_at([("dry", 9, 10)], config_key="k", when="t3")

    assert len(compare_to_history(current, history).prior) == 2


def test_a_rule_whose_check_changed_starts_its_own_band():
    """The hole the rule-id key left open.

    Changing a check's arguments under the same id kept every key,
    so the new numbers were judged against a band the old check
    built. A different definition now leaves that rule unbanded.
    """
    history = [
        record_at([("dry", 1, 10)], definitions={"dry": "old"}, when="t1"),
        record_at([("dry", 2, 10)], definitions={"dry": "old"}, when="t2"),
    ]
    current = record_at(
        [("dry", 9, 10)], definitions={"dry": "new"}, when="t3"
    )

    result = compare_to_history(current, history)

    assert result.rules[0].verdict is Verdict.NO_BASELINE


def test_a_rule_recorded_before_definitions_still_counts():
    """A missing definition is unknown, not different."""
    history = [
        record_at([("dry", 1, 10)], when="t1"),
        record_at([("dry", 2, 10)], when="t2"),
    ]
    current = record_at([("dry", 9, 10)], definitions={"dry": "d"}, when="t3")

    result = compare_to_history(current, history)

    assert result.rules[0].verdict is Verdict.ABOVE_BASELINE


def test_a_record_carries_each_definition_and_its_configuration_key():
    ruled = StyleContract(
        rules=(replace(contract().rules[0], definition="abc"),),
        scripts=contract().scripts,
    )
    agent = make_chatter()
    report = score_transcripts(ruled, [[("a", "¿y tú?")]])

    made = record_style_run(report, ruled, agent, "v1")

    assert made.rule("ends_q").definition == "abc"
    assert made.config_key == configuration_key(ruled, agent, report)


def test_the_summary_names_the_key_it_compared_on():
    """Not the fingerprint, which the priors no longer share.

    Seen on one agent's first run after a rule was added: the header
    named the new fingerprint as "the same configuration" while every
    prior carried the old one.
    """
    history = [
        record_at([("dry", 1, 10)], fingerprint="fpold1", config_key="cfg1"),
    ]
    current = record_at(
        [("dry", 9, 10)], fingerprint="fpnew1", config_key="cfg1", when="t2"
    )

    header = compare_to_history(current, history).summary().splitlines()[0]

    assert "key cfg1" in header
    assert "fpnew1" not in header


def test_definitions_and_the_key_survive_a_round_trip(tmp_path):
    made = record_at(
        [("dry", 1, 10)], config_key="k", definitions={"dry": "d"}
    )
    path = append_style_run(made, tmp_path / "h.jsonl")

    (loaded,) = load_style_history(path)

    assert loaded.config_key == "k"
    assert loaded.rule("dry").definition == "d"


def test_a_record_written_before_definitions_reads_none(tmp_path):
    raw = record_at([("dry", 1, 10)]).to_dict()
    del raw["config_key"]
    for rule_raw in raw["rules"]:
        del rule_raw["definition"]
    path = tmp_path / "h.jsonl"
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    (loaded,) = load_style_history(path)

    assert loaded.config_key is None
    assert loaded.rule("dry").definition is None


def test_a_baseline_only_counts_runs_answered_by_the_same_model():
    """The bug this fixes: one key, two models, one band.

    One agent's unchanged config scored a rule at 75% on
    `deepseek-v4-flash` and 10% on `deepseek-flash`, and both sat
    in the band its next run was judged against.
    """
    old, new = ("m::old",), ("m::new",)
    history = [
        record_at([("dry", 8, 10)], when="t1", answered_by=old),
        record_at([("dry", 7, 10)], when="t2", answered_by=old),
        record_at([("dry", 1, 10)], when="t3", answered_by=new),
        record_at([("dry", 2, 10)], when="t4", answered_by=new),
    ]
    current = record_at([("dry", 1, 10)], when="t5", answered_by=new)

    result = compare_to_history(current, history)

    assert [r.recorded_at for r in result.prior] == ["t3", "t4"]
    assert [r.recorded_at for r in result.other_models] == ["t1", "t2"]
    assert result.rules[0].band == (0.1, 0.2)


def test_the_summary_names_the_model_that_was_left_out():
    """A switch is reported, not silently reset."""
    history = [
        record_at([("dry", 8, 10)], when="t1", answered_by=("m::old",)),
    ]
    current = record_at([("dry", 1, 10)], when="t2", answered_by=("m::new",))

    summary = compare_to_history(current, history).summary()

    assert "Answered by m::new" in summary
    assert "answered by m::old" in summary


def test_runs_with_no_recorded_model_are_left_out_once_one_is():
    """An unrecorded model cannot be shown to be the same one."""
    history = [
        record_at([("dry", 8, 10)], when="t1"),
        record_at([("dry", 7, 10)], when="t2"),
    ]
    current = record_at([("dry", 1, 10)], when="t3", answered_by=("m::a",))

    result = compare_to_history(current, history)

    assert result.prior == ()
    assert len(result.other_models) == 2
    assert "no recorded model" in result.summary()


def test_a_baseline_only_counts_runs_whose_guard_answered_alike():
    """Same chatter model, guard moved underneath: not one band."""
    chat = ("m::chat",)
    history = [
        record_at(
            [("dry", 8, 10)],
            when="t1",
            answered_by=chat,
            guard_answered_by=("g::old",),
        ),
        record_at(
            [("dry", 1, 10)],
            when="t2",
            answered_by=chat,
            guard_answered_by=("g::new",),
        ),
    ]
    current = record_at(
        [("dry", 1, 10)],
        when="t3",
        answered_by=chat,
        guard_answered_by=("g::new",),
    )

    result = compare_to_history(current, history)

    assert [r.recorded_at for r in result.prior] == ["t2"]
    assert [r.recorded_at for r in result.other_models] == ["t1"]
    assert "(guard g::old)" in result.summary()


def test_a_record_with_no_guard_model_ignores_the_priors_guard():
    """An unguarded run records no guard model and compares as
    before, whatever its priors' guards were."""
    history = [
        record_at([("dry", 1, 10)], when="t1", guard_answered_by=("g::a",)),
    ]
    current = record_at([("dry", 1, 10)], when="t2")

    result = compare_to_history(current, history)

    assert len(result.prior) == 1
    assert result.other_models == ()


def test_a_record_with_no_model_compares_as_before():
    """Transcript scoring makes no calls, so it records no model."""
    history = [
        record_at([("dry", 1, 10)], when="t1", answered_by=("m::a",)),
        record_at([("dry", 2, 10)], when="t2"),
    ]
    current = record_at([("dry", 1, 10)], when="t3")

    result = compare_to_history(current, history)

    assert len(result.prior) == 2
    assert result.other_models == ()


def test_min_baseline_runs_is_the_floor_not_a_recommendation():
    """Documented as a floor, and the summary reports the count."""
    assert MIN_BASELINE_RUNS == 2
    history = [record_at([("dry", 1, 10)], when=f"t{i}") for i in range(2)]
    current = record_at([("dry", 1, 10)], when="t9")
    assert (
        "2 prior invocation(s)"
        in compare_to_history(current, history).summary()
    )


# ── Sampling margin ────────────────────────────────────────


def test_a_handful_of_events_does_not_flag_on_one_more():
    """The real false alarm this margin was added to stop.

    Two invocations of one agent saw one stray greeting each, giving a
    band of roughly 0.6%-1.0%. A third saw four out of 175. Being
    outside the band alone called that a regression; four events
    against one is not distinguishable at that sample size.
    """
    history = [
        record_at([("greets", 1, 105)], when="t1"),
        record_at([("greets", 1, 175)], when="t2"),
    ]
    current = record_at([("greets", 4, 175)], when="t3")

    result = compare_to_history(current, history)

    assert result.rules[0].verdict is Verdict.WITHIN_BASELINE
    assert not result.worse


def test_a_large_move_still_flags_despite_the_margin():
    """The margin must not swallow a real change."""
    history = [
        record_at([("dry", 2, 80)], when="t1"),
        record_at([("dry", 3, 80)], when="t2"),
    ]
    current = record_at([("dry", 60, 80)], when="t3")

    assert (
        compare_to_history(current, history).rules[0].verdict
        is Verdict.ABOVE_BASELINE
    )


# ── Rarely broken rules are reported, not judged ───────────


def test_a_rarely_broken_rule_is_reported_not_judged():
    """The false failure that made rare rules report-only.

    One agent's gate failed `no_trailing_laugh` at 3/111 against
    priors that were almost all zero, on an unchanged config, and
    the three laughs were in three different conversations.
    """
    history = [
        record_at([("laugh", v, 111)], when=f"t{i}")
        for i, v in enumerate((0, 1, 0, 0))
    ]
    current = record_at([("laugh", 3, 111)], when="t9")

    result = compare_to_history(current, history)

    assert result.rules[0].verdict is Verdict.RARE
    assert not result.rules[0].changed
    assert not result.worse
    assert "rare" in result.summary()


def test_a_rare_rule_is_not_called_better_either():
    history = [
        record_at([("laugh", v, 111)], when=f"t{i}")
        for i, v in enumerate((0, 1, 1, 1))
    ]
    current = record_at([("laugh", 0, 111)], when="t9")

    assert not compare_to_history(current, history).better


def test_a_rule_broken_once_per_invocation_is_still_judged():
    """The threshold is below one violation, not at it."""
    history = [
        record_at([("dry", 1, 111)], when=f"t{i}")
        for i in range(MIN_SPREAD_PRIORS)
    ]
    current = record_at([("dry", 60, 111)], when="t9")

    assert (
        compare_to_history(current, history).rules[0].verdict
        is Verdict.ABOVE_BASELINE
    )


def test_a_quiet_rule_that_jumps_is_not_judged_either():
    """The accepted price of report-only rare rules.

    One agent's v0.2.9 gate scored `one_message_when_dry` at 10/48
    after priors of 1, 1, 0 and 0. It is shown, not failed.
    """
    history = [
        record_at([("dry", v, 48)], when=f"t{i}")
        for i, v in enumerate((1, 1, 0, 0))
    ]
    current = record_at([("dry", 10, 48)], when="t9")

    result = compare_to_history(current, history)

    assert result.rules[0].verdict is Verdict.RARE
    assert not result.worse


def test_a_rare_rule_still_needs_a_baseline_first():
    history = [record_at([("laugh", 0, 111)], when="t1")]
    current = record_at([("laugh", 3, 111)], when="t2")

    assert (
        compare_to_history(current, history).rules[0].verdict
        is Verdict.NO_BASELINE
    )


def test_unexercised_priors_do_not_make_a_rule_rare():
    """A prior that never triggered the rule says nothing about it."""
    history = [
        record_at([("dry", 0, 0)], when="t1"),
        record_at([("dry", 20, 80)], when="t2"),
        record_at([("dry", 22, 80)], when="t3"),
    ]
    current = record_at([("dry", 70, 80)], when="t9")

    assert (
        compare_to_history(current, history).rules[0].verdict
        is Verdict.ABOVE_BASELINE
    )


def test_the_margin_widens_as_the_sample_shrinks():
    """Fewer replies, less certainty, a wider margin."""
    assert sampling_margin(5, 20) > sampling_margin(25, 100)


def test_a_zero_rate_still_carries_a_margin():
    """Zero violations is not perfect precision.

    Without the half-event floor the margin would be 0.0 and any
    band above zero would read as an improvement.
    """
    assert sampling_margin(0, 100) > 0.0


def test_the_margin_is_zero_when_nothing_applied():
    assert sampling_margin(0, 0) == 0.0


# ── Clustering ─────────────────────────────────────────────
#
# A contract plays a script, so its replies are not drawn
# independently: every reply after the first answers an agent that
# has read its own previous ones. The run is the independent unit,
# and a margin computed over the pooled replies understates the
# uncertainty by however much one conversation agrees with itself.
# On one agent that understatement made three invocations of an
# unchanged config look like two regressions.


def test_runs_that_disagree_widen_the_margin():
    """The spread across runs beats the pooled denominator."""
    assert sampling_margin(56, 80, (0.25, 1.0, 0.5, 1.0, 0.75)) > (
        sampling_margin(56, 80)
    )


def test_runs_that_agree_buy_no_extra_confidence():
    """The binomial figure is a floor, not a starting point.

    Five runs landing on the same rate does not make the rate
    better measured than its sample size allows.
    """
    assert sampling_margin(56, 80, (0.7,) * 5) == sampling_margin(56, 80)


def test_a_single_run_is_not_a_spread():
    """One run has nothing to disagree with."""
    assert sampling_margin(56, 80, (0.7,)) == sampling_margin(56, 80)


def test_a_rule_whose_conversations_disagree_is_not_called_changed():
    """The failure this whole section exists for.

    One agent's `one_message_when_dry` scored 23%, 70% and 41% over
    three invocations of one unchanged config, and the pooled
    margin called that a regression. Its runs disagree by enough to
    produce that move on their own, so the honest verdict is that
    nothing was distinguished.
    """
    history = [
        record_at([("dry", 20, 80)], when="t1"),
        record_at([("dry", 36, 80)], when="t2"),
    ]
    current = record_at(
        [("dry", 56, 80, (0.25, 1.0, 0.5, 1.0, 0.75))], when="t3"
    )

    assert (
        compare_to_history(current, history).rules[0].verdict
        is Verdict.WITHIN_BASELINE
    )


def test_a_rule_whose_conversations_agree_still_flags():
    """Widening the margin must not blind the instrument.

    The same pooled move as above, from runs that agree with each
    other, is still a change — which is what makes the verdict
    above informative rather than merely cautious.
    """
    history = [
        record_at([("dry", 20, 80)], when="t1"),
        record_at([("dry", 36, 80)], when="t2"),
    ]
    current = record_at(
        [("dry", 56, 80, (0.6875, 0.6875, 0.75, 0.6875, 0.6875))],
        when="t3",
    )

    assert (
        compare_to_history(current, history).rules[0].verdict
        is Verdict.ABOVE_BASELINE
    )


def test_a_record_without_run_rates_falls_back_to_binomial():
    """History written before run rates were persisted still reads.

    One agent's first three records were seeded from printed run
    output and carry none, so this is the state of a real file.
    """
    history = [
        record_at([("dry", 20, 80)], when="t1"),
        record_at([("dry", 36, 80)], when="t2"),
    ]
    current = record_at([("dry", 56, 80)], when="t3")

    assert (
        compare_to_history(current, history).rules[0].verdict
        is Verdict.ABOVE_BASELINE
    )


def test_a_binomial_fallback_margin_is_marked_as_one():
    """A weaker figure must not print like the real one.

    Every margin one agent's own history can show today is this
    fallback, and unmarked it reads as the measured spread.
    """
    history = [
        record_at([("dry", 20, 80)], when="t1"),
        record_at([("dry", 36, 80)], when="t2"),
    ]
    current = record_at([("dry", 56, 80)], when="t3")

    comparison = compare_to_history(current, history)
    assert comparison.rules[0].margin_source is MarginSource.REPLIES
    assert "**" in comparison.summary()


def test_a_margin_taken_from_this_run_is_marked_as_a_placeholder():
    """The within-run spread is a proxy and must read as one.

    It ran two to three times too wide on four of one agent's rules
    and too narrow on a fifth, so printing it unmarked invites the
    reader to trust a bar that hides real regressions.
    """
    history = [
        record_at([("dry", 20, 80)], when="t1"),
        record_at([("dry", 36, 80)], when="t2"),
    ]
    current = record_at(
        [("dry", 56, 80, (0.25, 1.0, 0.5, 1.0, 0.75))], when="t3"
    )

    comparison = compare_to_history(current, history)
    assert comparison.rules[0].margin_source is MarginSource.RUNS
    assert "*" in comparison.summary()


# ── The margin comes from the prior invocations ────────────
#
# What is being judged is whether the *invocation's* score
# moved, so the honest bar is how much that score has moved
# before. Everything above is a proxy for it, and measured on
# one agent not a close one.


def test_enough_priors_replace_the_proxy_with_their_own_spread():
    history = [
        record_at([("dry", v, 80, (0.2, 0.9, 0.4))], when=f"t{i}")
        for i, v in enumerate((40, 42, 44, 46))
    ]
    current = record_at([("dry", 44, 80, (0.2, 0.9, 0.4))], when="t9")

    comparison = compare_to_history(current, history)

    assert comparison.rules[0].margin_source is MarginSource.PRIORS
    assert comparison.rules[0].margin == pytest.approx(
        2.0 * _spread_upper_bound([0.5, 0.525, 0.55, 0.575])
    )


# ── Four agreeing priors must not make a hair trigger ──────
#
# A standard deviation from four values has three degrees of
# freedom, so its point estimate is often far too small. The
# spread is read at its upper `SPREAD_CONFIDENCE` bound instead.


def _agreeing_priors():
    return [
        record_at([("copies", v, 1000)], when=f"t{i}")
        for i, v in enumerate((149, 150, 160, 165))
    ]


def test_four_agreeing_priors_do_not_gate_on_a_small_move():
    """One agent, 15th September 2026: an unchanged config failed.

    `copies_an_example` scored 18.4% against priors of 14.9%-16.5%,
    and the point estimate of their spread gave a 1.4-point margin.
    """
    current = record_at([("copies", 184, 1000)], when="t9")

    result = compare_to_history(current, _agreeing_priors())

    assert result.rules[0].verdict is Verdict.WITHIN_BASELINE


def test_four_agreeing_priors_still_catch_a_large_move():
    current = record_at([("copies", 260, 1000)], when="t9")

    result = compare_to_history(current, _agreeing_priors())

    assert result.rules[0].verdict is Verdict.ABOVE_BASELINE


def test_the_upper_bound_fades_as_priors_accumulate():
    few = [0.40, 0.42, 0.44, 0.46]
    many = few * 10

    few_factor = _spread_upper_bound(few) / stdev(few)
    many_factor = _spread_upper_bound(many) / stdev(many)

    assert few_factor > many_factor > 1.0


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_chi2_quantile_matches_its_closed_forms(p):
    """Two degrees of freedom is exponential; one is a squared normal."""
    assert _chi2_quantile(p, 2) == pytest.approx(-2.0 * log(1.0 - p))
    z = NormalDist().inv_cdf((1.0 + p) / 2.0)
    assert _chi2_quantile(p, 1) == pytest.approx(z * z)


def test_a_prior_shift_carries_the_margins_upper_bound():
    """The bar `detectable_shift` sizes is the one the gate applies."""
    priors = (0.40, 0.42, 0.44, 0.46)

    estimate = detectable_shift(
        planned_runs=4, prior_rates=priors, prior_runs_each=4
    )

    assert estimate.rate == pytest.approx(
        _expected_max(MIN_BASELINE_RUNS) * stdev(priors)
        + 2.0 * _spread_upper_bound(priors)
    )


def test_the_proxy_holds_until_the_priors_are_enough():
    """Three is not enough; the fourth is what switches it."""
    runs = (0.2, 0.9, 0.4)
    history = [
        record_at([("dry", 40 + i, 80, runs)], when=f"t{i}")
        for i in range(MIN_SPREAD_PRIORS - 1)
    ]
    current = record_at([("dry", 44, 80, runs)], when="t9")

    assert (
        compare_to_history(current, history).rules[0].margin_source
        is MarginSource.RUNS
    )

    history.append(record_at([("dry", 46, 80, runs)], when="t8"))
    assert (
        compare_to_history(current, history).rules[0].margin_source
        is MarginSource.PRIORS
    )


def test_the_priors_margin_is_not_floored_by_the_proxy():
    """A `max()` of the two would keep every too-wide rule wide.

    Which is the failure this exists to fix: on one agent's gate
    history the within-run figure ran three times the real
    movement, so keeping it as a floor would change nothing.
    """
    scattered = (0.0, 1.0, 0.0, 1.0)
    history = [
        record_at([("dry", v, 80, scattered)], when=f"t{i}")
        for i, v in enumerate((40, 40, 41, 41))
    ]
    current = record_at([("dry", 41, 80, scattered)], when="t9")

    comparison = compare_to_history(current, history)

    assert comparison.rules[0].margin < sampling_margin(41, 80, scattered)


def test_agreeing_priors_still_buy_no_more_than_half_an_event():
    """Four priors that happen to agree must not gate on one event.

    Walked forward over one agent's gate history, two priors gave a
    spread of exactly 0.0% on `no_trailing_laugh`, and a rule whose
    bar is zero flags on the next single event. The floor is what
    stops that surviving into the four-prior case.
    """
    history = [
        record_at([("rare", 0, 80, (0.0, 0.0))], when=f"t{i}")
        for i in range(4)
    ]
    current = record_at([("rare", 1, 80, (0.0, 0.05))], when="t9")

    comparison = compare_to_history(current, history)

    assert comparison.rules[0].margin == pytest.approx(2.0 * 0.5 / 80)
    # Priors of zero also make it a rare rule, which is not judged.
    assert comparison.rules[0].verdict is Verdict.RARE
    assert not comparison.worse


def test_a_margin_taken_from_the_priors_is_not_marked():
    history = [
        record_at([("dry", v, 80)], when=f"t{i}")
        for i, v in enumerate((40, 42, 44, 46))
    ]
    current = record_at([("dry", 44, 80)], when="t9")

    assert "*" not in compare_to_history(current, history).summary()


# ── Sizing a contract ──────────────────────────────────────
#
# `sampling_margin` says how uncertain a finished run is.
# These say how many runs it would take to be certain enough,
# which is the question asked when setting a contract's
# `repeats` — and the one whose intuitive answer, lengthen the
# scripts, is wrong.


def test_the_expected_maximum_matches_its_closed_forms():
    """Two draws give 1/sqrt(pi), three give 3/(2*sqrt(pi))."""
    root_pi = math.pi**0.5
    assert _expected_max(2) == pytest.approx(1.0 / root_pi, abs=1e-6)
    assert _expected_max(3) == pytest.approx(1.5 / root_pi, abs=1e-6)


def test_one_draw_has_no_maximum_to_speak_of():
    assert _expected_max(1) == 0.0


def test_runs_that_disagree_need_more_of_them():
    """The spread is the whole input — a wider one costs runs."""
    steady = runs_for_shift(0.10, (0.48, 0.50, 0.52, 0.50))
    scattered = runs_for_shift(0.10, (0.20, 0.80, 0.45, 0.55))
    assert scattered.runs > steady.runs


def test_four_times_the_runs_sees_half_the_move():
    """The square-root law, which is what makes runs expensive."""
    rates = (0.20, 0.80, 0.45, 0.55)
    assert detectable_shift(rates, 40).rate == pytest.approx(
        detectable_shift(rates, 10).rate / 2.0
    )


def test_more_runs_see_a_smaller_move():
    rates = (0.20, 0.80, 0.45, 0.55)
    assert detectable_shift(rates, 20).rate < detectable_shift(rates, 5).rate


def test_a_longer_baseline_raises_the_bar():
    """The maximum of five priors sits further above the truth.

    Counterintuitive and real: a min/max band is harder to
    escape the more invocations it was drawn from.
    """
    rates = (0.20, 0.80, 0.45, 0.55)
    assert (
        detectable_shift(rates, 10, prior_invocations=5).rate
        > detectable_shift(rates, 10, prior_invocations=2).rate
    )


def test_the_count_it_returns_actually_sees_the_shift():
    """The two functions agree — the inverse really inverts."""
    rates = (0.20, 0.80, 0.45, 0.55)
    needed = runs_for_shift(0.10, rates).runs
    assert detectable_shift(rates, needed).rate <= 0.10
    assert detectable_shift(rates, needed - 1).rate > 0.10


def test_runs_default_to_however_many_were_observed():
    rates = (0.20, 0.80, 0.45, 0.55)
    assert detectable_shift(rates).rate == detectable_shift(rates, 4).rate
    assert detectable_shift(rates).runs == 4


def test_runs_that_agreed_exactly_report_no_spread():
    """Unmeasured, not absent — no rule truly has zero spread."""
    assert detectable_shift((0.5, 0.5, 0.5), 10).rate == 0.0


def test_a_single_run_cannot_size_anything():
    with pytest.raises(AgentConfigurationError, match="needs a spread"):
        detectable_shift((0.5,), 10)


def test_a_shift_worth_catching_is_a_positive_one():
    with pytest.raises(AgentConfigurationError, match="must be positive"):
        runs_for_shift(0.0, (0.2, 0.8))


def test_an_invocation_has_at_least_one_run():
    with pytest.raises(AgentConfigurationError, match="at least 1"):
        detectable_shift((0.2, 0.8), 0)


# ── Which spread a sizing answer was built from ────────────
#
# The same three sources `sampling_margin` takes, in the same
# order. Sizing used to be given no choice but the weakest, and
# on one agent's five committed invocations that one was wrong
# by between 0.38 and 3.76 times — in both directions, so no
# constant corrects it.


def test_bare_rates_are_the_weakest_source():
    assert detectable_shift((0.2, 0.8, 0.5, 0.4)).source is (
        MarginSource.RUNS
    )


def test_labelled_runs_keep_two_scripts_apart():
    """A fixed gap between scripts is structure, not noise.

    Both scripts are steady inside themselves and far apart from
    each other. Pooling reads that gap as run-to-run variation and
    demands a far larger move than the contract really needs.
    """
    runs = tuple(
        RunScore(label=f"{script}#{i}", violations=v, applicable=100)
        for script, v in (("cold", 80), ("warm", 20))
        for i in range(3)
    )
    labelled = detectable_shift(runs=runs, planned_runs=6)
    pooled = detectable_shift(tuple(r.rate for r in runs), 6)

    assert labelled.source is MarginSource.RUNS
    assert labelled.rate < pooled.rate / 10.0


def test_priors_outrank_the_runs_inside_one_invocation():
    """Draws of the quantity being judged beat a proxy for it."""
    estimate = detectable_shift(
        (0.20, 0.80, 0.45, 0.55),
        prior_rates=(0.40, 0.42, 0.44, 0.46),
        prior_runs_each=4,
    )
    assert estimate.source is MarginSource.PRIORS


def test_three_priors_are_too_few_to_outrank_anything():
    estimate = detectable_shift(
        (0.20, 0.80, 0.45, 0.55),
        prior_rates=(0.40, 0.42, 0.44),
        prior_runs_each=4,
    )
    assert estimate.source is MarginSource.RUNS


def test_a_prior_spread_is_rescaled_to_the_planned_size():
    """Four times the runs, half the spread — the same law."""
    priors = (0.40, 0.42, 0.44, 0.46)
    at_four = detectable_shift(
        planned_runs=4, prior_rates=priors, prior_runs_each=4
    )
    at_sixteen = detectable_shift(
        planned_runs=16, prior_rates=priors, prior_runs_each=4
    )
    assert at_sixteen.rate == pytest.approx(at_four.rate / 2.0)


def test_a_prior_spread_without_its_run_count_is_refused():
    """Their spread means nothing without the size behind it."""
    with pytest.raises(AgentConfigurationError, match="prior_runs_each"):
        detectable_shift(prior_rates=(0.40, 0.42, 0.44, 0.46))


def test_priors_alone_can_size_a_contract():
    """No run rates at all, and the answer still comes out."""
    estimate = runs_for_shift(
        0.10, prior_rates=(0.40, 0.42, 0.44, 0.46), prior_runs_each=10
    )
    assert estimate.source is MarginSource.PRIORS
    assert estimate.rate == 0.10
    assert estimate.runs >= 1


def test_nothing_at_all_cannot_size_a_contract():
    with pytest.raises(AgentConfigurationError, match="needs a spread"):
        runs_for_shift(0.10)


def test_both_functions_fill_every_field():
    """One shape, asked from either end — see `ShiftEstimate`."""
    rates = (0.20, 0.80, 0.45, 0.55)
    forward = detectable_shift(rates, 10)
    backward = runs_for_shift(0.10, rates)

    for estimate in (forward, backward):
        assert isinstance(estimate, ShiftEstimate)
        assert estimate.rate > 0.0
        assert estimate.runs >= 1
        assert estimate.source in tuple(MarginSource)


# ── Which script a run came from ───────────────────────────
#
# `RuleResult.per_run` always knew — its label names the script
# and the repeat. Persisting the rates alone threw that away, so
# once a run reached `style_history.jsonl` the cold conversations
# could not be told from the warm ones, and the margin computed
# from them counted the gap between two scripts as noise.


def scored(label, violations, applicable):
    return RunScore(label=label, violations=violations, applicable=applicable)


def test_a_record_says_which_script_each_run_played():
    """The whole item: the label has to survive being persisted."""
    report = score_transcripts(
        contract(),
        [[("a", "¿y tú?")], [("a", "pues eso")]],
    )
    made = record_style_run(report, contract(), make_chatter(), "v1")

    rule = made.rule("ends_q")
    assert [r.label for r in rule.runs] == [
        "conversation#1",
        "conversation#2",
    ]
    assert [r.script for r in rule.runs] == ["conversation"] * 2


def test_the_labels_survive_a_round_trip_through_json(tmp_path):
    """A file read back has to answer what the run answered."""
    report = score_transcripts(
        contract(),
        [[("a", "¿y tú?")], [("a", "pues eso")]],
    )
    made = record_style_run(report, contract(), make_chatter(), "v1")
    path = append_style_run(made, tmp_path / "h.jsonl")

    back = load_style_history(path)[0].rule("ends_q")
    assert back.runs == made.rule("ends_q").runs
    assert back.run_rates == made.rule("ends_q").run_rates


def test_the_flat_rates_are_derived_when_they_are_not_given():
    """One accessor answers for a record of either shape."""
    record = RuleRecord(
        id="r",
        budget=None,
        violations=3,
        applicable=6,
        runs=(scored("cold#1", 3, 4), scored("warm#1", 0, 2)),
    )

    assert record.run_rates == (0.75, 0.0)


def test_a_run_that_triggered_nothing_is_kept_but_has_no_rate():
    """The flat view drops it; the runs still say it happened."""
    record = RuleRecord(
        id="r",
        budget=None,
        violations=1,
        applicable=2,
        runs=(scored("cold#1", 1, 2), scored("warm#1", 0, 0)),
    )

    assert len(record.runs) == 2
    assert record.run_rates == (0.5,)


def test_a_record_written_before_labels_existed_still_loads(tmp_path):
    """The history is append-only, so old lines have to keep working."""
    line = {
        "recorded_at": "2026-08-30T10:00:00+00:00",
        "subject": "old",
        "fingerprint": "fp",
        "agent": "a",
        "agent_version": 1,
        "model_id": "ollama::x",
        "repeats": 2,
        "replies": 4,
        "calls": 4,
        "cost_cents": None,
        "rules": [
            {
                "id": "r",
                "budget": None,
                "violations": 2,
                "applicable": 4,
                "run_rates": [0.75, 0.25],
            }
        ],
    }
    path = tmp_path / "h.jsonl"
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")

    rule = load_style_history(path)[0].rule("r")
    assert rule.runs == ()
    assert rule.run_rates == (0.75, 0.25)
    assert rule.rate == 0.5


# ── A pooled rate split by script ──────────────────────────


def split_record(runs, when="2026-08-30T12:00:00+00:00"):
    """A one-rule record whose score comes from labelled *runs*."""
    return replace(
        record_at([], when=when),
        rules=(
            RuleRecord(
                id="r",
                budget=None,
                violations=sum(run.violations for run in runs),
                applicable=sum(run.applicable for run in runs),
                runs=tuple(runs),
            ),
        ),
    )


def split_history():
    return [
        record_at([("r", 9, 40)], when="2026-08-30T10:00:00+00:00"),
        record_at([("r", 10, 40)], when="2026-08-30T11:00:00+00:00"),
    ]


def test_runs_add_up_per_script_in_the_order_they_ran():
    """Repeats of one script are one entry, whatever order they ran."""
    record = split_record(
        (
            scored("cold#1", 3, 10),
            scored("warm#1", 1, 8),
            scored("cold#2", 5, 10),
        )
    ).rule("r")

    assert record.by_script == (
        RunScore(label="cold", violations=8, applicable=20),
        RunScore(label="warm", violations=1, applicable=8),
    )


def test_the_summary_splits_a_rate_by_script():
    """A move confined to one script is visible under the pooled rate."""
    current = split_record(
        (
            scored("cold#1", 8, 10),
            scored("cold#2", 8, 10),
            scored("warm#1", 1, 10),
            scored("warm#2", 1, 10),
        )
    )

    summary = compare_to_history(current, split_history()).summary()

    assert "cold 80%  warm 10%" in summary
    assert "Only the combined rate is judged." in summary


def test_a_rule_nothing_broke_prints_no_split():
    """``cold 0%  warm 0%`` says nothing the pooled 0% did not."""
    current = split_record((scored("cold#1", 0, 10), scored("warm#1", 0, 8)))

    comparison = compare_to_history(current, split_history())

    assert comparison.rules[0].script_split == ""
    assert "Indented lines" not in comparison.summary()


def test_a_rule_only_one_script_exercised_prints_no_split():
    """Its pooled rate already is that script's rate."""
    current = split_record((scored("cold#1", 3, 10), scored("warm#1", 0, 0)))

    comparison = compare_to_history(current, split_history())

    assert comparison.rules[0].script_split == ""
    assert "Indented lines" not in comparison.summary()


def test_a_record_without_labelled_runs_prints_no_split():
    """A line written before labels existed has nothing to split."""
    comparison = compare_to_history(
        record_at([("r", 10, 40)], when="2026-08-30T12:00:00+00:00"),
        split_history(),
    )

    assert comparison.rules[0].by_script == ()
    assert comparison.rules[0].script_split == ""


# ── The margin, once the scripts can be told apart ─────────
#
# Every invocation replays every script, in the same proportion,
# so the difference between two scripts is fixed structure rather
# than something that varies from one invocation to the next.
# Counting it as noise made the margin far wider than the
# instrument's real precision, which is what "the margin does not
# match reality" meant.


def test_the_gap_between_two_scripts_is_not_counted_as_noise():
    """Runs that each agree with their own script are precise."""
    runs = (
        scored("cold#1", 8, 10),
        scored("cold#2", 8, 10),
        scored("warm#1", 2, 10),
        scored("warm#2", 2, 10),
    )
    rates = tuple(r.rate for r in runs)

    labelled = sampling_margin(20, 40, rates, runs)
    unlabelled = sampling_margin(20, 40, rates)

    assert labelled < unlabelled / 2


def test_disagreement_inside_one_script_still_widens_the_margin():
    """Stratifying must not become a way to hide real variance."""
    steady = (scored("cold#1", 5, 10), scored("cold#2", 5, 10))
    scattered = (scored("cold#1", 10, 10), scored("cold#2", 0, 10))

    assert sampling_margin(10, 20, runs=scattered) > sampling_margin(
        10, 20, runs=steady
    )


def test_one_script_of_equal_length_runs_matches_the_flat_estimate():
    """The change is about several scripts, not about one."""
    runs = (scored("s#1", 8, 10), scored("s#2", 2, 10))
    rates = tuple(r.rate for r in runs)

    assert sampling_margin(10, 20, runs=runs) == pytest.approx(
        sampling_margin(10, 20, rates)
    )


def test_one_run_per_script_falls_back_to_the_flat_estimate():
    """Nothing inside a script disagrees, so nothing was measured."""
    runs = (scored("cold#1", 8, 10), scored("warm#1", 2, 10))
    rates = tuple(r.rate for r in runs)

    assert sampling_margin(10, 20, rates, runs) == sampling_margin(
        10, 20, rates
    )


def test_the_binomial_floor_still_applies_when_the_runs_agree():
    """Runs that agree exactly must not buy perfect precision."""
    runs = (scored("s#1", 5, 10), scored("s#2", 5, 10))

    assert sampling_margin(10, 20, runs=runs) == sampling_margin(10, 20)


def test_a_comparison_uses_the_labelled_runs_when_it_has_them():
    """The margin on the page is the stratified one, not the flat one."""
    runs = (
        scored("cold#1", 8, 10),
        scored("cold#2", 8, 10),
        scored("warm#1", 2, 10),
        scored("warm#2", 2, 10),
    )
    record = record_at([("r", 20, 40)])
    record = replace(
        record,
        rules=(
            RuleRecord(
                id="r", budget=None, violations=20, applicable=40, runs=runs
            ),
        ),
    )

    comparison = compare_to_history(record, [record_at([("r", 20, 40)])])

    assert comparison.rules[0].margin == sampling_margin(
        20, 40, tuple(r.rate for r in runs), runs
    )
    assert comparison.rules[0].margin_source is MarginSource.RUNS


# ── A partial invocation says it is one ────────────────────


def test_a_record_counts_the_runs_that_never_finished():
    """A run going missing explains a move as readily as a config edit."""
    report = score_transcripts(contract(), [[("a", "¿y tú?")]])
    report = replace(
        report,
        lost=(LostRun(label="warm#1", error="Connection error.", replies=1),),
    )

    made = record_style_run(report, contract(), make_chatter(), "v1")
    assert made.lost_runs == 1


def test_the_lost_count_survives_a_round_trip(tmp_path):
    report = score_transcripts(contract(), [[("a", "¿y tú?")]])
    report = replace(
        report,
        lost=(LostRun(label="warm#1", error="boom", replies=0),),
    )
    made = record_style_run(report, contract(), make_chatter(), "v1")
    path = append_style_run(made, tmp_path / "h.jsonl")

    assert load_style_history(path)[0].lost_runs == 1


def test_a_record_written_before_lost_runs_existed_reads_zero(tmp_path):
    made = record_at([("r", 1, 2)])
    raw = made.to_dict()
    del raw["lost_runs"]
    path = tmp_path / "h.jsonl"
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    assert load_style_history(path)[0].lost_runs == 0


# ── What each run was shown ────────────────────────────────


def _shown(texts):
    """A two-conversation report, each run shown its own text."""
    report = score_transcripts(
        contract(),
        [[("a", "¿y tú?")], [("a", "pues eso")]],
    )
    return replace(
        report,
        variants={
            f"conversation#{i + 1}": {"dry": text}
            for i, text in enumerate(texts)
        },
    )


def test_the_record_keeps_what_each_run_was_shown():
    made = record_style_run(
        _shown(["x", "y"]), contract(), make_chatter(), "v1"
    )

    assert made.run_variants == {
        "conversation#1": {"dry": "x"},
        "conversation#2": {"dry": "y"},
    }


def test_the_variants_survive_a_round_trip(tmp_path):
    made = record_style_run(
        _shown(["x", "y"]), contract(), make_chatter(), "v1"
    )
    path = append_style_run(made, tmp_path / "h.jsonl")

    assert load_style_history(path)[0].run_variants == made.run_variants


def test_a_record_written_before_variants_existed_reads_empty(tmp_path):
    raw = record_at([("r", 1, 2)]).to_dict()
    del raw["run_variants"]
    path = tmp_path / "h.jsonl"
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    assert load_style_history(path)[0].run_variants == {}


def test_a_rate_splits_by_what_was_shown():
    made = record_style_run(
        _shown(["x", "y"]), contract(), make_chatter(), "v1"
    )

    split = made.by_variant("ends_q", "dry")

    assert split == {
        "x": RunScore(label="x", violations=1, applicable=1),
        "y": RunScore(label="y", violations=0, applicable=1),
    }


def test_runs_shown_the_same_text_add_up():
    made = record_style_run(
        _shown(["x", "x"]), contract(), make_chatter(), "v1"
    )

    assert made.by_variant("ends_q", "dry") == {
        "x": RunScore(label="x", violations=1, applicable=2)
    }


def test_a_split_leaves_out_runs_that_recorded_nothing():
    made = record_style_run(_shown(["x"]), contract(), make_chatter(), "v1")

    assert made.by_variant("ends_q", "dry") == {
        "x": RunScore(label="x", violations=1, applicable=1)
    }
    assert made.by_variant("ends_q", "other") == {}
    assert made.by_variant("missing", "dry") == {}
