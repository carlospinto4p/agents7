"""Unit tests for style contracts (no LLM required)."""

import asyncio

import httpx
import pytest

from agents7.eval.style import (
    ConversationScript,
    ReplyContext,
    StyleContract,
    StyleRule,
    build_rule,
    capitalises_after_stop,
    copies_example,
    ends_with_question,
    first_char_lowercase,
    forbidden_pattern,
    forbidden_text,
    greets_after_opening_with,
    load_style_contract,
    max_emoji,
    max_line_chars,
    max_messages,
    repeats_opener,
    repeats_question_ending,
    reproduces_verbatim,
    run_style_contract,
    run_style_contract_async,
    score_transcript,
    score_transcripts,
)
from agents7.exceptions import AgentConfigurationError, ProviderError
from agents7.models import ChatResponse, LLMMetrics


# --- Helpers ---


def ctx(text, prompt="algo", index=1, previous=None):
    """Build a ReplyContext the way the runner would."""
    return ReplyContext(
        index=index,
        prompt=prompt,
        text=text,
        messages=tuple(
            line.strip() for line in text.splitlines() if line.strip()
        ),
        previous=previous,
    )


def rule(check, budget=0.0, rid="r"):
    return StyleRule(id=rid, description="d", check=check, budget=budget)


class ScriptedChatter:
    """Smallest thing the runner needs: replies from a list.

    Not a `Chatter` — the runner only calls `open_chat`, `chat`
    and `_set_attr`, so a stand-in keeps these tests free of an
    engine and of history bookkeeping that is tested elsewhere.
    """

    def __init__(self, replies):
        self._replies = list(replies)
        self.call_logger = None
        self.seen = []

    def _set_attr(self, name, value):
        setattr(self, name, value)

    def _next(self):
        return ChatResponse(
            response=self._replies.pop(0),
            thinking=None,
            metrics=LLMMetrics(
                tokens=5, prompt_tokens=100, total_duration=0.5
            ),
        )

    def open_chat(self):
        return self._next()

    def chat(self, message):
        self.seen.append(message)
        return self._next()

    async def open_chat_async(self):
        await asyncio.sleep(0)
        return self._next()

    async def chat_async(self, message):
        self.seen.append(message)
        await asyncio.sleep(0)
        return self._next()


class ConcurrencyProbe(ScriptedChatter):
    """Records how many conversations were ever in flight at once.

    The counter is shared across every agent the factory builds,
    because what is being measured is the runner's behaviour, not
    one agent's.
    """

    def __init__(self, replies, state):
        super().__init__(replies)
        self._state = state

    async def chat_async(self, message):
        self._state["live"] += 1
        self._state["peak"] = max(self._state["peak"], self._state["live"])
        try:
            await asyncio.sleep(0.01)
            return super()._next()
        finally:
            self._state["live"] -= 1

    async def open_chat_async(self):
        return super()._next()


class FlakyChatter(ScriptedChatter):
    """Raises when the counterpart says one of *fail_on*.

    Stands in for the real failure: on 30th August 2026 a provider
    closed the connection without finishing a chunked response,
    fifteen minutes into one agent's invocation, and every
    conversation already scored went with it.

    Keyed on the counterpart's line rather than on a turn counter
    so the same agent fails identically whether the runner plays
    the conversations in order or all at once.
    """

    def __init__(self, replies, fail_on=()):
        super().__init__(replies)
        self._fail_on = set(fail_on)

    def chat(self, message):
        if message in self._fail_on:
            raise ConnectionError("Connection error.")
        return super().chat(message)

    async def chat_async(self, message):
        if message in self._fail_on:
            raise ConnectionError("Connection error.")
        return await super().chat_async(message)


# ── Individual checks ──────────────────────────────────────


def test_forbidden_text_is_case_insensitive():
    """A banned word matches however it is capitalised."""
    check = forbidden_text(["joder", "hostia"])
    assert check(ctx("pues Joder, qué frío")) is True
    assert check(ctx("pues vaya, qué frío")) is False


def test_forbidden_pattern_matches_anywhere():
    """The trailing-laugh rule, as its regex."""
    check = forbidden_pattern(r"[.!?]\s+[Jj]aja")
    assert check(ctx("estoy esperando. Jajaja")) is True
    assert check(ctx("jajaja estoy esperando")) is False


def test_forbidden_pattern_can_be_limited_to_one_kind_of_prompt():
    """Off the prompts it is about, the rule does not apply."""
    check = forbidden_pattern("(?i)secreto", when_prompt_matches="nombre")
    asked = "y ese nombre de dónde sale?"
    assert check(ctx("Secreto, sin más", prompt=asked)) is True
    assert check(ctx("me lo puso mi abuela", prompt=asked)) is False
    assert check(ctx("un secreto", prompt="qué es esto?")) is None
    assert check(ctx("Secreto", prompt=None)) is None


def test_max_messages_counts_messages_not_lines():
    """Blank lines between messages do not count as messages."""
    check = max_messages(1)
    assert check(ctx("una línea")) is False
    assert check(ctx("una línea\n\notra línea")) is True


def test_max_messages_only_applies_to_a_short_prompt():
    """The rule is about answering a monosyllable, nothing else.

    Without the guard it would be diluted by every turn it was
    never about, and its rate would mean nothing.
    """
    check = max_messages(1, when_prompt_words_at_most=3)
    assert check(ctx("a\n\nb", prompt="no")) is True
    assert (
        check(ctx("a\n\nb", prompt="pues mira, hoy ha sido larguísimo"))
        is None
    )


def test_max_messages_skips_the_opening_turn():
    """An opening turn has no prompt to measure against."""
    check = max_messages(1, when_prompt_words_at_most=3)
    assert check(ctx("hola", prompt=None, index=0)) is None


def test_max_line_chars_flags_the_longest_message():
    check = max_line_chars(20)
    assert check(ctx("corto")) is False
    assert check(ctx("x" * 21)) is True


def test_ends_with_question_reads_the_last_message():
    """A question mid-reply does not count; the ending does."""
    check = ends_with_question()
    assert check(ctx("¿y tú?\n\npues eso")) is False
    assert check(ctx("pues eso\n\n¿y tú?")) is True


def test_repeats_question_ending_needs_a_previous_reply():
    """The first reply cannot repeat anything."""
    check = repeats_question_ending()
    first = ctx("¿y tú?", index=0)
    assert check(first) is None
    assert check(ctx("¿y qué más?", previous=first)) is True


def test_repeats_question_ending_is_clean_when_one_stops():
    check = repeats_question_ending()
    first = ctx("¿y tú?", index=0)
    assert check(ctx("pues te pega", previous=first)) is False


def test_first_char_lowercase_skips_spanish_punctuation():
    """`¿` must not make every reply pass by accident."""
    check = first_char_lowercase()
    assert check(ctx("¿Y tú qué?")) is True
    assert check(ctx("¿y tú qué?")) is False


def test_first_char_lowercase_honours_exemptions():
    """The persona allows a standalone greeting in upper case."""
    check = first_char_lowercase(exempt_prefixes=["¡buenas", "toc toc"])
    assert check(ctx("¡Buenas! ¿Qué tal?")) is False
    assert check(ctx("Vale, acepto encantado")) is True


def test_first_char_lowercase_ignores_a_message_with_no_letter():
    check = first_char_lowercase()
    assert check(ctx("...")) is None


def test_max_emoji_counts_per_message():
    check = max_emoji(1, [":)", ";)"])
    assert check(ctx("bien :)\n\notra :)")) is False
    assert check(ctx("bien :) :)")) is True


def test_greets_after_opening_allows_the_opening_turn():
    """Opening with a greeting is the point of an opening turn."""
    check = greets_after_opening_with(["¡buenas", "hola"])
    assert check(ctx("¡Buenas! ¿Qué tal?", prompt=None, index=0)) is None
    assert check(ctx("¡Buenas! ¿Qué tal?", index=7)) is True
    assert check(ctx("adiós", index=7)) is False


def test_reproduces_verbatim_ignores_case_and_final_stop():
    """A copied line with the full stop dropped is still copied."""
    check = reproduces_verbatim(["vale, acepto encantado."])
    assert check(ctx("Vale, acepto encantado")) is True
    assert check(ctx("vale, me vale")) is False


# The lines below are the kind a persona's examples hold, and the
# kind a model writes back from them.


def test_copies_example_catches_a_line_with_a_word_changed():
    check = copies_example(
        ["¿y para qué estás aquí? Que ahora me pica la curiosidad"]
    )
    reply = "¿y para qué has venido? Que ahora me pica la curiosidad"
    assert check(ctx(reply)) is True


def test_copies_example_catches_a_line_with_something_appended():
    """A whole example inside a longer message is still a copy."""
    check = copies_example(["¿dejas el móvil boca arriba o boca abajo?"])
    reply = (
        "¿dejas el móvil boca arriba o boca abajo ya te lo he "
        "preguntado, eh? no me repitas trucos"
    )
    assert check(ctx(reply)) is True


def test_copies_example_leaves_a_line_in_the_same_voice_alone():
    """Same shape, his own words: what the examples are for."""
    check = copies_example(["uy, no es lo mío. Pero pasadlo bien"])
    assert check(ctx("ah pues nada, no es lo mío. ¡Suerte!")) is False


def test_copies_example_checks_every_message_in_a_reply():
    check = copies_example(["algo harás, aunque sea mirar el techo"])
    assert check(ctx("jajaja\nalgo harás, aunque sea mirar el techo")) is True


def test_copies_example_ignores_short_stock_lines():
    """A person says "recibido" the same way every time."""
    check = copies_example(["me has pillado", "recibido"])
    assert check(ctx("me has pillado")) is None
    assert check(ctx("recibido\nbueno, cuéntame qué tal el día")) is False


def test_copies_example_rejects_thresholds_out_of_range():
    with pytest.raises(AgentConfigurationError):
        copies_example(["x"], min_similarity=0)
    with pytest.raises(AgentConfigurationError):
        copies_example(["x"], min_chars=0)


# ── Openers, compared across conversations ─────────────────
#
# One agent opened every conversation of two reads with one line
# while told to vary it. No per-reply check can see that, so this
# one is handed the whole invocation.


def _opener_contract():
    return StyleContract(
        rules=(rule(repeats_opener(), budget=None),),
        scripts=(ConversationScript(id="s", lines=("a",)),),
    )


def test_repeats_opener_flags_every_conversation_sharing_a_line():
    """Casing and end punctuation do not make a line different."""
    report = score_transcripts(
        _opener_contract(),
        [
            [(None, "Toc toc.")],
            [(None, "toc toc")],
            [(None, "¡Buenas! ¿Qué tal?")],
        ],
    )
    result = report.results[0]
    assert (result.violations, result.applicable) == (2, 3)


def test_repeats_opener_passes_varied_openers():
    report = score_transcripts(
        _opener_contract(),
        [[(None, "Toc toc.")], [(None, "Hm... Probando...")]],
    )
    result = report.results[0]
    assert (result.violations, result.applicable) == (0, 2)


def test_repeats_opener_judges_only_the_first_reply():
    """A later line two conversations share is not an opener."""
    report = score_transcripts(
        _opener_contract(),
        [
            [(None, "Toc toc."), ("hola", "igual")],
            [(None, "¡Hola!"), ("hola", "igual")],
        ],
    )
    result = report.results[0]
    assert (result.violations, result.applicable) == (0, 2)


def test_repeats_opener_has_nothing_to_compare_in_one_conversation():
    report = score_transcript(_opener_contract(), [(None, "Toc toc.")])
    result = report.results[0]
    assert (result.violations, result.applicable) == (0, 1)


def test_repeats_opener_refuses_to_judge_a_single_reply():
    with pytest.raises(AgentConfigurationError):
        repeats_opener()(ctx("Toc toc.", prompt=None, index=0))


def test_repeats_opener_is_declarable_and_scored_by_the_runner():
    built = build_rule(
        {
            "id": "r",
            "description": "d",
            "check": "repeats_opener",
            "budget": None,
        }
    )
    contract = StyleContract(
        rules=(built,),
        scripts=(ConversationScript(id="s", lines=("a",), opens=True),),
        repeats=3,
    )
    report = run_style_contract(
        lambda: ScriptedChatter(["Toc toc.", "vale"]), contract
    )
    result = report.results[0]
    assert (result.violations, result.applicable) == (3, 3)


def test_a_declared_rule_carries_the_definition_of_its_check():
    """What a rule measures, so its history is kept apart by it."""
    spec = {
        "id": "r",
        "description": "d",
        "check": "copies_example",
        "args": {"lines": ["a"], "min_chars": 1},
    }
    first = build_rule(spec)

    reordered = dict(spec, args={"min_chars": 1, "lines": ["a"]})
    assert build_rule(reordered).definition == first.definition
    assert build_rule(dict(spec, budget=0.5)).definition == first.definition
    changed = dict(spec, args={"lines": ["b"], "min_chars": 1})
    assert build_rule(changed).definition != first.definition


def test_a_rule_built_around_a_callable_has_no_definition():
    assert rule(ends_with_question()).definition is None


# ── Scoring ────────────────────────────────────────────────


def test_rate_divides_by_applicable_not_by_replies():
    """A rule that skipped a turn is not diluted by it."""
    contract = StyleContract(
        rules=(rule(max_messages(1, when_prompt_words_at_most=3)),),
        scripts=(ConversationScript(id="s", lines=("no",)),),
    )
    report = score_transcript(
        contract,
        [("no", "a\n\nb"), ("una frase entera y larga de verdad", "c\n\nd")],
    )
    result = report.results[0]
    assert (result.violations, result.applicable) == (1, 1)
    assert result.rate == 1.0


def test_a_budget_lets_a_tendency_pass():
    """A soft rule tolerates the rate it declares."""
    contract = StyleContract(
        rules=(rule(ends_with_question(), budget=0.5),),
        scripts=(ConversationScript(id="s", lines=("a",)),),
    )
    report = score_transcript(contract, [("a", "¿y tú?"), ("b", "pues eso")])
    assert report.results[0].rate == 0.5
    assert report.passed


def test_a_budget_still_fails_when_exceeded():
    contract = StyleContract(
        rules=(rule(ends_with_question(), budget=0.4),),
        scripts=(ConversationScript(id="s", lines=("a",)),),
    )
    report = score_transcript(contract, [("a", "¿y tú?"), ("b", "pues eso")])
    assert not report.passed
    assert report.failures[0].rule.id == "r"


def test_a_measure_only_rule_never_fails():
    """`budget: None` is a number to watch, not a threshold."""
    contract = StyleContract(
        rules=(rule(ends_with_question(), budget=None),),
        scripts=(ConversationScript(id="s", lines=("a",)),),
    )
    report = score_transcript(contract, [("a", "¿y tú?")])
    assert report.results[0].rate == 1.0
    assert report.passed


def test_an_unexercised_rule_is_flagged_not_passed_silently():
    """A script that never triggers a rule is a gap in the script.

    Reporting it as a pass would let a contract go green by
    testing nothing, which is the failure mode this whole module
    exists to prevent.
    """
    contract = StyleContract(
        rules=(rule(max_messages(1, when_prompt_words_at_most=1)),),
        scripts=(ConversationScript(id="s", lines=("x",)),),
    )
    report = score_transcript(
        contract, [("una frase larga de verdad aquí", "a\n\nb")]
    )
    result = report.results[0]
    assert result.unexercised
    assert result.rate is None
    assert "??" in report.summary()


def test_report_looks_rules_up_by_id():
    contract = StyleContract(
        rules=(rule(ends_with_question(), budget=None, rid="ends_q"),),
        scripts=(ConversationScript(id="s", lines=("a",)),),
    )
    report = score_transcript(contract, [("a", "¿y tú?")])
    assert report.rate("ends_q") == 1.0
    with pytest.raises(KeyError):
        report.rate("nope")


def test_examples_carry_the_offending_reply():
    """A rate says how often; an example says what broke."""
    contract = StyleContract(
        rules=(rule(forbidden_text(["joder"])),),
        scripts=(ConversationScript(id="s", lines=("a",)),),
    )
    report = score_transcript(contract, [("a", "joder qué frío")])
    assert report.results[0].examples == ("joder qué frío",)


def test_score_transcript_reports_no_cost():
    """Scoring a past conversation spends nothing.

    None rather than 0.0 — an unpriced run must never be
    reported as a free one.
    """
    contract = StyleContract(
        rules=(rule(ends_with_question(), budget=None),),
        scripts=(ConversationScript(id="s", lines=("a",)),),
    )
    report = score_transcript(contract, [("a", "¿y tú?")])
    assert report.cost_cents is None
    assert report.calls == ()


# ── Running against an agent ───────────────────────────────


def test_run_plays_the_script_in_order():
    agent = ScriptedChatter(["uno", "dos", "tres"])
    contract = StyleContract(
        rules=(rule(ends_with_question(), budget=None),),
        scripts=(ConversationScript(id="s", lines=("a", "b", "c")),),
        repeats=1,
    )

    report = run_style_contract(lambda: agent, contract)

    assert agent.seen == ["a", "b", "c"]
    assert [r.text for r in report.replies] == ["uno", "dos", "tres"]


def test_run_uses_the_opening_turn_when_the_script_opens():
    agent = ScriptedChatter(["¡Buenas!", "uno"])
    contract = StyleContract(
        rules=(rule(ends_with_question(), budget=None),),
        scripts=(ConversationScript(id="s", lines=("a",), opens=True),),
        repeats=1,
    )

    report = run_style_contract(lambda: agent, contract)

    assert report.replies[0].prompt is None
    assert report.replies[0].text == "¡Buenas!"


class ContextChatter(ScriptedChatter):
    """Records what `set_context` was given, and when."""

    def __init__(self, replies):
        super().__init__(replies)
        self.context = {}
        self.context_set_after = None

    def set_context(self, **values):
        self.context.update(values)
        self.context_set_after = len(self._replies)
        return self


@pytest.mark.parametrize("opens", [True, False])
def test_run_sets_the_script_context_before_the_first_turn(opens):
    agent = ContextChatter(["uno", "dos"])
    script = ConversationScript(
        id="s", lines=("a",), opens=opens, context={"nick": "Marta_87"}
    )
    contract = StyleContract(
        rules=(rule(ends_with_question(), budget=None),),
        scripts=(script,),
        repeats=1,
    )

    run_style_contract(lambda: agent, contract)

    assert agent.context == {"nick": "Marta_87"}
    assert agent.context_set_after == 2


def test_async_run_sets_the_script_context_before_the_first_turn():
    agent = ContextChatter(["uno", "dos"])
    script = ConversationScript(
        id="s", lines=("a",), opens=True, context={"nick": "Marta_87"}
    )
    contract = StyleContract(
        rules=(rule(ends_with_question(), budget=None),),
        scripts=(script,),
        repeats=1,
    )

    asyncio.run(run_style_contract_async(lambda: agent, contract))

    assert agent.context == {"nick": "Marta_87"}
    assert agent.context_set_after == 2


def test_a_script_without_context_never_calls_set_context():
    """`ScriptedChatter` has no `set_context`, so a call would raise."""
    agent = ScriptedChatter(["uno"])
    contract = StyleContract(
        rules=(rule(ends_with_question(), budget=None),),
        scripts=(ConversationScript(id="s", lines=("a",)),),
        repeats=1,
    )

    report = run_style_contract(lambda: agent, contract)

    assert report.complete


def test_load_reads_a_script_context():
    contract = load_style_contract(
        {
            "rules": [
                {"id": "q", "description": "d", "check": "ends_with_question"}
            ],
            "scripts": [
                {"id": "s", "lines": ["a"], "context": {"nick": "Marta_87"}},
                {"id": "t", "lines": ["a"]},
            ],
        }
    )

    assert contract.scripts[0].context == {"nick": "Marta_87"}
    assert contract.scripts[1].context == {}


def test_run_builds_a_fresh_agent_per_repeat():
    """Each run starts an empty conversation, not a continued one."""
    built = []

    def factory():
        agent = ScriptedChatter(["uno", "dos"])
        built.append(agent)
        return agent

    contract = StyleContract(
        rules=(rule(ends_with_question(), budget=None),),
        scripts=(ConversationScript(id="s", lines=("a", "b")),),
        repeats=3,
    )

    report = run_style_contract(factory, contract)

    assert len(built) == 3
    assert len(report.replies) == 6


def test_run_collects_every_llm_call():
    """The report can price a run, guard calls included."""
    agent = ScriptedChatter(["uno", "dos"])
    contract = StyleContract(
        rules=(rule(ends_with_question(), budget=None),),
        scripts=(ConversationScript(id="s", lines=("a", "b")),),
        repeats=1,
    )

    def fake_turn(message):
        # Two records per turn, as a guarded chatter emits.
        for _ in range(2):
            agent.call_logger(_record(cost_cents=0.25, total_duration=1.0))
        return ScriptedChatter.chat(agent, message)

    agent.chat = fake_turn
    report = run_style_contract(lambda: agent, contract)

    assert len(report.calls) == 4
    assert report.cost_cents == 1.0
    assert report.duration == 4.0


def test_run_keeps_an_existing_call_logger():
    """Wiring the contract's collector must not silence the app's."""
    seen = []
    agent = ScriptedChatter(["uno"])
    agent.call_logger = seen.append
    contract = StyleContract(
        rules=(rule(ends_with_question(), budget=None),),
        scripts=(ConversationScript(id="s", lines=("a",)),),
        repeats=1,
    )

    def fake_turn(message):
        agent.call_logger(_record())
        return ScriptedChatter.chat(agent, message)

    agent.chat = fake_turn
    report = run_style_contract(lambda: agent, contract)

    assert len(seen) == 1
    assert len(report.calls) == 1


def _record(**kwargs):
    from agents7.call_log import CallRecord

    defaults = {
        "trace_id": "t",
        "agent_name": "a",
        "agent_version": 1,
        "model_id": "ollama::x",
        "success": True,
    }
    defaults.update(kwargs)
    return CallRecord(**defaults)


# ── Declaring a contract ───────────────────────────────────


def test_build_rule_from_a_declared_spec():
    built = build_rule(
        {
            "id": "no_tacos",
            "description": "Nunca dice tacos.",
            "check": "forbidden_text",
            "args": {"values": ["joder"]},
        }
    )
    assert built.id == "no_tacos"
    assert built.check(ctx("joder")) is True


def test_build_rule_rejects_an_unknown_check():
    """A dropped rule would report a clean run it never made."""
    with pytest.raises(AgentConfigurationError, match="unknown check"):
        build_rule({"id": "x", "description": "d", "check": "nope"})


def test_build_rule_rejects_args_the_check_cannot_take():
    with pytest.raises(AgentConfigurationError, match="rejected"):
        build_rule(
            {
                "id": "x",
                "description": "d",
                "check": "max_line_chars",
                "args": {"nonsense": 1},
            }
        )


def test_build_rule_requires_a_description():
    with pytest.raises(AgentConfigurationError, match="description"):
        build_rule({"id": "x", "check": "ends_with_question"})


def test_a_rule_budget_must_be_a_rate():
    with pytest.raises(AgentConfigurationError, match="between"):
        StyleRule(id="x", description="d", check=lambda c: False, budget=2.0)


def test_rule_ids_must_be_unique():
    """Two rules with one id would overwrite each other's result."""
    with pytest.raises(AgentConfigurationError, match="unique"):
        StyleContract(
            rules=(
                rule(ends_with_question(), rid="same"),
                rule(max_line_chars(10), rid="same"),
            ),
            scripts=(ConversationScript(id="s", lines=("a",)),),
        )


def test_repeats_must_be_at_least_one():
    with pytest.raises(AgentConfigurationError, match="at least 1"):
        StyleContract(
            rules=(rule(ends_with_question()),),
            scripts=(ConversationScript(id="s", lines=("a",)),),
            repeats=0,
        )


def test_load_style_contract_from_yaml(tmp_path):
    path = tmp_path / "contract.yml"
    path.write_text(
        "repeats: 2\n"
        "rules:\n"
        "  - id: one_message_when_dry\n"
        "    description: Una línea, no dos.\n"
        "    check: max_messages\n"
        "    args:\n"
        "      limit: 1\n"
        "      when_prompt_words_at_most: 3\n"
        "  - id: ends_on_a_question\n"
        "    description: Preguntar es una jugada.\n"
        "    check: ends_with_question\n"
        "    budget: 0.5\n"
        "scripts:\n"
        "  - id: cold\n"
        "    opens: true\n"
        "    lines: ['bien', 'normal']\n",
        encoding="utf-8",
    )

    contract = load_style_contract(str(path))

    assert contract.repeats == 2
    assert [r.id for r in contract.rules] == [
        "one_message_when_dry",
        "ends_on_a_question",
    ]
    assert contract.rules[0].budget == 0.0
    assert contract.rules[1].budget == 0.5
    assert contract.scripts[0].opens is True
    assert contract.scripts[0].lines == ("bien", "normal")


def test_load_style_contract_needs_rules(tmp_path):
    path = tmp_path / "empty.yml"
    path.write_text("scripts: [{id: s, lines: [a]}]\n", encoding="utf-8")
    with pytest.raises(AgentConfigurationError, match="no rules"):
        load_style_contract(str(path))


def test_load_style_contract_needs_scripts():
    """Rules with nothing to run against would score nothing."""
    with pytest.raises(AgentConfigurationError, match="no scripts"):
        load_style_contract(
            {
                "rules": [
                    {
                        "id": "x",
                        "description": "d",
                        "check": "ends_with_question",
                    }
                ]
            }
        )


def test_score_transcripts_keeps_conversations_apart():
    """Each conversation is numbered and judged from its own start.

    Pooling them into one flat list would read the second one's
    legitimate opening greeting as a greeting mid-conversation,
    and its first reply as following the last reply of the first.
    """
    contract = StyleContract(
        rules=(
            rule(greets_after_opening_with(["hola"]), rid="greets"),
            rule(repeats_question_ending(), budget=None, rid="repeat_q"),
        ),
        scripts=(ConversationScript(id="s", lines=("a",)),),
    )

    # The first ends on a question; the second opens with a
    # greeting that is also a question. Flattened, that second
    # opening is adjacent to the first conversation's last reply.
    conversations = [
        [(None, "buenas"), ("a", "¿y tú?")],
        [(None, "hola, ¿qué tal?"), ("b", "pues eso")],
    ]

    pooled = score_transcript(contract, [t for c in conversations for t in c])
    apart = score_transcripts(contract, conversations)

    # Flattened, the second "hola" looks like a mid-chat greeting
    # and the second "¿y tú?" looks like a repeat.
    assert pooled.results[0].violations == 1
    assert pooled.results[1].violations == 1
    assert apart.results[0].violations == 0
    assert apart.results[1].violations == 0


def test_score_transcript_is_one_conversation():
    """The singular form is the plural form with one entry."""
    contract = StyleContract(
        rules=(rule(greets_after_opening_with(["hola"]), rid="greets"),),
        scripts=(ConversationScript(id="s", lines=("a",)),),
    )
    turns = [(None, "hola"), ("a", "hola otra vez")]
    assert (
        score_transcript(contract, turns).results[0].violations
        == score_transcripts(contract, [turns]).results[0].violations
        == 1
    )


# ── Spread across runs ─────────────────────────────────────


def test_per_run_scores_keep_each_run_separate():
    """The pooled rate cannot say whether the runs agreed."""
    contract = StyleContract(
        rules=(rule(forbidden_text(["mal"]), budget=None),),
        scripts=(ConversationScript(id="s", lines=("a",)),),
    )
    report = score_transcripts(
        contract,
        [
            [("a", "mal"), ("b", "mal")],
            [("a", "bien"), ("b", "bien")],
        ],
    )
    result = report.results[0]
    assert result.rate == 0.5
    assert result.run_rates == (1.0, 0.0)
    assert result.spread == (0.0, 1.0)


def test_a_wide_spread_is_flagged_volatile():
    """One bad run and one clean one is not a 50% baseline.

    The whole reason `per_run` exists: on one agent one repeat
    scored a rule at 81% and three scored it at 23%, and pooling
    could not tell those apart.
    """
    contract = StyleContract(
        rules=(rule(forbidden_text(["mal"]), budget=None),),
        scripts=(ConversationScript(id="s", lines=("a",)),),
    )
    report = score_transcripts(
        contract,
        [[("a", "mal"), ("b", "mal")], [("a", "bien"), ("b", "bien")]],
    )
    assert report.results[0].volatile
    assert report.volatile
    assert "!" in report.summary()
    assert "not a baseline" in report.summary()


def test_agreeing_runs_are_not_volatile():
    contract = StyleContract(
        rules=(rule(forbidden_text(["mal"]), budget=None),),
        scripts=(ConversationScript(id="s", lines=("a",)),),
    )
    report = score_transcripts(
        contract,
        [[("a", "mal"), ("b", "bien")], [("a", "mal"), ("b", "bien")]],
    )
    result = report.results[0]
    assert result.spread == (0.5, 0.5)
    assert not result.volatile
    assert not report.volatile


def test_spread_needs_two_exercising_runs():
    """One run has no spread, and implying one would be a lie."""
    contract = StyleContract(
        rules=(rule(forbidden_text(["mal"]), budget=None),),
        scripts=(ConversationScript(id="s", lines=("a",)),),
    )
    report = score_transcripts(contract, [[("a", "mal")]])
    assert report.results[0].spread is None
    assert not report.results[0].volatile


def test_a_run_that_never_triggered_a_rule_is_left_out_of_the_spread():
    """An unexercised run is not a 0% run.

    Counting it as zero would drag the range down and make a rule
    look steadier than the runs that actually tested it.
    """
    contract = StyleContract(
        rules=(
            rule(
                max_messages(1, when_prompt_words_at_most=2),
                budget=None,
            ),
        ),
        scripts=(ConversationScript(id="s", lines=("a",)),),
    )
    report = score_transcripts(
        contract,
        [
            [("no", "a\n\nb")],
            [("una frase bastante larga de verdad", "c\n\nd")],
        ],
    )
    result = report.results[0]
    assert [r.applicable for r in result.per_run] == [1, 0]
    assert result.run_rates == (1.0,)
    assert result.spread is None


def test_pooled_totals_are_unchanged_by_the_split():
    """Scoring per run then summing equals scoring the pool."""
    contract = StyleContract(
        rules=(rule(ends_with_question(), budget=None),),
        scripts=(ConversationScript(id="s", lines=("a",)),),
    )
    conversations = [
        [("a", "¿y tú?"), ("b", "pues eso")],
        [("a", "¿y qué más?"), ("b", "¿seguro?")],
    ]
    report = score_transcripts(contract, conversations)
    result = report.results[0]
    assert (result.violations, result.applicable) == (3, 4)
    assert sum(r.violations for r in result.per_run) == result.violations


def test_run_labels_name_the_script_and_repeat():
    """A volatile rule is useless without knowing which run."""
    agent_replies = ["uno", "dos"]
    contract = StyleContract(
        rules=(rule(ends_with_question(), budget=None),),
        scripts=(ConversationScript(id="cold", lines=("a",)),),
        repeats=2,
    )
    report = run_style_contract(
        lambda: ScriptedChatter(list(agent_replies)), contract
    )
    assert [r.label for r in report.results[0].per_run] == [
        "cold#1",
        "cold#2",
    ]


def test_summary_keeps_a_fractional_budget_distinguishable():
    """A budget must not print as a different budget.

    One agent sets two rules at 0.5% — one stray event's worth of
    an invocation — and whole-percent rounding printed those as
    `<=0%`, which is exactly what a rule tolerating nothing
    prints. Two thresholds a whole event apart read as the same
    number, so a reader could not tell which one a run cleared.
    """
    contract = StyleContract(
        rules=(
            rule(ends_with_question(), budget=0.005, rid="stray"),
            rule(ends_with_question(), budget=0.0, rid="never"),
        ),
        scripts=(ConversationScript(id="s", lines=("a",)),),
    )
    summary = score_transcripts(contract, [[("a", "pues eso")]]).summary()

    assert "<=0.5%" in summary
    assert "<=0%" in summary


def test_summary_reports_the_run_count():
    contract = StyleContract(
        rules=(rule(ends_with_question(), budget=None),),
        scripts=(ConversationScript(id="s", lines=("a",)),),
    )
    report = score_transcripts(
        contract, [[("a", "¿y tú?")], [("a", "pues eso")]]
    )
    assert "over 2 runs" in report.summary()


def test_a_run_says_which_script_it_played():
    """The label carries the script; reading it is not the caller's job.

    Anything estimating how much a rule varies has to keep two
    scripts apart, because they are two different stimuli with two
    different true rates.
    """
    report = run_style_contract(
        lambda: ScriptedChatter(["uno?", "dos"]), _pair_contract(repeats=2)
    )

    assert [s.script for s in report.results[0].per_run] == [
        "cold",
        "warm",
        "cold",
        "warm",
    ]


def test_a_label_with_no_repeat_marker_is_its_own_script():
    """`score_transcripts` labels conversations, not repeats."""
    report = score_transcripts(
        StyleContract(
            rules=(rule(ends_with_question(), budget=None),),
            scripts=(ConversationScript(id="s", lines=("a",)),),
        ),
        [[("a", "uno?")]],
    )

    assert report.results[0].per_run[0].script == "conversation"


# ── Running conversations concurrently ─────────────────────
#
# The conversations were always independent — each gets its own
# agent and shares no state — so running them serially cost wall
# clock and bought nothing. What these guard is that concurrency
# changes only the clock: the same runs, the same labels, the
# same scores, and turns still ordered inside a conversation.


def _pair_contract(repeats=3):
    return StyleContract(
        rules=(rule(ends_with_question(), budget=None),),
        scripts=(
            ConversationScript(id="cold", lines=("a", "b")),
            ConversationScript(id="warm", lines=("c", "d")),
        ),
        repeats=repeats,
    )


async def test_concurrent_and_sequential_agree_exactly():
    """The whole claim: same measurement, less wall clock."""
    contract = _pair_contract()

    def factory():
        return ScriptedChatter(["uno?", "dos"] * 2)

    serial = run_style_contract(factory, contract)
    parallel = await run_style_contract_async(factory, contract)

    assert [r.text for r in serial.replies] == [
        r.text for r in parallel.replies
    ]
    assert serial.results[0].violations == parallel.results[0].violations
    assert serial.results[0].applicable == parallel.results[0].applicable
    assert [s.label for s in serial.results[0].per_run] == [
        s.label for s in parallel.results[0].per_run
    ]


async def test_turns_stay_ordered_inside_a_conversation():
    """Only whole conversations overlap, never turns within one."""
    agent = ScriptedChatter(["uno", "dos", "tres"])
    contract = StyleContract(
        rules=(rule(ends_with_question(), budget=None),),
        scripts=(ConversationScript(id="s", lines=("a", "b", "c")),),
        repeats=1,
    )

    await run_style_contract_async(lambda: agent, contract)

    assert agent.seen == ["a", "b", "c"]


async def test_conversations_really_do_overlap():
    """Without this the async path could be serial and still pass."""
    state = {"live": 0, "peak": 0}
    contract = _pair_contract()

    await run_style_contract_async(
        lambda: ConcurrencyProbe(["uno", "dos"] * 2, state), contract
    )

    assert state["peak"] > 1


async def test_concurrency_is_capped_at_what_was_asked_for():
    """The provider's rate limit is the reason this is a knob."""
    state = {"live": 0, "peak": 0}

    await run_style_contract_async(
        lambda: ConcurrencyProbe(["uno", "dos"] * 2, state),
        _pair_contract(repeats=5),
        concurrency=2,
    )

    assert state["peak"] == 2


async def test_every_call_is_captured_across_conversations():
    """Interleaved runs must not lose a call from the log."""
    contract = _pair_contract()

    report = await run_style_contract_async(
        lambda: ScriptedChatter(["uno", "dos"] * 2), contract
    )

    assert len(report.replies) == 12


class ClosingChatter(ScriptedChatter):
    """Records whether the runner closed it, and on which loop."""

    def __init__(self, replies, closed):
        super().__init__(replies)
        self._closed = closed

    async def aclose(self):
        self._closed.append(asyncio.get_running_loop())


async def test_every_built_agent_is_closed_before_returning():
    """Left open, their clients close on a loop already shut."""
    closed = []

    await run_style_contract_async(
        lambda: ClosingChatter(["uno", "dos"] * 2, closed), _pair_contract()
    )

    assert closed == [asyncio.get_running_loop()] * 6


async def test_an_agent_returned_twice_is_closed_once():
    closed = []
    agent = ClosingChatter(["uno"] * 4, closed)
    contract = StyleContract(
        rules=(rule(ends_with_question(), budget=None),),
        scripts=(ConversationScript(id="s", lines=("a",)),),
        repeats=4,
    )

    await run_style_contract_async(lambda: agent, contract)

    assert len(closed) == 1


async def test_concurrency_below_one_is_refused():
    with pytest.raises(AgentConfigurationError, match="at least 1"):
        await run_style_contract_async(
            lambda: ScriptedChatter(["uno"]), _pair_contract(), concurrency=0
        )


# ── A conversation that dies takes only itself ─────────────
#
# One dropped connection used to end the whole invocation:
# `asyncio.gather` cancelled every sibling still in flight and the
# conversations already finished were discarded unscored. Measured
# on one agent, that was about seventeen minutes and seventeen
# cents lost to one failed turn, with nothing recorded.


def _flaky_factory(fail_on):
    return lambda: FlakyChatter(["uno?", "dos"], fail_on=fail_on)


def test_a_failed_conversation_does_not_lose_the_others():
    """The whole point: a partial invocation beats no invocation."""
    report = run_style_contract(
        _flaky_factory({"d"}), _pair_contract(repeats=2)
    )

    assert [r.label for r in report.results[0].per_run] == [
        "cold#1",
        "cold#2",
    ]
    assert [r.label for r in report.lost] == ["warm#1", "warm#2"]


async def test_the_async_runner_loses_only_the_failed_conversation():
    """gather must not cancel the siblings still in flight."""
    report = await run_style_contract_async(
        _flaky_factory({"d"}), _pair_contract(repeats=2)
    )

    assert [r.label for r in report.results[0].per_run] == [
        "cold#1",
        "cold#2",
    ]
    assert [r.label for r in report.lost] == ["warm#1", "warm#2"]


def test_a_lost_run_says_what_it_managed_and_why_it_stopped():
    """A partial invocation has to be readable as one."""
    report = run_style_contract(
        _flaky_factory({"d"}), _pair_contract(repeats=1)
    )

    lost = report.lost[0]
    assert lost.label == "warm#1"
    assert lost.replies == 1
    assert "Connection error." in lost.error
    assert report.complete is False


def test_the_replies_of_a_lost_run_are_not_scored():
    """A prefix of a script is a different stimulus, not a run.

    Folding the turns a dying conversation managed into a band
    built from whole conversations compares two different things.
    """
    report = run_style_contract(
        _flaky_factory({"d"}), _pair_contract(repeats=1)
    )

    # cold plays two lines; warm produced one reply before dying.
    assert len(report.replies) == 2
    assert len(report.results[0].per_run) == 1


# ── Stopping when the provider stops answering ─────────────
#
# On 14th September 2026 DeepSeek accepted requests and never
# answered them, so every conversation spent its whole retry
# budget before failing, one after another, and nothing could
# have been measured.


def _unavailable() -> ProviderError:
    """What the engine raises once a timeout has used every retry."""
    error = ProviderError(
        "All 2 retries exhausted. Last error: Request timed out."
    )
    error.__cause__ = httpx.TimeoutException("Request timed out.")
    return error


class UnavailableChatter(ScriptedChatter):
    """A provider that accepts every request and answers none."""

    def __init__(self, delay=0.0):
        super().__init__([])
        self._delay = delay

    def chat(self, message):
        raise _unavailable()

    async def chat_async(self, message):
        await asyncio.sleep(self._delay)
        raise _unavailable()


class SlowChatter(ScriptedChatter):
    """Answers, but slowly enough to still be mid-conversation."""

    async def chat_async(self, message):
        self.seen.append(message)
        await asyncio.sleep(0.05)
        return self._next()


def _sequence_factory(*kinds):
    """One agent per conversation, in the order they are built."""
    queue = list(kinds)
    built = {
        "ok": lambda: ScriptedChatter(["uno?", "dos"]),
        "slow": lambda: SlowChatter(["uno?", "dos"]),
        "down": UnavailableChatter,
        "down_later": lambda: UnavailableChatter(delay=0.01),
        "broken": lambda: FlakyChatter(["uno?", "dos"], fail_on={"a", "c"}),
    }
    made = []

    def factory():
        agent = built[queue.pop(0)]()
        made.append(agent)
        return agent

    factory.made = made
    return factory


def test_the_runner_stops_once_the_provider_stops_answering():
    """Two conversations in a row not answered: play no more."""
    factory = _sequence_factory("ok", "down", "down", "ok", "ok", "ok")

    report = run_style_contract(factory, _pair_contract(repeats=3))

    assert len(factory.made) == 3
    assert [r.label for r in report.lost] == [
        "warm#1",
        "cold#2",
        "warm#2",
        "cold#3",
        "warm#3",
    ]
    assert report.stopped_early is True
    assert "not played" in report.lost[-1].error


def test_a_finished_conversation_resets_the_count():
    factory = _sequence_factory("down", "ok", "down", "ok", "down", "ok")

    report = run_style_contract(factory, _pair_contract(repeats=3))

    assert (len(factory.made), report.stopped_early) == (6, False)


def test_other_errors_never_stop_the_run():
    """Only a provider that is not answering stops it."""
    factory = _sequence_factory("ok", *["broken"] * 5)

    report = run_style_contract(factory, _pair_contract(repeats=3))

    assert (len(factory.made), report.stopped_early) == (6, False)


def test_stopping_can_be_turned_off():
    factory = _sequence_factory("ok", *["down"] * 5)

    report = run_style_contract(
        factory, _pair_contract(repeats=3), stop_after_unavailable=None
    )

    assert (len(factory.made), report.stopped_early) == (6, False)


async def test_the_async_runner_stops_a_conversation_before_its_next_turn():
    """A turn in flight finishes; the next one is never sent."""
    factory = _sequence_factory("ok", "slow", "down_later", "down_later")

    report = await run_style_contract_async(
        factory, _pair_contract(repeats=2), concurrency=3
    )

    slow = factory.made[1]
    assert slow.seen == ["c"]
    lost = {r.label: r for r in report.lost}
    assert lost["warm#1"].replies == 1
    assert report.stopped_early is True


def test_the_summary_says_the_run_stopped_early():
    factory = _sequence_factory("ok", "down", "down", "ok", "ok", "ok")

    report = run_style_contract(factory, _pair_contract(repeats=3))

    assert "stopped early" in report.summary()


def test_a_clean_run_reports_nothing_lost():
    report = run_style_contract(
        lambda: ScriptedChatter(["uno?", "dos"]), _pair_contract(repeats=1)
    )

    assert report.lost == ()
    assert report.complete is True


def test_the_summary_names_every_run_it_did_not_score():
    report = run_style_contract(
        _flaky_factory({"d"}), _pair_contract(repeats=1)
    )

    line = [ln for ln in report.summary().splitlines() if "warm#1" in ln]
    assert len(line) == 1
    assert "not scored" in line[0]


def test_an_invocation_that_scored_nothing_raises():
    """A report over no runs would pass every budget silently."""
    with pytest.raises(ConnectionError, match="Connection error."):
        run_style_contract(
            _flaky_factory({"a", "c"}), _pair_contract(repeats=1)
        )


async def test_an_async_invocation_that_scored_nothing_raises():
    with pytest.raises(ConnectionError, match="Connection error."):
        await run_style_contract_async(
            _flaky_factory({"a", "c"}), _pair_contract(repeats=1)
        )


# ── Capitalisation, which is two rules and not one ─────────
#
# Opening a chat message in lower case is a register choice.
# Capitalising a new sentence inside the message is not. A persona
# that asks for both usually insists on only the second, and
# reading the insistence as attaching to the first enforces a
# preference while leaving the actual rule unchecked. The cases
# below are the kind of worked examples a persona gives.


def test_a_new_sentence_inside_a_message_is_capitalised():
    check = capitalises_after_stop()
    assert check(ctx("una adivinanza. ¿abro yo o abres tú?")) is True
    assert check(ctx("una adivinanza. ¿Abro yo o abres tú?")) is False


def test_it_looks_past_spanish_opening_marks():
    """`¿` must not shield the letter that follows it."""
    check = capitalises_after_stop()
    assert check(ctx("qué va, cero series. ni netflix ni nada")) is True
    assert check(ctx("qué va, cero series. Ni Netflix ni nada")) is False


def test_the_opening_letter_is_not_its_business():
    """A lower-case first letter is the other rule's subject."""
    check = capitalises_after_stop()
    assert check(ctx("vale, lo dejamos")) is None
    assert check(ctx("vale. Lo dejamos")) is False


def test_an_ellipsis_is_not_a_sentence_end():
    """Trailing off continues the thought, in lower case."""
    check = capitalises_after_stop()
    assert check(ctx("no sé... igual mañana")) is None


def test_a_decimal_point_is_not_a_sentence_end():
    check = capitalises_after_stop()
    assert check(ctx("me quedan 3.5 kilómetros")) is None


def test_it_checks_every_message_of_a_reply():
    check = capitalises_after_stop()
    assert check(ctx("todo bien\n\nvale. ni idea")) is True


def test_exclamations_and_questions_end_sentences_too():
    check = capitalises_after_stop()
    assert check(ctx("¡qué va! no me lo creo")) is True
    assert check(ctx("¿en serio? Pues vaya")) is False


# ── What each run was shown ────────────────────────────────
#
# Since v9.12.0 an agent with `prompt_variants` draws its texts per
# conversation, so two runs of one config can be shown different
# examples. A rate that moves with what was shown looks exactly
# like one that moves with the config unless the run says which.


def _drawing_factory(make, texts):
    """Build agents that each report the next text as drawn."""
    remaining = list(texts)

    def factory():
        agent = make()
        agent.variant_values = {"dry": remaining.pop(0)}
        return agent

    return factory


def test_the_report_keeps_what_each_run_was_shown():
    factory = _drawing_factory(
        lambda: ScriptedChatter(["uno?", "dos"]), ["x", "y", "z", "w"]
    )

    report = run_style_contract(factory, _pair_contract(repeats=2))

    assert report.variants == {
        "cold#1": {"dry": "x"},
        "warm#1": {"dry": "y"},
        "cold#2": {"dry": "z"},
        "warm#2": {"dry": "w"},
    }


async def test_the_async_report_keeps_what_each_run_was_shown():
    factory = _drawing_factory(
        lambda: ScriptedChatter(["uno?", "dos"]), ["x", "y"]
    )

    report = await run_style_contract_async(
        factory, _pair_contract(repeats=1), concurrency=1
    )

    assert report.variants == {"cold#1": {"dry": "x"}, "warm#1": {"dry": "y"}}


def test_a_lost_run_keeps_no_variants():
    """Only scored runs have a score to attribute."""
    factory = _drawing_factory(
        lambda: FlakyChatter(["uno?", "dos"], fail_on={"d"}), ["x", "y"]
    )

    report = run_style_contract(factory, _pair_contract(repeats=1))

    assert report.variants == {"cold#1": {"dry": "x"}}


def test_an_agent_without_variants_leaves_the_report_empty():
    report = run_style_contract(
        lambda: ScriptedChatter(["uno?", "dos"]), _pair_contract(repeats=1)
    )

    assert report.variants == {}
