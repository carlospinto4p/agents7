"""Unit tests for character facts (store, extractor, Chatter)."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from agents7.agents.chatter import Chatter
from agents7.agents.facts import (
    CharacterFact,
    ExtractedFacts,
    FactExtractor,
    JsonFileFactStore,
)
from agents7.agents.persona import Persona
from agents7.exceptions import AgentConfigurationError
from agents7.llm import LLMEngine
from agents7.models import AgentResponse, LLMMetrics

from helpers import DUMMY_METRICS, ENGINE, chat_response_dict, make_chatter


# Facts are stored in the character's own language — David is
# Spanish-speaking.
FILM = "Su película favorita es El Club de la Lucha"
CITY = "Vive en Malasaña"


def make_extractor(**kwargs) -> FactExtractor:
    """Create a `FactExtractor` with test defaults."""
    defaults = {"name": "test-facts", "version": 1, "engine": ENGINE}
    defaults.update(kwargs)
    return FactExtractor(**defaults)


def extractor_response(facts: list[str]) -> AgentResponse:
    """Build a mock `act()` return value for the extractor."""
    return AgentResponse(
        response=json.dumps({"facts": facts}),
        model=ExtractedFacts(facts=facts),
        thinking=None,
        metrics=LLMMetrics(**DUMMY_METRICS),
    )


# ── JsonFileFactStore Tests ─────────────────────────────────


def test_store_empty_for_unknown_character(tmp_path):
    """A character with no file has no facts."""
    store = JsonFileFactStore(str(tmp_path))
    assert store.load("david") == []


def test_store_add_returns_written_facts(tmp_path):
    """add() returns exactly what it wrote."""
    store = JsonFileFactStore(str(tmp_path))
    added = store.add("david", [FILM, CITY])
    assert [f.text for f in added] == [FILM, CITY]


def test_store_persists_across_instances(tmp_path):
    """Facts survive into a fresh store on the same path."""
    JsonFileFactStore(str(tmp_path)).add("david", [FILM])
    reloaded = JsonFileFactStore(str(tmp_path)).load("david")
    assert [f.text for f in reloaded] == [FILM]


def test_store_skips_exact_duplicate(tmp_path):
    """The same fact twice is written once."""
    store = JsonFileFactStore(str(tmp_path))
    store.add("david", [FILM])
    assert store.add("david", [FILM]) == []
    assert len(store.load("david")) == 1


def test_store_duplicate_ignores_case_and_full_stop(tmp_path):
    """De-duplication ignores case, spacing and a trailing dot."""
    store = JsonFileFactStore(str(tmp_path))
    store.add("david", [FILM])
    assert store.add("david", [f"  {FILM.lower()}.  "]) == []


def test_store_skips_blank_facts(tmp_path):
    """Empty and whitespace-only facts are dropped."""
    store = JsonFileFactStore(str(tmp_path))
    assert store.add("david", ["", "   "]) == []
    assert store.load("david") == []


def test_store_keeps_source_conversation(tmp_path):
    """The originating conversation is stored and read back."""
    store = JsonFileFactStore(str(tmp_path))
    store.add(
        "david",
        [CharacterFact(text=FILM, source_conversation="chat-7")],
    )
    assert store.load("david")[0].source_conversation == "chat-7"


def test_store_characters_are_isolated(tmp_path):
    """One character's facts never reach another."""
    store = JsonFileFactStore(str(tmp_path))
    store.add("david", [FILM])
    store.add("marta", [CITY])
    assert [f.text for f in store.load("david")] == [FILM]
    assert [f.text for f in store.load("marta")] == [CITY]


def test_store_clear_removes_only_that_character(tmp_path):
    """clear() empties one character and leaves the rest."""
    store = JsonFileFactStore(str(tmp_path))
    store.add("david", [FILM])
    store.add("marta", [CITY])
    store.clear("david")
    assert store.load("david") == []
    assert len(store.load("marta")) == 1


def test_store_lists_characters(tmp_path):
    """list_characters() reports every stored character."""
    store = JsonFileFactStore(str(tmp_path))
    store.add("david", [FILM])
    store.add("marta", [CITY])
    assert store.list_characters() == ["david", "marta"]


def test_store_lists_nothing_before_first_write(tmp_path):
    """A store whose folder does not exist lists nothing."""
    store = JsonFileFactStore(str(tmp_path / "missing"))
    assert store.list_characters() == []


def test_store_file_is_readable_json(tmp_path):
    """Facts land on disk as editable JSON."""
    store = JsonFileFactStore(str(tmp_path))
    store.add("david", [FILM])
    raw = json.loads((tmp_path / "david.json").read_text("utf-8"))
    assert raw == [{"text": FILM, "source_conversation": ""}]


def test_store_sanitises_character_id(tmp_path):
    """A path-traversing id cannot escape the folder."""
    store = JsonFileFactStore(str(tmp_path))
    store.add("../escape", [FILM])
    assert store.load("../escape")[0].text == FILM
    assert not (tmp_path.parent / "escape.json").exists()


# ── FactExtractor Tests ─────────────────────────────────────


def test_extractor_returns_parsed_facts():
    """extract() returns the model's fact list."""
    extractor = make_extractor()
    with patch.object(
        FactExtractor, "act", return_value=extractor_response([FILM])
    ):
        assert extractor.extract("David", [], []) == [FILM]


def test_extractor_drops_blank_facts():
    """Blank entries from the model are discarded."""
    extractor = make_extractor()
    with patch.object(
        FactExtractor, "act", return_value=extractor_response([FILM, "  "])
    ):
        assert extractor.extract("David", []) == [FILM]


def test_extractor_returns_empty_when_parse_failed():
    """An unparseable response yields no facts, not an error."""
    extractor = make_extractor()
    failed = AgentResponse(
        response="not json",
        model=None,
        thinking=None,
        metrics=LLMMetrics(**DUMMY_METRICS),
    )
    with patch.object(FactExtractor, "act", return_value=failed):
        assert extractor.extract("David", []) == []


def test_extractor_input_carries_transcript_and_known_facts():
    """The prompt payload names the character, turns and known facts."""
    extractor = make_extractor()
    transcript = [{"role": "user", "content": "¿Qué peli te gusta?"}]
    with patch.object(
        FactExtractor, "act", return_value=extractor_response([])
    ) as mock_act:
        extractor.extract("David", transcript, [CITY])
    payload = mock_act.call_args[0][0]
    assert payload["character"] == "David"
    assert payload["known_facts"] == [CITY]
    assert payload["transcript"] == transcript


def test_extractor_language_reaches_the_prompt():
    """The character's language is sent to the model."""
    extractor = make_extractor()
    with patch.object(
        FactExtractor, "act", return_value=extractor_response([])
    ) as mock_act:
        extractor.extract("David", [], language="Spanish")
    assert mock_act.call_args[0][0]["language"] == "Spanish"


def test_extractor_language_is_empty_by_default():
    """With no language given, the model follows the transcript."""
    extractor = make_extractor()
    with patch.object(
        FactExtractor, "act", return_value=extractor_response([])
    ) as mock_act:
        extractor.extract("David", [])
    assert mock_act.call_args[0][0]["language"] == ""


def test_extractor_prompt_names_the_language_field():
    """The prompt tells the model to write in ``language``."""
    prompt = make_extractor().system_prompt
    assert "``language``" in prompt
    assert "One short sentence, in ``language``" in prompt


def test_extractor_prompt_keeps_proper_nouns():
    """Names and titles are kept as the character wrote them."""
    prompt = make_extractor().system_prompt
    assert "Keep proper nouns exactly as the character wrote" in prompt
    assert 'do not render it as "Fight Club"' in prompt


def test_extractor_prompt_keeps_the_characters_own_tense():
    """A past thing stays past, with the words that date it.

    A character that said "la de mi piso en Bilbao, hace años"
    could be recorded as "Tiene un piso en Bilbao": the "hace
    años" dropped, and a flat the character had lived in became
    one it has.
    A fact store is replayed into every later prompt, so a wrong
    tense compounds.
    """
    prompt = make_extractor().system_prompt
    assert "Keep the character's own tense" in prompt
    assert 'never "Tiene un piso en Bilbao"' in prompt
    assert "Never promote a past fact into a present one" in prompt
    assert 'never drop a "hace años"' in prompt


def test_extractor_prompt_rejects_passing_states_by_example():
    """Passing states carry a durability test and examples.

    Three of four local models turned a conversation with no
    durable content into "is busy with the computer" until the
    passing-states rule was given a test and worked examples.
    """
    prompt = make_extractor().system_prompt
    assert "stops being true once the conversation ends" in prompt
    assert '"is busy with the computer"' in prompt
    assert '"laughed at the first question"' in prompt


def test_extractor_prompt_prefers_empty_over_a_weak_fact():
    """Coming back empty-handed beats softening a passing state."""
    prompt = make_extractor().system_prompt
    assert "Never soften a passing state into a fact" in prompt
    assert "an empty list costs nothing" in prompt


@pytest.mark.asyncio
async def test_extractor_async_returns_parsed_facts():
    """extract_async() mirrors the sync version."""
    extractor = make_extractor()
    with patch.object(
        FactExtractor,
        "act_async",
        new=AsyncMock(return_value=extractor_response([FILM])),
    ):
        assert await extractor.extract_async("David", []) == [FILM]


# ── Chatter Fact Reading Tests ──────────────────────────────


def test_chatter_without_store_has_no_facts():
    """A chatter with no store reports no facts."""
    assert make_chatter().character_facts == []


def test_chatter_prompt_unchanged_without_facts(tmp_path):
    """An empty store adds nothing to the system prompt."""
    agent = make_chatter(facts=JsonFileFactStore(str(tmp_path)))
    system = agent._build_messages()[0]["content"]
    assert system == "You are a helpful assistant."


def test_chatter_loads_facts_on_init(tmp_path):
    """Recorded facts are read when the chatter is built."""
    store = JsonFileFactStore(str(tmp_path))
    store.add("test-chatter", [FILM])
    agent = make_chatter(facts=store)
    assert [f.text for f in agent.character_facts] == [FILM]


def test_chatter_character_id_defaults_to_name(tmp_path):
    """Without an explicit id, the agent's name keys the store."""
    agent = make_chatter(facts=JsonFileFactStore(str(tmp_path)))
    assert agent.character_id == "test-chatter"


def test_chatter_explicit_character_id_is_kept(tmp_path):
    """An explicit character_id wins over the name."""
    store = JsonFileFactStore(str(tmp_path))
    store.add("david", [FILM])
    agent = make_chatter(facts=store, character_id="david")
    assert [f.text for f in agent.character_facts] == [FILM]


def test_chatter_folds_facts_into_system_prompt(tmp_path):
    """Facts reach the LLM as a system-prompt section."""
    store = JsonFileFactStore(str(tmp_path))
    store.add("test-chatter", [FILM, CITY])
    agent = make_chatter(facts=store)
    system = agent._build_messages()[0]["content"]
    assert system.startswith("You are a helpful assistant.")
    assert f"- {FILM}" in system
    assert f"- {CITY}" in system


def test_chatter_two_agents_share_one_character(tmp_path):
    """A second conversation sees the first one's facts."""
    store = JsonFileFactStore(str(tmp_path))
    store.add("david", [FILM])
    second = make_chatter(
        facts=store,
        character_id="david",
        conversation_id="chat-2",
    )
    assert FILM in second._build_messages()[0]["content"]


# ── Chatter remember() Tests ────────────────────────────────


@patch.object(FactExtractor, "act")
def test_remember_writes_extracted_facts(mock_act, tmp_path):
    """remember() stores what the extractor found."""
    mock_act.return_value = extractor_response([FILM])
    store = JsonFileFactStore(str(tmp_path))
    agent = make_chatter(
        facts=store,
        character_id="david",
        fact_extractor=make_extractor(),
    )
    with patch.object(
        type(agent.engine), "chat", return_value=chat_response_dict("Hola")
    ):
        agent.chat("¿Qué peli te gusta?")

    added = agent.remember()
    assert [f.text for f in added] == [FILM]
    assert [f.text for f in store.load("david")] == [FILM]


@patch.object(FactExtractor, "act")
def test_remember_tags_the_source_conversation(mock_act, tmp_path):
    """A stored fact records which conversation produced it."""
    mock_act.return_value = extractor_response([FILM])
    store = JsonFileFactStore(str(tmp_path))
    agent = make_chatter(
        facts=store,
        fact_extractor=make_extractor(),
        conversation_id="chat-7",
    )
    with patch.object(
        type(agent.engine), "chat", return_value=chat_response_dict("Hola")
    ):
        agent.chat("Hola")

    assert agent.remember()[0].source_conversation == "chat-7"


@patch.object(FactExtractor, "act")
def test_remember_feeds_known_facts_to_the_extractor(mock_act, tmp_path):
    """Already-recorded facts are sent so they are not repeated."""
    mock_act.return_value = extractor_response([])
    store = JsonFileFactStore(str(tmp_path))
    store.add("test-chatter", [CITY])
    agent = make_chatter(facts=store, fact_extractor=make_extractor())
    with patch.object(
        type(agent.engine), "chat", return_value=chat_response_dict("Hola")
    ):
        agent.chat("Hola")

    agent.remember()
    assert mock_act.call_args[0][0]["known_facts"] == [CITY]


@patch.object(FactExtractor, "act")
def test_remember_skips_already_known_fact(mock_act, tmp_path):
    """A fact the store already holds is not written twice."""
    mock_act.return_value = extractor_response([CITY])
    store = JsonFileFactStore(str(tmp_path))
    store.add("test-chatter", [CITY])
    agent = make_chatter(facts=store, fact_extractor=make_extractor())
    with patch.object(
        type(agent.engine), "chat", return_value=chat_response_dict("Hola")
    ):
        agent.chat("Hola")

    assert agent.remember() == []
    assert len(store.load("test-chatter")) == 1


@patch.object(FactExtractor, "act")
def test_remembered_fact_reaches_the_next_prompt(mock_act, tmp_path):
    """A newly recorded fact is in the very next system prompt."""
    mock_act.return_value = extractor_response([FILM])
    agent = make_chatter(
        facts=JsonFileFactStore(str(tmp_path)),
        fact_extractor=make_extractor(),
    )
    with patch.object(
        type(agent.engine), "chat", return_value=chat_response_dict("Hola")
    ):
        agent.chat("Hola")

    assert FILM not in agent._build_messages()[0]["content"]
    agent.remember()
    assert FILM in agent._build_messages()[0]["content"]


def test_remember_on_empty_conversation_calls_nothing(tmp_path):
    """With no turns yet, remember() does not call the LLM."""
    agent = make_chatter(
        facts=JsonFileFactStore(str(tmp_path)),
        fact_extractor=make_extractor(),
    )
    with patch.object(FactExtractor, "act") as mock_act:
        assert agent.remember() == []
    mock_act.assert_not_called()


def test_remember_without_store_raises(tmp_path):
    """remember() refuses rather than silently discarding facts."""
    agent = make_chatter(fact_extractor=make_extractor())
    with pytest.raises(AgentConfigurationError, match="`facts` store"):
        agent.remember()


def test_remember_without_extractor_raises(tmp_path):
    """A store with no extractor cannot record anything."""
    agent = make_chatter(facts=JsonFileFactStore(str(tmp_path)))
    with pytest.raises(AgentConfigurationError, match="fact_extractor"):
        agent.remember()


@pytest.mark.asyncio
async def test_remember_async_writes_extracted_facts(tmp_path):
    """remember_async() mirrors the sync version."""
    store = JsonFileFactStore(str(tmp_path))
    agent = make_chatter(
        facts=store,
        character_id="david",
        fact_extractor=make_extractor(),
    )
    agent._history.append({"role": "user", "content": "Hola"})
    with patch.object(
        FactExtractor,
        "act_async",
        new=AsyncMock(return_value=extractor_response([FILM])),
    ):
        added = await agent.remember_async()

    assert [f.text for f in added] == [FILM]
    assert [f.text for f in store.load("david")] == [FILM]


# ── remember() Call Log Tests ───────────────────────────────


def _extractor_reply(text: str) -> dict:
    """A mock `engine.chat()` return value for the extractor."""
    return chat_response_dict(text)


def _remembering_chatter(tmp_path, **kwargs) -> Chatter:
    """A chatter with one exchange in history, ready to remember."""
    defaults = {
        "facts": JsonFileFactStore(str(tmp_path)),
        "fact_extractor": make_extractor(),
    }
    defaults.update(kwargs)
    agent = make_chatter(**defaults)
    agent._history.append({"role": "user", "content": "Hola"})
    return agent


@patch.object(LLMEngine, "chat")
def test_remember_logs_the_extractor_call(mock_chat, tmp_path):
    """The extraction lands in the chatter's call log."""
    mock_chat.return_value = _extractor_reply(json.dumps({"facts": [FILM]}))
    records = []
    agent = _remembering_chatter(tmp_path, call_logger=records.append)

    agent.remember()

    assert [(r.agent_name, r.success) for r in records] == [
        ("test-facts", True)
    ]


@patch.object(LLMEngine, "chat")
def test_remember_logs_an_unparseable_extraction_as_failed(
    mock_chat, tmp_path
):
    """No facts from a parse failure is told apart from none found."""
    mock_chat.return_value = _extractor_reply("not json")
    records = []
    agent = _remembering_chatter(tmp_path, call_logger=records.append)

    assert agent.remember() == []
    assert [r.success for r in records] == [False]
    assert records[0].prompt is not None


@pytest.mark.asyncio
@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
async def test_remember_async_logs_the_extractor_call(mock_chat, tmp_path):
    """remember_async() logs the extraction too."""
    mock_chat.return_value = _extractor_reply(json.dumps({"facts": [FILM]}))
    records = []
    agent = _remembering_chatter(tmp_path, call_logger=records.append)

    await agent.remember_async()

    assert [r.agent_name for r in records] == ["test-facts"]


@patch.object(LLMEngine, "chat")
def test_remember_uses_a_logger_attached_after_construction(
    mock_chat, tmp_path
):
    """An eval run swaps the logger in later; it still catches it."""
    mock_chat.return_value = _extractor_reply(json.dumps({"facts": [FILM]}))
    records = []
    agent = _remembering_chatter(tmp_path)
    agent._set_attr("call_logger", records.append)

    agent.remember()

    assert len(records) == 1


@patch.object(LLMEngine, "chat")
def test_remember_keeps_the_extractor_own_logger(mock_chat, tmp_path):
    """An extractor declared with its own logger writes there only."""
    mock_chat.return_value = _extractor_reply(json.dumps({"facts": [FILM]}))
    chatter_records, extractor_records = [], []
    agent = _remembering_chatter(
        tmp_path,
        call_logger=chatter_records.append,
        fact_extractor=make_extractor(call_logger=extractor_records.append),
    )

    agent.remember()

    assert (len(chatter_records), len(extractor_records)) == (0, 1)


@patch.object(FactExtractor, "act")
def test_remember_uses_persona_full_name(mock_act, tmp_path):
    """The extractor is told the persona's name, not the agent's."""
    mock_act.return_value = extractor_response([])
    agent = Chatter(
        name="david-chat",
        version=1,
        engine=ENGINE,
        system_prompt="",
        persona=Persona(full_name="David"),
        facts=JsonFileFactStore(str(tmp_path)),
        fact_extractor=make_extractor(),
    )
    agent._history.append({"role": "user", "content": "Hola"})
    agent.remember()
    assert mock_act.call_args[0][0]["character"] == "David"


@patch.object(FactExtractor, "act")
def test_remember_passes_the_persona_language(mock_act, tmp_path):
    """Facts are written in the language the persona declares."""
    mock_act.return_value = extractor_response([])
    agent = Chatter(
        name="david-chat",
        version=1,
        engine=ENGINE,
        system_prompt="",
        persona=Persona(full_name="David", language="Spanish"),
        facts=JsonFileFactStore(str(tmp_path)),
        fact_extractor=make_extractor(),
    )
    # The other person writes English; the fact is still Spanish.
    agent._history.append({"role": "user", "content": "Hi there"})
    agent.remember()
    assert mock_act.call_args[0][0]["language"] == "Spanish"


@patch.object(FactExtractor, "act")
def test_remember_without_a_persona_sends_no_language(mock_act, tmp_path):
    """No persona means the extractor follows the transcript."""
    mock_act.return_value = extractor_response([])
    agent = make_chatter(
        facts=JsonFileFactStore(str(tmp_path)),
        fact_extractor=make_extractor(),
    )
    agent._history.append({"role": "user", "content": "Hola"})
    agent.remember()
    assert mock_act.call_args[0][0]["language"] == ""


@patch.object(FactExtractor, "act")
def test_remember_persona_without_a_language_sends_none(mock_act, tmp_path):
    """A persona that declares no language leaves it empty."""
    mock_act.return_value = extractor_response([])
    agent = Chatter(
        name="thomas",
        version=1,
        engine=ENGINE,
        system_prompt="",
        persona=Persona(full_name="Old Thomas"),
        facts=JsonFileFactStore(str(tmp_path)),
        fact_extractor=make_extractor(),
    )
    agent._history.append({"role": "user", "content": "Good evening"})
    agent.remember()
    assert mock_act.call_args[0][0]["language"] == ""


# ── Banned Fact Phrase Tests ────────────────────────────────


SINGLE = "Cocina fatal, sin remedio."


def test_stored_fact_with_banned_phrase_stays_out_of_prompt(tmp_path):
    """A fact recorded before a ban never reaches the prompt.

    A store can keep a phrase from a chat held before the persona
    banned it, and the model would then say it again.
    """
    store = JsonFileFactStore(str(tmp_path))
    store.add("test-chatter", [SINGLE, FILM])
    agent = make_chatter(facts=store, banned_fact_phrases=("sin remedio",))
    system = agent._build_messages()[0]["content"]
    assert SINGLE not in system
    assert FILM in system


def test_banned_phrase_match_ignores_case_and_spacing(tmp_path):
    """The ban matches however the fact happens to be written."""
    store = JsonFileFactStore(str(tmp_path))
    store.add("test-chatter", ["Cocina fatal y SIN   remedio"])
    agent = make_chatter(facts=store, banned_fact_phrases=("Sin remedio",))
    assert agent.character_facts == []


def test_banned_fact_is_left_in_the_file(tmp_path):
    """The store is the operator's to edit, so nothing is deleted."""
    store = JsonFileFactStore(str(tmp_path))
    store.add("test-chatter", [SINGLE])
    make_chatter(facts=store, banned_fact_phrases=("sin remedio",))
    assert [f.text for f in store.load("test-chatter")] == [SINGLE]


def test_banned_stored_fact_is_logged(tmp_path, caplog):
    """Leaving a fact out says which one, so it can be cleaned up."""
    store = JsonFileFactStore(str(tmp_path))
    store.add("test-chatter", [SINGLE])
    with caplog.at_level("WARNING", logger="agents7"):
        make_chatter(facts=store, banned_fact_phrases=("sin remedio",))
    assert SINGLE in caplog.text
    assert "sin remedio" in caplog.text


@patch.object(FactExtractor, "act")
def test_remember_does_not_store_a_banned_fact(mock_act, tmp_path):
    """An extracted fact with a banned phrase is never written."""
    mock_act.return_value = extractor_response([SINGLE, FILM])
    store = JsonFileFactStore(str(tmp_path))
    agent = make_chatter(
        facts=store,
        fact_extractor=make_extractor(),
        banned_fact_phrases=("sin remedio",),
    )
    agent._history.append({"role": "user", "content": "¿Tienes perro?"})

    added = agent.remember()
    assert [f.text for f in added] == [FILM]
    assert [f.text for f in store.load("test-chatter")] == [FILM]


def test_blank_banned_phrase_is_rejected(tmp_path):
    """A blank phrase would match every fact and empty the prompt."""
    with pytest.raises(AgentConfigurationError, match="banned"):
        make_chatter(
            facts=JsonFileFactStore(str(tmp_path)),
            banned_fact_phrases=("sin remedio", "  "),
        )


def test_from_yaml_reads_banned_phrases(tmp_path):
    """`character_facts.banned_phrases` reaches the chatter."""
    path = write_config(
        tmp_path,
        "character_facts:\n"
        f"    path: {tmp_path / 'facts'}\n"
        "    banned_phrases:\n"
        "        - sin remedio\n",
    )
    agent = Chatter.from_yaml(path)
    assert agent.banned_fact_phrases == ("sin remedio",)


def test_from_yaml_without_banned_phrases_bans_nothing(tmp_path):
    """No `banned_phrases` key leaves every fact in the prompt."""
    path = write_config(
        tmp_path,
        f"character_facts:\n    path: {tmp_path / 'facts'}\n",
    )
    assert Chatter.from_yaml(path).banned_fact_phrases == ()


# ── Chatter forget() and reset() Tests ──────────────────────


def test_forget_clears_store_and_prompt(tmp_path):
    """forget() empties the store and the system prompt."""
    store = JsonFileFactStore(str(tmp_path))
    store.add("test-chatter", [FILM])
    agent = make_chatter(facts=store)

    agent.forget()
    assert agent.character_facts == []
    assert store.load("test-chatter") == []
    assert FILM not in agent._build_messages()[0]["content"]


def test_reset_keeps_the_facts(tmp_path):
    """Clearing the conversation does not erase the character."""
    store = JsonFileFactStore(str(tmp_path))
    store.add("test-chatter", [FILM])
    agent = make_chatter(facts=store)

    agent.reset()
    assert [f.text for f in agent.character_facts] == [FILM]


def test_forget_without_store_is_a_no_op():
    """forget() on a chatter with no store does nothing."""
    make_chatter().forget()


# ── describe() Tests ────────────────────────────────────────


def test_describe_reports_fact_count(tmp_path):
    """describe() names the character and its fact count."""
    store = JsonFileFactStore(str(tmp_path))
    store.add("david", [FILM, CITY])
    agent = make_chatter(facts=store, character_id="david")
    assert "Character facts: 2 recorded for `david`" in agent.describe()


def test_describe_omits_facts_without_a_store():
    """A chatter with no store says nothing about facts."""
    assert "Character facts" not in make_chatter().describe()


# ── YAML Loading Tests ──────────────────────────────────────


def write_config(tmp_path, facts_section: str) -> str:
    """Write a Chatter YAML config and return its path."""
    yaml_file = tmp_path / "chatter.yml"
    yaml_file.write_text(
        "name: FactChatter\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Hello.\n" + facts_section,
        encoding="utf-8",
    )
    return str(yaml_file)


def test_from_yaml_builds_store_and_extractor(tmp_path):
    """A character_facts section wires up both halves."""
    path = write_config(
        tmp_path,
        "character_facts:\n"
        f"    path: {tmp_path / 'facts'}\n"
        "    character_id: david\n",
    )
    agent = Chatter.from_yaml(path)
    assert isinstance(agent.facts, JsonFileFactStore)
    assert isinstance(agent.fact_extractor, FactExtractor)
    assert agent.character_id == "david"


def test_from_yaml_extractor_reuses_the_agent_engine(tmp_path):
    """Without its own engine, the extractor shares the agent's."""
    path = write_config(
        tmp_path,
        f"character_facts:\n    path: {tmp_path / 'facts'}\n",
    )
    agent = Chatter.from_yaml(path)
    assert agent.fact_extractor.engine.model_name == "qwen3"


def test_from_yaml_extractor_takes_its_own_engine(tmp_path):
    """A declared extractor engine overrides the agent's."""
    path = write_config(
        tmp_path,
        "character_facts:\n"
        f"    path: {tmp_path / 'facts'}\n"
        "    extractor:\n"
        "        engine:\n"
        "            provider: ollama\n"
        "            model: gemma3\n",
    )
    agent = Chatter.from_yaml(path)
    assert agent.fact_extractor.engine.model_name == "gemma3"


def test_from_yaml_character_id_defaults_to_name(tmp_path):
    """No character_id in YAML means the agent's name is the key."""
    path = write_config(
        tmp_path,
        f"character_facts:\n    path: {tmp_path / 'facts'}\n",
    )
    assert Chatter.from_yaml(path).character_id == "FactChatter"


def test_from_yaml_without_facts_section(tmp_path):
    """No character_facts section leaves both fields unset."""
    agent = Chatter.from_yaml(write_config(tmp_path, ""))
    assert agent.facts is None
    assert agent.fact_extractor is None


def test_guarded_scene_from_yaml_builds_facts(tmp_path):
    """The section works on the guarded-scene loader too.

    This is the shape a guarded scene chatter with facts uses.
    """
    from agents7.agents.guarded_scene import GuardedSceneChatter

    yaml_file = tmp_path / "guarded_scene.yml"
    yaml_file.write_text(
        "name: david-chat\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "persona:\n"
        "    full_name: David\n"
        "scene:\n"
        "    setting: Un café en Malasaña\n"
        "character_facts:\n"
        f"    path: {tmp_path / 'facts'}\n"
        "    character_id: david\n",
        encoding="utf-8",
    )
    agent = GuardedSceneChatter.from_yaml(str(yaml_file))
    assert isinstance(agent.facts, JsonFileFactStore)
    assert isinstance(agent.fact_extractor, FactExtractor)
    assert agent.character_id == "david"


def test_from_yaml_reads_facts_written_earlier(tmp_path):
    """A YAML-built chatter picks up facts already on disk."""
    JsonFileFactStore(str(tmp_path / "facts")).add("david", [FILM])
    path = write_config(
        tmp_path,
        "character_facts:\n"
        f"    path: {tmp_path / 'facts'}\n"
        "    character_id: david\n",
    )
    agent = Chatter.from_yaml(path)
    assert FILM in agent._build_messages()[0]["content"]


# ── Store Path Resolution Tests ─────────────────────────────


def test_relative_facts_path_sits_beside_the_config(tmp_path, monkeypatch):
    """A relative path follows the config, not the caller.

    The same config loaded from two working directories must
    build the same store — otherwise a web backend and a script
    each record into their own folder and neither sees the
    other's facts.
    """
    config_dir = tmp_path / "agent"
    config_dir.mkdir()
    path = write_config(
        config_dir,
        "character_facts:\n    path: facts/\n    character_id: david\n",
    )

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    agent = Chatter.from_yaml(path)
    agent.facts.add("david", [FILM])

    assert (config_dir / "facts" / "david.json").exists()
    assert not (elsewhere / "facts").exists()


def test_the_default_facts_path_is_also_config_relative(tmp_path):
    """No `path` at all still lands beside the config."""
    config_dir = tmp_path / "agent"
    config_dir.mkdir()
    path = write_config(config_dir, "character_facts:\n    a: 1\n")

    Chatter.from_yaml(path).facts.add("FactChatter", [FILM])
    assert (config_dir / "facts" / "FactChatter.json").exists()


def test_an_absolute_facts_path_is_left_alone(tmp_path):
    """An explicit absolute path is honoured as written."""
    store_dir = tmp_path / "somewhere" / "facts"
    path = write_config(
        tmp_path,
        f"character_facts:\n    path: {store_dir}\n",
    )
    Chatter.from_yaml(path).facts.add("FactChatter", [FILM])
    assert (store_dir / "FactChatter.json").exists()


def test_relative_memory_path_sits_beside_the_config(tmp_path, monkeypatch):
    """`memory.path` is anchored the same way as the fact store."""
    config_dir = tmp_path / "agent"
    config_dir.mkdir()
    path = write_config(
        config_dir,
        "memory:\n    backend: json\n    path: memory/\n",
    )

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    agent = Chatter.from_yaml(path)
    agent.memory.save("default", [{"role": "user", "content": "hola"}])

    assert (config_dir / "memory" / "default.json").exists()
    assert not (elsewhere / "memory").exists()
