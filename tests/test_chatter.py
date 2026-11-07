"""Unit tests for the Chatter agent (no LLM required)."""

from unittest.mock import AsyncMock, patch

import pytest

from agents7.agents.chatter import (
    Chatter,
)
from agents7.agents.personality import Persona, Scene
from agents7.exceptions import AgentConfigurationError
from agents7.llm import LLMEngine
from agents7.models import ChatResponse

from helpers import (
    DUMMY_METRICS,
    async_iter,
    chat_chunk,
    chat_response_dict,
    make_chatter,
)


# --- Initialization ---


def test_chatter_init_defaults():
    """Chatter initializes with correct defaults."""
    agent = make_chatter()
    assert agent.name == "test-chatter"
    assert agent.version == 1
    assert agent.temperature == 0.7
    assert agent.think is False
    assert agent.max_history == 50
    assert agent.history == []


def test_chatter_custom_params():
    """Chatter accepts custom temperature, think, max_history."""
    agent = make_chatter(
        temperature=0.9,
        think=True,
        max_history=10,
    )
    assert agent.temperature == 0.9
    assert agent.think is True
    assert agent.max_history == 10


def test_scene_without_persona_raises():
    """A meaningful scene with no persona is a misconfiguration."""
    scene = Scene(setting="A medieval tavern", rules=["No modern tech."])
    with pytest.raises(AgentConfigurationError, match="scene.*requires"):
        make_chatter(scene=scene)


def test_empty_scene_without_persona_ok():
    """An empty scene (no content) with no persona does not raise."""
    agent = make_chatter(scene=Scene())
    assert agent.scene == Scene()
    assert agent.persona is None


def test_scene_with_persona_ok():
    """A scene paired with a persona builds the role-play prompt."""
    agent = make_chatter(
        persona=Persona(full_name="Old Thomas"),
        scene=Scene(setting="A medieval tavern"),
    )
    assert "medieval tavern" in agent.system_prompt
    assert "Old Thomas" in agent.system_prompt


def test_chatter_invalid_prompt_params():
    """Non-dict prompt_params raises PromptParameterError."""
    from agents7.exceptions import PromptParameterError

    with pytest.raises(PromptParameterError):
        make_chatter(prompt_params="not a dict")


# --- YAML loading ---


def test_from_yaml_valid(tmp_path):
    """Valid YAML config produces correct Chatter."""
    yaml_content = (
        "name: MyChatter\n"
        "version: 2\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: |\n"
        "  You are a pirate.\n"
        "temperature: 0.9\n"
        "max_history: 20\n"
    )
    yaml_file = tmp_path / "chatter.yml"
    yaml_file.write_text(yaml_content)

    agent = Chatter.from_yaml(str(yaml_file))
    assert agent.name == "MyChatter"
    assert agent.version == 2
    assert agent.temperature == 0.9
    assert agent.max_history == 20
    assert "pirate" in agent.system_prompt


def test_from_yaml_minimal(tmp_path):
    """Minimal YAML config uses defaults for optional fields."""
    yaml_content = (
        "name: MinChatter\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Hello.\n"
    )
    yaml_file = tmp_path / "minimal.yml"
    yaml_file.write_text(yaml_content)

    agent = Chatter.from_yaml(str(yaml_file))
    assert agent.temperature == 0.7
    assert agent.max_history == 50
    assert agent.think is False


def test_from_yaml_missing_system_prompt(tmp_path):
    """Missing system_prompt raises AgentConfigurationError."""
    yaml_content = (
        "name: Bad\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
    )
    yaml_file = tmp_path / "bad.yml"
    yaml_file.write_text(yaml_content)

    with pytest.raises(AgentConfigurationError, match="system_prompt"):
        Chatter.from_yaml(str(yaml_file))


def test_from_yaml_missing_engine(tmp_path):
    """Missing engine raises AgentConfigurationError."""
    yaml_content = "name: Bad\nversion: 1\nsystem_prompt: Hello.\n"
    yaml_file = tmp_path / "bad.yml"
    yaml_file.write_text(yaml_content)

    with pytest.raises(AgentConfigurationError, match="engine"):
        Chatter.from_yaml(str(yaml_file))


def test_from_yaml_invalid_memory_backend(tmp_path):
    """Invalid memory backend raises AgentConfigurationError."""
    yaml_content = (
        "name: Mem\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Hello.\n"
        "memory:\n"
        "    backend: redis\n"
        "    path: /tmp/mem\n"
    )
    yaml_file = tmp_path / "mem.yml"
    yaml_file.write_text(yaml_content)

    with pytest.raises(AgentConfigurationError, match="backend"):
        Chatter.from_yaml(str(yaml_file))


def test_from_yaml_circular_inheritance(tmp_path):
    """Circular base_config raises AgentConfigurationError."""
    (tmp_path / "a.yml").write_text("base_config: b.yml\nname: A\n")
    (tmp_path / "b.yml").write_text("base_config: a.yml\nname: B\n")

    with pytest.raises(AgentConfigurationError, match="Circular"):
        Chatter.from_yaml(str(tmp_path / "a.yml"))


def test_from_yaml_malformed_yaml(tmp_path):
    """Malformed YAML raises a YAML parse error."""
    import yaml

    yaml_file = tmp_path / "bad.yml"
    yaml_file.write_text("name: Bad\n  unexpected_indent: [unclosed\n")

    with pytest.raises(yaml.YAMLError):
        Chatter.from_yaml(str(yaml_file))


def test_from_yaml_empty_persona_dict_requires_full_name(tmp_path):
    """An empty `persona:` mapping raises a clear config error."""
    yaml_content = (
        "name: PersonaBot\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "persona: {}\n"
    )
    yaml_file = tmp_path / "persona.yml"
    yaml_file.write_text(yaml_content)

    with pytest.raises(AgentConfigurationError, match="persona.*full_name"):
        Chatter.from_yaml(str(yaml_file))


def test_from_yaml_unknown_persona_field_raises(tmp_path):
    """An unknown persona field raises a clear config error."""
    yaml_content = (
        "name: PersonaBot\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "persona:\n"
        "    full_name: Old Thomas\n"
        "    bogus_field: nope\n"
    )
    yaml_file = tmp_path / "persona.yml"
    yaml_file.write_text(yaml_content)

    with pytest.raises(AgentConfigurationError, match="persona"):
        Chatter.from_yaml(str(yaml_file))


def test_from_yaml_persona_with_full_name_loads(tmp_path):
    """A persona with full_name auto-generates the system prompt."""
    yaml_content = (
        "name: PersonaBot\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "persona:\n"
        "    full_name: Old Thomas\n"
    )
    yaml_file = tmp_path / "persona.yml"
    yaml_file.write_text(yaml_content)

    agent = Chatter.from_yaml(str(yaml_file))
    assert agent.persona is not None
    assert agent.persona.full_name == "Old Thomas"
    assert agent.scene is None
    assert "Old Thomas" in agent.system_prompt


def test_from_yaml_extra_instructions_stripped(tmp_path):
    """Surrounding whitespace on extra_instructions is stripped."""
    yaml_content = (
        "name: PersonaBot\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "persona:\n"
        "    full_name: Old Thomas\n"
        'extra_instructions: "  Speak in riddles.  "\n'
    )
    yaml_file = tmp_path / "extra.yml"
    yaml_file.write_text(yaml_content)

    agent = Chatter.from_yaml(str(yaml_file))
    assert agent.extra_instructions == "Speak in riddles."


def test_from_yaml_scene_without_persona_raises(tmp_path):
    """A meaningful scene with no persona raises via YAML."""
    yaml_content = (
        "name: SceneBot\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "scene:\n"
        "    setting: A medieval tavern\n"
    )
    yaml_file = tmp_path / "scene.yml"
    yaml_file.write_text(yaml_content)

    with pytest.raises(AgentConfigurationError, match="scene.*requires"):
        Chatter.from_yaml(str(yaml_file))


# --- chat() ---


@patch.object(LLMEngine, "chat")
def test_chat_returns_chat_response(mock_chat):
    """chat() returns a ChatResponse with the LLM reply."""
    mock_chat.return_value = chat_response_dict("Hi there!")
    agent = make_chatter()

    result = agent.chat("Hello")

    assert isinstance(result, ChatResponse)
    assert result.response == "Hi there!"
    assert result.thinking is None
    assert result.metrics.tokens == DUMMY_METRICS["tokens"]


@patch.object(LLMEngine, "chat")
def test_chat_with_thinking(mock_chat):
    """chat() passes thinking content through."""
    mock_chat.return_value = chat_response_dict(
        "Reply", thinking="Let me think..."
    )
    agent = make_chatter(think=True)

    result = agent.chat("Question")

    assert result.thinking == "Let me think..."


@patch.object(LLMEngine, "chat")
def test_chat_builds_correct_messages(mock_chat):
    """chat() sends system prompt + history to the engine."""
    mock_chat.return_value = chat_response_dict("Response")
    agent = make_chatter(system_prompt="Be nice.")

    agent.chat("First message")

    call_kwargs = mock_chat.call_args
    messages = call_kwargs.kwargs["messages"]
    assert messages[0] == {
        "role": "system",
        "content": "Be nice.",
    }
    assert messages[1] == {
        "role": "user",
        "content": "First message",
    }


@patch.object(LLMEngine, "chat")
def test_chat_uses_temperature(mock_chat):
    """chat() passes the agent's temperature to the engine."""
    mock_chat.return_value = chat_response_dict("OK")
    agent = make_chatter(temperature=0.3)

    agent.chat("Hi")

    call_kwargs = mock_chat.call_args
    assert call_kwargs.kwargs["temperature"] == 0.3


# --- History management ---


@patch.object(LLMEngine, "chat")
def test_history_accumulates(mock_chat):
    """History grows with each chat turn."""
    mock_chat.return_value = chat_response_dict("Reply")
    agent = make_chatter()

    agent.chat("First")
    agent.chat("Second")

    assert len(agent.history) == 4
    assert agent.history[0]["content"] == "First"
    assert agent.history[1]["content"] == "Reply"
    assert agent.history[2]["content"] == "Second"
    assert agent.history[3]["content"] == "Reply"


# --- Empty lines between messages ---


@pytest.mark.parametrize(
    ("generated", "stored"),
    [
        ("Uno.\n\nDos.", "Uno.\nDos."),
        ("\nUno.\n  \n\nDos.\n", "Uno.\nDos."),
        ("Uno.\nDos.", "Uno.\nDos."),
        ("Uno.", "Uno."),
    ],
)
@patch.object(LLMEngine, "chat")
def test_chat_drops_empty_lines(mock_chat, generated, stored):
    """An empty line would reach a chat client as an empty message."""
    mock_chat.return_value = chat_response_dict(generated)
    agent = make_chatter()

    resp = agent.chat("Hola")

    assert (resp.response, agent.history[-1]["content"]) == (stored, stored)


@pytest.mark.asyncio
@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
async def test_chat_async_drops_empty_lines(mock_chat):
    """The async path stores and returns the same cleaned reply."""
    mock_chat.return_value = chat_response_dict("Uno.\n\nDos.")
    agent = make_chatter()

    resp = await agent.chat_async("Hola")

    assert (resp.response, agent.history[-1]["content"]) == (
        "Uno.\nDos.",
        "Uno.\nDos.",
    )


@patch.object(LLMEngine, "chat_stream")
def test_chat_stream_drops_empty_lines_once_finished(mock_stream):
    """Chunks stream as generated; the finished reply is cleaned."""
    from agents7.llm.streaming import StreamResult

    chunks = [
        chat_chunk(content="Uno.\n\n"),
        chat_chunk(content="Dos."),
        chat_chunk(
            done=True,
            eval_count=10,
            prompt_eval_count=5,
            total_duration=100_000_000,
        ),
    ]
    mock_stream.return_value = StreamResult(iter(chunks), "chat")
    agent = make_chatter()

    result = agent.chat_stream("Hola")
    list(result)

    assert (
        result.chat_response.response,
        agent.history[-1]["content"],
    ) == ("Uno.\nDos.", "Uno.\nDos.")


@patch.object(LLMEngine, "chat")
def test_history_multi_turn_messages(mock_chat):
    """Engine receives full history on subsequent turns."""
    responses = [
        chat_response_dict("First reply"),
        chat_response_dict("Second reply"),
    ]
    mock_chat.side_effect = responses
    agent = make_chatter()

    agent.chat("Hello")
    agent.chat("Follow up")

    second_call = mock_chat.call_args_list[1]
    messages = second_call.kwargs["messages"]
    # system + user1 + assistant1 + user2 (assistant2 not yet)
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "Hello"
    assert messages[2]["content"] == "First reply"
    assert messages[3]["content"] == "Follow up"


@patch.object(LLMEngine, "chat")
def test_history_trimming(mock_chat):
    """History is trimmed when it exceeds max_history."""
    mock_chat.return_value = chat_response_dict("R")
    agent = make_chatter(max_history=2)

    # 3 turns = 6 messages, but max is 2*2=4
    agent.chat("A")
    agent.chat("B")
    agent.chat("C")

    assert len(agent.history) == 4
    # Oldest messages should be trimmed
    assert agent.history[0]["content"] == "B"
    assert agent.history[1]["content"] == "R"
    assert agent.history[2]["content"] == "C"
    assert agent.history[3]["content"] == "R"


@patch.object(LLMEngine, "chat")
def test_max_history_none_means_unlimited(mock_chat):
    """max_history=None keeps all messages."""
    mock_chat.return_value = chat_response_dict("R")
    agent = make_chatter(max_history=None)

    for i in range(20):
        agent.chat(f"Msg {i}")

    # 20 turns = 40 messages, all kept
    assert len(agent.history) == 40


def test_reset_clears_history():
    """reset() empties the conversation history."""
    agent = make_chatter()
    # Manually add to history for test
    agent._history.append({"role": "user", "content": "Hi"})
    assert len(agent.history) == 1

    agent.reset()

    assert agent.history == []


def test_history_returns_copy():
    """history property returns a copy, not the internal list."""
    agent = make_chatter()
    agent._history.append({"role": "user", "content": "Hi"})

    history = agent.history
    history.clear()

    assert len(agent.history) == 1


# --- act() ---


@patch.object(LLMEngine, "chat")
def test_act_delegates_to_chat(mock_chat):
    """act() delegates to chat() and returns AgentResponse."""
    mock_chat.return_value = chat_response_dict("Hello!")
    agent = make_chatter()

    result = agent.act("Hi")

    assert result.response == "Hello!"
    assert result.model is None
    assert result.metrics.tokens == DUMMY_METRICS["tokens"]


# --- chat_async() ---


@pytest.mark.asyncio
@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
async def test_chat_async(mock_chat):
    """chat_async() returns ChatResponse."""
    mock_chat.return_value = chat_response_dict("Async hi!")
    agent = make_chatter()

    result = await agent.chat_async("Hello")

    assert isinstance(result, ChatResponse)
    assert result.response == "Async hi!"


@pytest.mark.asyncio
@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
async def test_chat_async_updates_history(mock_chat):
    """chat_async() updates conversation history."""
    mock_chat.return_value = chat_response_dict("Reply")
    agent = make_chatter()

    await agent.chat_async("Question")

    assert len(agent.history) == 2
    assert agent.history[0]["content"] == "Question"
    assert agent.history[1]["content"] == "Reply"


# --- chat_stream() ---


@patch.object(LLMEngine, "chat_stream")
def test_chat_stream_yields_chunks(mock_stream):
    """chat_stream() yields text chunks."""
    from agents7.llm.streaming import StreamResult

    chunks = [
        chat_chunk(content="Hel"),
        chat_chunk(content="lo!"),
        chat_chunk(
            done=True,
            eval_count=10,
            prompt_eval_count=5,
            total_duration=100_000_000,
        ),
    ]
    stream = StreamResult(iter(chunks), "chat")
    mock_stream.return_value = stream

    agent = make_chatter()
    result = agent.chat_stream("Hi")
    collected = list(result)

    assert collected == ["Hel", "lo!"]
    assert result.chat_response is not None
    assert result.chat_response.response == "Hello!"


@patch.object(LLMEngine, "chat_stream")
def test_chat_stream_updates_history(mock_stream):
    """chat_stream() updates history after iteration."""
    from agents7.llm.streaming import StreamResult

    chunks = [
        chat_chunk(content="Reply"),
        chat_chunk(
            done=True,
            eval_count=10,
            prompt_eval_count=5,
            total_duration=100_000_000,
        ),
    ]
    stream = StreamResult(iter(chunks), "chat")
    mock_stream.return_value = stream

    agent = make_chatter()
    result = agent.chat_stream("Hi")
    list(result)  # Consume the stream

    assert len(agent.history) == 2
    assert agent.history[0]["content"] == "Hi"
    assert agent.history[1]["content"] == "Reply"


# --- chat_stream_async() ---


@pytest.mark.asyncio
@patch.object(LLMEngine, "chat_stream_async", new_callable=AsyncMock)
async def test_chat_stream_async_yields_chunks(mock_stream):
    """chat_stream_async() yields text chunks."""
    from agents7.llm.streaming import AsyncStreamResult

    chunks = [
        chat_chunk(content="He"),
        chat_chunk(content="y!"),
        chat_chunk(
            done=True,
            eval_count=10,
            prompt_eval_count=5,
            total_duration=100_000_000,
        ),
    ]
    stream = AsyncStreamResult(async_iter(chunks), "chat")
    mock_stream.return_value = stream

    agent = make_chatter()
    result = await agent.chat_stream_async("Hi")
    collected = [chunk async for chunk in result]

    assert collected == ["He", "y!"]
    assert result.chat_response is not None
    assert result.chat_response.response == "Hey!"


@pytest.mark.asyncio
@patch.object(LLMEngine, "chat_stream_async", new_callable=AsyncMock)
async def test_chat_stream_async_updates_history(mock_stream):
    """chat_stream_async() updates history after iteration."""
    from agents7.llm.streaming import AsyncStreamResult

    chunks = [
        chat_chunk(content="Streamed"),
        chat_chunk(
            done=True,
            eval_count=10,
            prompt_eval_count=5,
            total_duration=100_000_000,
        ),
    ]
    stream = AsyncStreamResult(async_iter(chunks), "chat")
    mock_stream.return_value = stream

    agent = make_chatter()
    result = await agent.chat_stream_async("Question")
    async for _ in result:
        pass

    assert len(agent.history) == 2
    assert agent.history[1]["content"] == "Streamed"


# --- prompt_params ---


@patch.object(LLMEngine, "chat")
def test_prompt_params_substituted(mock_chat):
    """prompt_params are substituted into system_prompt."""
    mock_chat.return_value = chat_response_dict("OK")
    agent = make_chatter(
        system_prompt="You are $name, a $role.",
        prompt_params={"name": "Alice", "role": "teacher"},
    )

    agent.chat("Hi")

    messages = mock_chat.call_args.kwargs["messages"]
    assert messages[0]["content"] == "You are Alice, a teacher."


# --- $character ---


def test_character_placeholder_resolves_to_the_persona_name():
    """`$character` is the persona's own name, always.

    It is what lets a `chat_goal` file write a worked example
    without asserting who is in it, so the same goal reads
    correctly under any character.
    """
    agent = make_chatter(
        persona=Persona(full_name="David"),
        extra_instructions="$character: buenas",
    )
    assert "David: buenas" in agent.resolved_system_prompt


def test_character_placeholder_needs_no_declaration():
    """A persona agent gets it without asking."""
    agent = make_chatter(persona=Persona(full_name="David"))
    assert agent.prompt_params["character"] == "David"


def test_character_placeholder_swaps_with_the_persona():
    """The point of it: rename the character, nothing else."""
    goal = "$character: buenas"
    david = make_chatter(
        persona=Persona(full_name="David"),
        extra_instructions=goal,
    )
    marta = make_chatter(
        persona=Persona(full_name="Marta"),
        extra_instructions=goal,
    )
    assert "David: buenas" in david.resolved_system_prompt
    assert "Marta: buenas" in marta.resolved_system_prompt


def test_an_explicit_character_param_wins():
    """A config that sets it meant to set it."""
    agent = make_chatter(
        persona=Persona(full_name="David"),
        prompt_params={"character": "Davo"},
        extra_instructions="$character: buenas",
    )
    assert "Davo: buenas" in agent.resolved_system_prompt


def test_no_persona_declares_no_character_param():
    """An agent with no character has no name to substitute."""
    agent = make_chatter(system_prompt="Plain $character.")
    assert agent.prompt_params is None
    assert agent.resolved_system_prompt == "Plain $character."


@patch.object(LLMEngine, "chat")
def test_the_opening_block_resolves_the_placeholder(mock_chat):
    """The opening guidance joins the prompt per call.

    It is appended after `resolved_system_prompt` rather than
    being part of it, so it needs the substitution applied
    separately — a placeholder that worked in one block and not
    the other would be worse than none.
    """
    mock_chat.return_value = chat_response_dict("buenas")
    agent = make_chatter(
        persona=Persona(full_name="David"),
        can_open=True,
        opening_instructions="$character: toc toc",
        opening_prompt="[Abre tú, $character.]",
    )

    agent.open_chat()

    messages = mock_chat.call_args.kwargs["messages"]
    assert "David: toc toc" in messages[0]["content"]
    assert messages[-1]["content"] == "[Abre tú, David.]"


# --- describe() ---


def test_describe():
    """describe() returns a human-readable summary."""
    agent = make_chatter(
        name="TestBot",
        temperature=0.5,
        max_history=25,
    )
    desc = agent.describe()

    assert "TestBot" in desc
    assert "0.5" in desc
    assert "25" in desc


# --- warm_up ---


@patch.object(LLMEngine, "generate")
def test_warm_up_calls_generate(mock_generate):
    """warm_up() sends a trivial prompt to pre-load model."""
    mock_generate.return_value = {
        "response": "",
        "metrics": DUMMY_METRICS,
    }
    agent = make_chatter()
    agent.warm_up()

    mock_generate.assert_called_once_with(prompt="ping", max_tokens=1)


def test_warm_up_noop_for_non_ollama():
    """warm_up() is a no-op for non-Ollama providers."""
    from helpers import make_engine

    engine = make_engine()
    engine.provider = "openai_compat"
    agent = make_chatter(engine=engine)
    # Should not raise, and generate should NOT be called
    agent.warm_up()
    engine.llm_provider.generate.assert_not_called()


# --- Edge cases ---


@patch.object(LLMEngine, "chat")
def test_chat_after_reset_sends_no_history(mock_chat):
    """chat() after reset() sends only system+user, no prior turns."""
    mock_chat.return_value = chat_response_dict("OK")
    agent = make_chatter(system_prompt="Be helpful.")

    # First turn builds history
    agent.chat("First message")
    agent.reset()
    assert agent.history == []

    # Second turn after reset should only see system + new message
    agent.chat("Fresh start")
    messages = mock_chat.call_args.kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "Fresh start"


@patch.object(LLMEngine, "chat")
def test_max_history_zero_keeps_no_turns(mock_chat):
    """max_history=0 means history is always empty after each turn."""
    mock_chat.return_value = chat_response_dict("R")
    agent = make_chatter(max_history=0)

    agent.chat("A")
    agent.chat("B")

    assert agent.history == []


@patch.object(LLMEngine, "chat")
def test_max_history_zero_engine_sees_only_system(mock_chat):
    """With max_history=0, trim fires before engine call so only system sent.

    _append_user() trims immediately after appending, which with
    max_history=0 removes the just-added user message too. The engine
    therefore only sees the system prompt.
    """
    mock_chat.return_value = chat_response_dict("R")
    agent = make_chatter(max_history=0)

    agent.chat("First")
    agent.chat("Second")

    for call in mock_chat.call_args_list:
        messages = call.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "system"


@patch.object(LLMEngine, "chat")
def test_conversation_id_isolation_in_memory(mock_chat, tmp_path):
    """Two Chatters with different conversation_ids keep separate
    histories."""
    from agents7.agents.memory import MemoryConfig, create_memory

    mock_chat.return_value = chat_response_dict("R")
    mem = create_memory(
        MemoryConfig(backend="sqlite", path=str(tmp_path / "m.db"))
    )

    agent_a = make_chatter(
        memory=mem, conversation_id="conv-a", system_prompt="A"
    )
    agent_b = make_chatter(
        memory=mem, conversation_id="conv-b", system_prompt="B"
    )

    agent_a.chat("Hello from A")
    agent_b.chat("Hello from B")

    assert len(agent_a.history) == 2
    assert len(agent_b.history) == 2
    assert agent_a.history[0]["content"] == "Hello from A"
    assert agent_b.history[0]["content"] == "Hello from B"


@patch.object(LLMEngine, "chat")
def test_chat_very_long_message(mock_chat):
    """Very long user messages are passed through unchanged."""
    long_msg = "x" * 10_000
    mock_chat.return_value = chat_response_dict("OK")
    agent = make_chatter()

    agent.chat(long_msg)

    messages = mock_chat.call_args.kwargs["messages"]
    assert messages[-1]["content"] == long_msg
