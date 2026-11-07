"""Unit tests for the agent-opens-the-chat turn."""

from unittest.mock import AsyncMock, patch

import pytest

from agents7.agents.chatter import (
    _DEFAULT_OPENING_PROMPT,
    Chatter,
)
from agents7.agents.context import ContextParam
from agents7.exceptions import (
    AgentConfigurationError,
    ChatEndedError,
)
from agents7.llm import LLMEngine
from agents7.llm.streaming import AsyncStreamResult, StreamResult

from helpers import (
    async_iter,
    chat_chunk,
    chat_response_dict,
    make_chatter,
)


def opening_stream_chunks(text: str = "¿Qué tal?"):
    """Build the chunks of a finished one-message stream."""
    return [
        chat_chunk(content=text),
        chat_chunk(
            done=True,
            eval_count=10,
            prompt_eval_count=5,
            total_duration=100_000_000,
        ),
    ]


# --- can_open gate ---


def test_can_open_defaults_to_false():
    """An agent may not speak first unless told it may."""
    assert make_chatter().can_open is False


def test_open_chat_refused_without_can_open():
    """open_chat() raises when the agent may not open."""
    agent = make_chatter()
    with pytest.raises(AgentConfigurationError, match="can_open"):
        agent.open_chat()


@pytest.mark.asyncio
async def test_open_chat_async_refused_without_can_open():
    """open_chat_async() raises when the agent may not open."""
    agent = make_chatter()
    with pytest.raises(AgentConfigurationError, match="can_open"):
        await agent.open_chat_async()


def test_open_chat_stream_refused_without_can_open():
    """open_chat_stream() raises when the agent may not open."""
    agent = make_chatter()
    with pytest.raises(AgentConfigurationError, match="can_open"):
        agent.open_chat_stream()


@pytest.mark.asyncio
async def test_open_chat_stream_async_refused_without_can_open():
    """open_chat_stream_async() raises without can_open."""
    agent = make_chatter()
    with pytest.raises(AgentConfigurationError, match="can_open"):
        await agent.open_chat_stream_async()


@patch.object(LLMEngine, "chat")
def test_open_chat_refused_when_ended(mock_chat):
    """A finished conversation cannot be opened again."""
    mock_chat.return_value = chat_response_dict()
    agent = make_chatter(can_open=True, max_turns=1)
    agent.open_chat()

    with pytest.raises(ChatEndedError):
        agent.open_chat()


# --- open_chat() ---


@patch.object(LLMEngine, "chat")
def test_open_chat_returns_the_agents_line(mock_chat):
    """open_chat() returns what the agent wrote."""
    mock_chat.return_value = chat_response_dict("¡Hola! ¿Qué tal?")
    agent = make_chatter(can_open=True)

    response = agent.open_chat()

    assert response.response == "¡Hola! ¿Qué tal?"


@patch.object(LLMEngine, "chat")
def test_open_chat_history_starts_with_the_agent(mock_chat):
    """Only the agent's line is stored — not the instruction."""
    mock_chat.return_value = chat_response_dict("¡Hola!")
    agent = make_chatter(can_open=True)

    agent.open_chat()

    assert agent.history == [
        {"role": "assistant", "content": "¡Hola!"},
    ]
    assert agent.turns == 1


@patch.object(LLMEngine, "chat")
def test_open_chat_asks_the_model_for_a_first_message(mock_chat):
    """The instruction is sent as the last user message."""
    mock_chat.return_value = chat_response_dict()
    agent = make_chatter(can_open=True)

    agent.open_chat()

    messages = mock_chat.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[-1] == {
        "role": "user",
        "content": _DEFAULT_OPENING_PROMPT,
    }
    assert len(messages) == 2


@patch.object(LLMEngine, "chat")
def test_open_chat_uses_a_custom_opening_prompt(mock_chat):
    """`opening_prompt` replaces the built-in instruction."""
    mock_chat.return_value = chat_response_dict()
    agent = make_chatter(
        can_open=True,
        opening_prompt="[Escribe el primer mensaje.]",
    )

    agent.open_chat()

    messages = mock_chat.call_args.kwargs["messages"]
    assert messages[-1]["content"] == "[Escribe el primer mensaje.]"


@patch.object(LLMEngine, "chat")
def test_chat_after_open_chat_continues_the_transcript(mock_chat):
    """A reply to the opening line lands after it."""
    mock_chat.return_value = chat_response_dict("¡Hola!")
    agent = make_chatter(can_open=True)
    agent.open_chat()

    mock_chat.return_value = chat_response_dict("Bien, ¿y tú?")
    agent.chat("Hola, ¿qué tal?")

    assert [m["role"] for m in agent.history] == [
        "assistant",
        "user",
        "assistant",
    ]


@patch.object(LLMEngine, "chat")
def test_open_chat_applies_typos(mock_chat):
    """The opening turn goes through the typo jitter."""
    mock_chat.return_value = chat_response_dict("hola " * 40)
    agent = make_chatter(can_open=True, typo_rate=1.0)

    response = agent.open_chat()

    assert response.response != "hola " * 40


@patch.object(LLMEngine, "chat")
def test_open_chat_carries_the_conversation_context(mock_chat):
    """Who the agent is opening with reaches the first turn.

    The opening turn is the one that most needs the context —
    it is written before the other person has given anything
    away — so the resolved system prompt must already hold it.
    """
    mock_chat.return_value = chat_response_dict()
    agent = make_chatter(
        system_prompt="Hablas con $nick. Es de $moment.",
        context_params=(
            ContextParam(name="nick", default="alguien"),
            ContextParam(name="moment", resolver=lambda: "tarde"),
        ),
        can_open=True,
    )
    agent.set_context(nick="Monica28")

    agent.open_chat()

    system = mock_chat.call_args.kwargs["messages"][0]["content"]
    assert system == "Hablas con Monica28. Es de tarde."


def test_describe_reports_that_the_agent_may_open():
    """describe() says so when the agent can speak first."""
    assert "Can open a chat: yes" in make_chatter(can_open=True).describe()
    assert "Can open a chat" not in make_chatter().describe()


# --- open_chat_async() ---


@pytest.mark.asyncio
@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
async def test_open_chat_async_returns_the_agents_line(mock_chat):
    """open_chat_async() returns what the agent wrote."""
    mock_chat.return_value = chat_response_dict("¡Hola!")
    agent = make_chatter(can_open=True)

    response = await agent.open_chat_async()

    assert response.response == "¡Hola!"
    assert agent.history == [
        {"role": "assistant", "content": "¡Hola!"},
    ]


@pytest.mark.asyncio
@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
async def test_open_chat_async_sends_the_instruction(mock_chat):
    """The async opening turn carries the same instruction."""
    mock_chat.return_value = chat_response_dict()
    agent = make_chatter(can_open=True)

    await agent.open_chat_async()

    messages = mock_chat.call_args.kwargs["messages"]
    assert messages[-1]["content"] == _DEFAULT_OPENING_PROMPT


# --- open_chat_stream() ---


@patch.object(LLMEngine, "chat_stream")
def test_open_chat_stream_yields_chunks(mock_stream):
    """open_chat_stream() streams the opening line."""
    mock_stream.return_value = StreamResult(
        iter(opening_stream_chunks()), "chat"
    )
    agent = make_chatter(can_open=True)

    result = agent.open_chat_stream()
    collected = list(result)

    assert collected == ["¿Qué tal?"]
    assert result.chat_response.response == "¿Qué tal?"


@patch.object(LLMEngine, "chat_stream")
def test_open_chat_stream_history_starts_with_the_agent(mock_stream):
    """Streaming an opening turn stores only the agent's line."""
    mock_stream.return_value = StreamResult(
        iter(opening_stream_chunks("¡Hola!")), "chat"
    )
    agent = make_chatter(can_open=True)

    list(agent.open_chat_stream())

    assert agent.history == [
        {"role": "assistant", "content": "¡Hola!"},
    ]


@pytest.mark.asyncio
@patch.object(LLMEngine, "chat_stream_async", new_callable=AsyncMock)
async def test_open_chat_stream_async_yields_chunks(mock_stream):
    """open_chat_stream_async() streams the opening line."""
    mock_stream.return_value = AsyncStreamResult(
        async_iter(opening_stream_chunks()), "chat"
    )
    agent = make_chatter(can_open=True)

    result = await agent.open_chat_stream_async()
    collected = [chunk async for chunk in result]

    assert collected == ["¿Qué tal?"]
    assert agent.history == [
        {"role": "assistant", "content": "¿Qué tal?"},
    ]


# --- YAML configuration ---


def test_from_yaml_reads_can_open_and_opening_prompt(tmp_path):
    """Both fields can be set from a config file."""
    config = tmp_path / "opener.yml"
    config.write_text(
        "name: opener\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3:4b\n"
        "system_prompt: You are David.\n"
        "can_open: true\n"
        "opening_prompt: '[Escribe el primer mensaje.]'\n",
        encoding="utf-8",
    )

    agent = Chatter.from_yaml(str(config))

    assert agent.can_open is True
    assert agent.opening_prompt == "[Escribe el primer mensaje.]"


def test_from_yaml_defaults_to_not_opening(tmp_path):
    """A config that says nothing leaves the agent answering."""
    config = tmp_path / "quiet.yml"
    config.write_text(
        "name: quiet\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3:4b\n"
        "system_prompt: You are David.\n",
        encoding="utf-8",
    )

    assert Chatter.from_yaml(str(config)).can_open is False


# --- opening_instructions ---

OPENING_LINES = (
    "## How David opens\n\nDavid: ¡Buenas! ¿Qué tal?\nDavid: Toc toc."
)


@patch.object(LLMEngine, "chat")
def test_opening_instructions_reach_the_opening_turn(mock_chat):
    """The written opening lines are in the opening prompt."""
    mock_chat.return_value = chat_response_dict()
    agent = make_chatter(
        can_open=True,
        opening_instructions=OPENING_LINES,
    )

    agent.open_chat()

    system = mock_chat.call_args.kwargs["messages"][0]["content"]
    assert "Toc toc." in system


@patch.object(LLMEngine, "chat")
def test_opening_instructions_stay_out_of_later_turns(mock_chat):
    """The opening lines are gone once the chat is under way.

    The regression this guards: a bank of opening lines the
    model can read on every turn is a bank it can copy on
    every turn, so a character who has already said goodbye
    greets the same person again from the top.
    """
    mock_chat.return_value = chat_response_dict()
    agent = make_chatter(
        can_open=True,
        opening_instructions=OPENING_LINES,
    )

    agent.open_chat()
    agent.chat("hola")

    system = mock_chat.call_args.kwargs["messages"][0]["content"]
    assert "Toc toc." not in system


@patch.object(LLMEngine, "chat")
def test_opening_instructions_are_absent_by_default(mock_chat):
    """An agent that declares none sends the prompt unchanged."""
    mock_chat.return_value = chat_response_dict()
    plain = make_chatter(can_open=True)

    plain.open_chat()
    without = mock_chat.call_args.kwargs["messages"][0]["content"]

    assert plain.opening_instructions == ""
    assert without == plain.resolved_system_prompt


def test_opening_instructions_are_stripped():
    """A YAML block scalar's trailing newline is dropped."""
    agent = make_chatter(
        can_open=True,
        opening_instructions="\nToc toc.\n\n",
    )

    assert agent.opening_instructions == "Toc toc."


@pytest.mark.asyncio
@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
async def test_opening_instructions_reach_the_async_turn(mock_chat):
    """`open_chat_async` carries them too."""
    mock_chat.return_value = chat_response_dict()
    agent = make_chatter(
        can_open=True,
        opening_instructions=OPENING_LINES,
    )

    await agent.open_chat_async()

    system = mock_chat.call_args.kwargs["messages"][0]["content"]
    assert "Toc toc." in system


@patch.object(LLMEngine, "chat_stream")
def test_opening_instructions_reach_the_opening_stream(mock_stream):
    """The streamed opening carries them as well."""
    mock_stream.return_value = StreamResult(
        iter(opening_stream_chunks()), "chat"
    )
    agent = make_chatter(
        can_open=True,
        opening_instructions=OPENING_LINES,
    )

    list(agent.open_chat_stream())

    system = mock_stream.call_args.kwargs["messages"][0]["content"]
    assert "Toc toc." in system


def test_from_yaml_reads_opening_instructions(tmp_path):
    """The field can be set from a config file."""
    config = tmp_path / "opener.yml"
    config.write_text(
        "name: opener\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3:4b\n"
        "system_prompt: You are David.\n"
        "can_open: true\n"
        "opening_instructions: |\n"
        "    David: Toc toc.\n",
        encoding="utf-8",
    )

    agent = Chatter.from_yaml(str(config))

    assert agent.opening_instructions == "David: Toc toc."
