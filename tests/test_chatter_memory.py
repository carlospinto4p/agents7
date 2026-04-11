"""Integration tests for Chatter + memory backends."""

from unittest.mock import AsyncMock, patch

import pytest

from agents7.agents.chatter import Chatter
from agents7.agents.memory import JsonFileMemory, SqliteMemory
from agents7.llm import LLMEngine

from helpers import ENGINE, chat_response_dict, make_chatter


# --- JSON memory integration ---


@patch.object(LLMEngine, "chat")
def test_json_persists_across_instances(mock_chat, tmp_path):
    """History persists to JSON and loads in new instance."""
    mock_chat.return_value = chat_response_dict("Hello!")
    path = str(tmp_path / "history.json")
    mem = JsonFileMemory(path)

    agent1 = make_chatter(memory=mem, conversation_id="s1")
    agent1.chat("Hi")
    assert len(agent1.history) == 2

    # New instance with same memory and conversation_id
    agent2 = make_chatter(memory=JsonFileMemory(path), conversation_id="s1")
    assert len(agent2.history) == 2
    assert agent2.history[0]["content"] == "Hi"
    assert agent2.history[1]["content"] == "Hello!"


@patch.object(LLMEngine, "chat")
def test_json_separate_conversations(mock_chat, tmp_path):
    """Different conversation_ids are isolated."""
    mock_chat.return_value = chat_response_dict("Reply")
    path = str(tmp_path / "history.json")
    mem = JsonFileMemory(path)

    a1 = make_chatter(memory=mem, conversation_id="conv-a")
    a1.chat("Message A")

    a2 = make_chatter(memory=mem, conversation_id="conv-b")
    a2.chat("Message B")

    assert len(JsonFileMemory(path).load("conv-a")) == 2
    assert len(JsonFileMemory(path).load("conv-b")) == 2


@patch.object(LLMEngine, "chat")
def test_json_reset_clears_backend(mock_chat, tmp_path):
    """reset() clears both in-memory and persisted history."""
    mock_chat.return_value = chat_response_dict("Reply")
    path = str(tmp_path / "history.json")
    mem = JsonFileMemory(path)

    agent = make_chatter(memory=mem, conversation_id="s1")
    agent.chat("Hi")
    assert len(agent.history) == 2

    agent.reset()
    assert agent.history == []
    assert JsonFileMemory(path).load("s1") == []


# --- SQLite memory integration ---


@patch.object(LLMEngine, "chat")
def test_sqlite_persists_across_instances(mock_chat, tmp_path):
    """History persists to SQLite and loads in new instance."""
    mock_chat.return_value = chat_response_dict("Hello!")
    path = str(tmp_path / "history.db")
    mem = SqliteMemory(path)

    agent1 = make_chatter(memory=mem, conversation_id="s1")
    agent1.chat("Hi")

    agent2 = make_chatter(memory=SqliteMemory(path), conversation_id="s1")
    assert len(agent2.history) == 2
    assert agent2.history[0]["content"] == "Hi"


@patch.object(LLMEngine, "chat")
def test_sqlite_reset_clears_backend(mock_chat, tmp_path):
    """reset() clears both in-memory and SQLite history."""
    mock_chat.return_value = chat_response_dict("Reply")
    path = str(tmp_path / "history.db")
    mem = SqliteMemory(path)

    agent = make_chatter(memory=mem, conversation_id="s1")
    agent.chat("Hi")
    agent.reset()

    assert agent.history == []
    assert SqliteMemory(path).load("s1") == []


# --- No memory (default behavior unchanged) ---


@patch.object(LLMEngine, "chat")
def test_no_memory_works_as_before(mock_chat):
    """Chatter without memory works exactly as before."""
    mock_chat.return_value = chat_response_dict("Reply")
    agent = Chatter(
        name="test",
        version=1,
        engine=ENGINE,
        system_prompt="Hi.",
    )
    agent.chat("Hello")
    assert len(agent.history) == 2
    assert agent.memory is None


# --- Streaming + memory ---


@patch.object(LLMEngine, "chat_stream")
def test_stream_persists_to_memory(mock_stream, tmp_path):
    """Streaming chat persists history to memory backend."""
    from agents7.llm.streaming import StreamResult
    from helpers import chat_chunk

    chunks = [
        chat_chunk(content="Streamed"),
        chat_chunk(
            done=True,
            eval_count=10,
            prompt_eval_count=5,
            total_duration=100_000_000,
        ),
    ]
    stream = StreamResult(iter(chunks), "chat")
    mock_stream.return_value = stream

    path = str(tmp_path / "history.json")
    mem = JsonFileMemory(path)
    agent = make_chatter(memory=mem, conversation_id="s1")
    result = agent.chat_stream("Hi")
    list(result)  # Consume the stream

    # Verify persisted
    loaded = JsonFileMemory(path).load("s1")
    assert len(loaded) == 2
    assert loaded[1]["content"] == "Streamed"


# --- Async + memory ---


@pytest.mark.asyncio
@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
async def test_async_persists_to_memory(mock_chat, tmp_path):
    """Async chat persists history to memory backend."""
    mock_chat.return_value = chat_response_dict("Async!")
    path = str(tmp_path / "history.json")
    mem = JsonFileMemory(path)

    agent = make_chatter(memory=mem, conversation_id="s1")
    await agent.chat_async("Hello")

    loaded = JsonFileMemory(path).load("s1")
    assert len(loaded) == 2
    assert loaded[1]["content"] == "Async!"


# --- YAML loading with memory ---


def test_from_yaml_with_memory(tmp_path):
    """YAML config with memory section creates backend."""
    yaml_content = (
        "name: MemChatter\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Hello.\n"
        "memory:\n"
        f"    backend: json\n"
        f"    path: {tmp_path / 'mem.json'}\n"
        "conversation_id: yaml-session\n"
    )
    yaml_file = tmp_path / "chatter.yml"
    yaml_file.write_text(yaml_content)

    agent = Chatter.from_yaml(str(yaml_file))
    assert agent.memory is not None
    assert isinstance(agent.memory, JsonFileMemory)
    assert agent.conversation_id == "yaml-session"


def test_from_yaml_without_memory(tmp_path):
    """YAML config without memory section has no backend."""
    yaml_content = (
        "name: NoMem\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Hello.\n"
    )
    yaml_file = tmp_path / "chatter.yml"
    yaml_file.write_text(yaml_content)

    agent = Chatter.from_yaml(str(yaml_file))
    assert agent.memory is None
    assert agent.conversation_id == "default"
