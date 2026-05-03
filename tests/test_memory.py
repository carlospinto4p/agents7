"""Unit tests for the memory backends (no LLM required)."""

import json
import threading

import pytest

from agents7.agents.memory import (
    JsonFileMemory,
    MemoryConfig,
    SqliteMemory,
    create_memory,
)
from agents7.exceptions import AgentConfigurationError


MESSAGES = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]


# --- MemoryConfig ---


def test_config_defaults_json():
    """JSON backend gets default path."""
    config = MemoryConfig(backend="json")
    assert config.path == "memory/"


def test_config_defaults_sqlite():
    """SQLite backend gets default path."""
    config = MemoryConfig(backend="sqlite")
    assert config.path == "memory.db"


def test_config_custom_path():
    """Custom path is preserved."""
    config = MemoryConfig(backend="json", path="custom.json")
    assert config.path == "custom.json"


def test_config_invalid_backend():
    """Invalid backend raises AgentConfigurationError."""
    with pytest.raises(AgentConfigurationError, match="memory backend"):
        MemoryConfig(backend="redis")


# --- create_memory ---


def test_create_memory_json(tmp_path):
    """create_memory returns JsonFileMemory for json."""
    config = MemoryConfig(backend="json", path=str(tmp_path / "test"))
    memory = create_memory(config)
    assert isinstance(memory, JsonFileMemory)


def test_create_memory_sqlite(tmp_path):
    """create_memory returns SqliteMemory for sqlite."""
    config = MemoryConfig(backend="sqlite", path=str(tmp_path / "test.db"))
    memory = create_memory(config)
    assert isinstance(memory, SqliteMemory)


# --- JsonFileMemory ---


def test_json_save_and_load(tmp_path):
    """Save and load round-trips correctly."""
    mem = JsonFileMemory(str(tmp_path / "history"))
    mem.save("conv1", MESSAGES)
    loaded = mem.load("conv1")
    assert loaded == MESSAGES


def test_json_load_empty(tmp_path):
    """Loading non-existent conversation returns empty list."""
    mem = JsonFileMemory(str(tmp_path / "history"))
    assert mem.load("nonexistent") == []


def test_json_load_no_file(tmp_path):
    """Loading when directory doesn't exist returns empty list."""
    mem = JsonFileMemory(str(tmp_path / "missing"))
    assert mem.load("any") == []


def test_json_multiple_conversations(tmp_path):
    """Multiple conversations are stored independently."""
    mem = JsonFileMemory(str(tmp_path / "history"))
    mem.save("conv1", MESSAGES)
    mem.save("conv2", [{"role": "user", "content": "Bye"}])

    assert len(mem.load("conv1")) == 2
    assert len(mem.load("conv2")) == 1


def test_json_clear(tmp_path):
    """Clear removes a specific conversation."""
    mem = JsonFileMemory(str(tmp_path / "history"))
    mem.save("conv1", MESSAGES)
    mem.save("conv2", MESSAGES)
    mem.clear("conv1")

    assert mem.load("conv1") == []
    assert len(mem.load("conv2")) == 2


def test_json_list_conversations(tmp_path):
    """list_conversations returns all stored IDs."""
    mem = JsonFileMemory(str(tmp_path / "history"))
    mem.save("alpha", MESSAGES)
    mem.save("beta", MESSAGES)

    convs = mem.list_conversations()
    assert sorted(convs) == ["alpha", "beta"]


def test_json_list_conversations_empty(tmp_path):
    """list_conversations returns empty list when directory missing."""
    mem = JsonFileMemory(str(tmp_path / "missing"))
    assert mem.list_conversations() == []


def test_json_overwrite(tmp_path):
    """Saving to same ID replaces previous history."""
    mem = JsonFileMemory(str(tmp_path / "history"))
    mem.save("conv1", MESSAGES)
    new_msgs = [{"role": "user", "content": "New"}]
    mem.save("conv1", new_msgs)
    assert mem.load("conv1") == new_msgs


def test_json_creates_parent_dirs(tmp_path):
    """Save creates parent directories if needed."""
    path = tmp_path / "sub" / "dir" / "history"
    mem = JsonFileMemory(str(path))
    mem.save("conv1", MESSAGES)
    assert mem.load("conv1") == MESSAGES


def test_json_load_corrupted_file(tmp_path):
    """Loading a corrupted JSON file raises JSONDecodeError."""
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    (history_dir / "conv1.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    mem = JsonFileMemory(str(history_dir))
    with pytest.raises(json.JSONDecodeError):
        mem.load("conv1")


def test_json_conv_path_dotdot_sanitized(tmp_path):
    """conversation_id '.' and '..' map to '_' to avoid special names."""
    mem = JsonFileMemory(str(tmp_path / "history"))
    mem.save(".", MESSAGES)
    mem.save("..", MESSAGES)
    convs = mem.list_conversations()
    assert "." not in convs
    assert ".." not in convs
    assert "_" in convs


def test_json_conv_path_traversal_chars_replaced(tmp_path):
    """Path-separator characters in conversation_id are replaced."""
    mem = JsonFileMemory(str(tmp_path / "history"))
    mem.save("../evil", MESSAGES)
    loaded = mem.load("../evil")
    assert loaded == MESSAGES
    # Verify the file is a direct child of the history dir
    history_dir = tmp_path / "history"
    files = list(history_dir.iterdir())
    assert len(files) == 1
    assert files[0].parent == history_dir


# --- SqliteMemory ---


def test_sqlite_save_and_load(tmp_path):
    """Save and load round-trips correctly."""
    mem = SqliteMemory(str(tmp_path / "history.db"))
    mem.save("conv1", MESSAGES)
    loaded = mem.load("conv1")
    assert loaded == MESSAGES


def test_sqlite_load_empty(tmp_path):
    """Loading non-existent conversation returns empty list."""
    mem = SqliteMemory(str(tmp_path / "history.db"))
    assert mem.load("nonexistent") == []


def test_sqlite_multiple_conversations(tmp_path):
    """Multiple conversations are stored independently."""
    mem = SqliteMemory(str(tmp_path / "history.db"))
    mem.save("conv1", MESSAGES)
    mem.save("conv2", [{"role": "user", "content": "Bye"}])

    assert len(mem.load("conv1")) == 2
    assert len(mem.load("conv2")) == 1


def test_sqlite_clear(tmp_path):
    """Clear removes a specific conversation."""
    mem = SqliteMemory(str(tmp_path / "history.db"))
    mem.save("conv1", MESSAGES)
    mem.save("conv2", MESSAGES)
    mem.clear("conv1")

    assert mem.load("conv1") == []
    assert len(mem.load("conv2")) == 2


def test_sqlite_list_conversations(tmp_path):
    """list_conversations returns all stored IDs."""
    mem = SqliteMemory(str(tmp_path / "history.db"))
    mem.save("alpha", MESSAGES)
    mem.save("beta", MESSAGES)

    convs = mem.list_conversations()
    assert convs == ["alpha", "beta"]


def test_sqlite_list_conversations_empty(tmp_path):
    """list_conversations returns empty list when no data."""
    mem = SqliteMemory(str(tmp_path / "history.db"))
    assert mem.list_conversations() == []


def test_sqlite_overwrite(tmp_path):
    """Saving to same ID replaces previous history."""
    mem = SqliteMemory(str(tmp_path / "history.db"))
    mem.save("conv1", MESSAGES)
    new_msgs = [{"role": "user", "content": "New"}]
    mem.save("conv1", new_msgs)
    assert mem.load("conv1") == new_msgs


def test_sqlite_preserves_order(tmp_path):
    """Messages are loaded in the order they were saved."""
    mem = SqliteMemory(str(tmp_path / "history.db"))
    msgs = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Second"},
        {"role": "user", "content": "Third"},
        {"role": "assistant", "content": "Fourth"},
    ]
    mem.save("conv1", msgs)
    assert mem.load("conv1") == msgs


# --- SqliteMemory concurrent ---


def test_sqlite_concurrent_read_write(tmp_path):
    """Concurrent saves from multiple threads don't corrupt data."""
    mem = SqliteMemory(str(tmp_path / "history.db"))
    errors: list[str] = []

    def writer(conv_id: str) -> None:
        msgs = [{"role": "user", "content": conv_id}]
        try:
            mem.save(conv_id, msgs)
            loaded = mem.load(conv_id)
            if loaded != msgs:
                errors.append(f"{conv_id}: got {loaded}")
        except Exception as e:
            errors.append(str(e))

    threads = [
        threading.Thread(target=writer, args=(f"conv{i}",)) for i in range(10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(mem.list_conversations()) == 10
