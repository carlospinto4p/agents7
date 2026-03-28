"""Pluggable memory backends for Chatter conversation history.

Provides two backends:
- `JsonFileMemory`: JSON file storage (human-readable)
- `SqliteMemory`: SQLite-backed persistent storage

Usage::

    from agents7.agents.memory import MemoryConfig, create_memory

    memory = create_memory(MemoryConfig(
        backend="json", path="history.json"
    ))
    chatter = Chatter(..., memory=memory, conversation_id="session-1")
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from agents7.exceptions import AgentConfigurationError


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for Chatter conversation memory.

    Attributes:
        backend: Memory backend. ``"json"`` for JSON file
            storage, ``"sqlite"`` for SQLite database.
        path: File path for storage. Defaults to
            ``"memory.json"`` for JSON, ``"memory.db"``
            for SQLite.
    """

    backend: str = "json"
    path: str = field(default="", kw_only=True)

    def __post_init__(self) -> None:
        if self.backend not in ("json", "sqlite"):
            raise AgentConfigurationError(
                f"memory backend must be 'json' or "
                f"'sqlite', got '{self.backend}'"
            )
        if not self.path:
            default = "memory.json" if self.backend == "json" else "memory.db"
            object.__setattr__(self, "path", default)


def create_memory(
    config: MemoryConfig,
) -> "JsonFileMemory | SqliteMemory":
    """Create a memory backend from configuration.

    :param config: Memory configuration.
    :return: A memory backend instance.
    """
    if config.backend == "sqlite":
        return SqliteMemory(config.path)
    return JsonFileMemory(config.path)


class MemoryBackend(ABC):
    """Abstract base class for conversation memory."""

    @abstractmethod
    def load(self, conversation_id: str) -> list[dict[str, str]]:
        """Load conversation history.

        :param conversation_id: Conversation identifier.
        :return: List of message dicts with 'role' and
            'content' keys. Empty list if no history.
        """

    @abstractmethod
    def save(
        self,
        conversation_id: str,
        messages: list[dict[str, str]],
    ) -> None:
        """Save conversation history (full replace).

        :param conversation_id: Conversation identifier.
        :param messages: Complete message list to persist.
        """

    @abstractmethod
    def clear(self, conversation_id: str) -> None:
        """Clear history for a conversation.

        :param conversation_id: Conversation identifier.
        """

    @abstractmethod
    def list_conversations(self) -> list[str]:
        """List all stored conversation IDs.

        :return: List of conversation ID strings.
        """


class JsonFileMemory(MemoryBackend):
    """JSON file-backed conversation memory.

    Stores all conversations in a single JSON file as::

        {
            "conversation_id": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"}
            ]
        }

    Thread-safe via a lock. Creates the file on first
    write.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def load(self, conversation_id: str) -> list[dict[str, str]]:
        """Load conversation from JSON file."""
        with self._lock:
            data = self._read_all()
            return data.get(conversation_id, [])

    def save(
        self,
        conversation_id: str,
        messages: list[dict[str, str]],
    ) -> None:
        """Save conversation to JSON file."""
        with self._lock:
            data = self._read_all()
            data[conversation_id] = messages
            self._write_all(data)

    def clear(self, conversation_id: str) -> None:
        """Remove a conversation from the JSON file."""
        with self._lock:
            data = self._read_all()
            data.pop(conversation_id, None)
            self._write_all(data)

    def list_conversations(self) -> list[str]:
        """List all conversation IDs in the file."""
        with self._lock:
            data = self._read_all()
            return list(data.keys())

    def _read_all(self) -> dict:
        """Read the entire JSON file."""
        if not self._path.exists():
            return {}
        with self._path.open(encoding="utf-8") as f:
            return json.load(f)

    def _write_all(self, data: dict) -> None:
        """Write the entire JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class SqliteMemory(MemoryBackend):
    """SQLite-backed conversation memory.

    Stores messages in a ``messages`` table with columns:
    ``conversation_id``, ``position``, ``role``, ``content``.

    Thread-safe via a lock. Creates the database and table
    on first use.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            "  conversation_id TEXT NOT NULL,"
            "  position INTEGER NOT NULL,"
            "  role TEXT NOT NULL,"
            "  content TEXT NOT NULL,"
            "  PRIMARY KEY (conversation_id, position)"
            ")"
        )
        self._conn.commit()

    def load(self, conversation_id: str) -> list[dict[str, str]]:
        """Load conversation from SQLite."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages "
                "WHERE conversation_id = ? "
                "ORDER BY position",
                (conversation_id,),
            ).fetchall()
            return [
                {"role": role, "content": content} for role, content in rows
            ]

    def save(
        self,
        conversation_id: str,
        messages: list[dict[str, str]],
    ) -> None:
        """Save conversation to SQLite (full replace)."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            self._conn.executemany(
                "INSERT INTO messages "
                "(conversation_id, position, role, content) "
                "VALUES (?, ?, ?, ?)",
                [
                    (
                        conversation_id,
                        i,
                        msg["role"],
                        msg["content"],
                    )
                    for i, msg in enumerate(messages)
                ],
            )
            self._conn.commit()

    def clear(self, conversation_id: str) -> None:
        """Remove a conversation from SQLite."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            self._conn.commit()

    def list_conversations(self) -> list[str]:
        """List all conversation IDs in the database."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT conversation_id "
                "FROM messages ORDER BY conversation_id"
            ).fetchall()
            return [row[0] for row in rows]
