"""Pluggable memory backends for Chatter conversation history.

Provides two backends:
- `JsonFileMemory`: JSON file storage (human-readable)
- `SqliteMemory`: SQLite-backed persistent storage

Usage::

    from agents7.agents.memory import MemoryConfig, create_memory

    memory = create_memory(MemoryConfig(
        backend="json", path="memory/"
    ))
    chatter = Chatter(..., memory=memory, conversation_id="session-1")
"""

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
        path: Path for storage. Defaults to ``"memory/"``
            (directory) for JSON, ``"memory.db"`` for
            SQLite.
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
            default = "memory/" if self.backend == "json" else "memory.db"
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

    Stores each conversation as a separate JSON file in a
    directory::

        memory/
            session-1.json   # [{"role": "user", ...}, ...]
            session-2.json

    Thread-safe via a lock. Creates the directory on first
    write.
    """

    _MAX_CACHE: int = 100

    def __init__(self, path: str) -> None:
        self._dir = Path(path)
        self._lock = threading.Lock()
        self._cache: dict[str, list[dict[str, str]]] = {}

    def load(self, conversation_id: str) -> list[dict[str, str]]:
        """Load conversation from its JSON file."""
        with self._lock:
            if conversation_id in self._cache:
                return self._cache[conversation_id]
            conv_path = self._conv_path(conversation_id)
            if not conv_path.exists():
                return []
            with conv_path.open(encoding="utf-8") as f:
                messages: list[dict[str, str]] = json.load(f)
            self._cache[conversation_id] = messages
            if len(self._cache) > self._MAX_CACHE:
                self._cache.pop(next(iter(self._cache)))
            return messages

    def save(
        self,
        conversation_id: str,
        messages: list[dict[str, str]],
    ) -> None:
        """Save conversation to its JSON file."""
        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            with self._conv_path(conversation_id).open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
            self._cache[conversation_id] = messages
            if len(self._cache) > self._MAX_CACHE:
                self._cache.pop(next(iter(self._cache)))

    def clear(self, conversation_id: str) -> None:
        """Remove a conversation's JSON file."""
        with self._lock:
            self._cache.pop(conversation_id, None)
            self._conv_path(conversation_id).unlink(missing_ok=True)

    def list_conversations(self) -> list[str]:
        """List all conversation IDs."""
        with self._lock:
            if not self._dir.exists():
                return []
            return sorted(p.stem for p in self._dir.glob("*.json"))

    def _conv_path(self, conversation_id: str) -> Path:
        """Return the file path for a conversation."""
        allowed = frozenset(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789-_."
        )
        safe = "".join(c if c in allowed else "_" for c in conversation_id)
        return self._dir / f"{safe}.json"


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
        # check_same_thread=False is safe here: every public method
        # acquires self._lock before touching the connection, so
        # concurrent access is serialised at the Python level.
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
        self._saved_lengths: dict[str, int] = {}

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
        """Save conversation to SQLite.

        Appends only new messages when the list has grown.
        Falls back to full replace when messages are removed
        or on first save for this conversation.
        """
        new_len = len(messages)
        saved_len = self._saved_lengths.get(conversation_id, -1)
        with self._lock:
            if saved_len >= 0 and new_len > saved_len:
                self._conn.executemany(
                    "INSERT INTO messages "
                    "(conversation_id, position, role, content) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        (conversation_id, i, msg["role"], msg["content"])
                        for i, msg in enumerate(
                            messages[saved_len:], saved_len
                        )
                    ],
                )
            else:
                self._conn.execute(
                    "DELETE FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                )
                self._conn.executemany(
                    "INSERT INTO messages "
                    "(conversation_id, position, role, content) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        (conversation_id, i, msg["role"], msg["content"])
                        for i, msg in enumerate(messages)
                    ],
                )
            self._conn.commit()
        self._saved_lengths[conversation_id] = new_len

    def clear(self, conversation_id: str) -> None:
        """Remove a conversation from SQLite."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            self._conn.commit()
        self._saved_lengths.pop(conversation_id, None)

    def list_conversations(self) -> list[str]:
        """List all conversation IDs in the database."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT conversation_id "
                "FROM messages ORDER BY conversation_id"
            ).fetchall()
            return [row[0] for row in rows]
