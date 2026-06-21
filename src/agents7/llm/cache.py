"""Response cache backends for deterministic LLM calls.

Provides two backends:
- `ResponseCache`: in-memory LRU (default)
- `SqliteCache`: SQLite-backed persistent cache
"""

import hashlib
import json
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from agents7.exceptions import LLMException


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for LLM response caching.

    Attributes:
        max_size: Maximum number of cached entries.
            0 = unlimited.
        backend: Cache backend. ``"memory"`` for in-memory
            LRU, ``"sqlite"`` for persistent SQLite storage.
        path: File path for SQLite backend. Ignored when
            ``backend="memory"``. Defaults to
            ``"cache.db"``.
    """

    max_size: int = 128
    backend: str = field(default="memory", kw_only=True)
    path: str = field(default="cache.db", kw_only=True)

    def __post_init__(self) -> None:
        if self.max_size < 0:
            raise LLMException(f"max_size must be >= 0, got {self.max_size}")
        if self.backend not in ("memory", "sqlite"):
            raise LLMException(
                f"backend must be 'memory' or 'sqlite', got '{self.backend}'"
            )


def create_cache(config: CacheConfig) -> "ResponseCache | SqliteCache":
    """Create a cache instance from configuration.

    :param config: Cache configuration.
    :return: A `ResponseCache` or `SqliteCache` instance.
    """
    if config.backend == "sqlite":
        return SqliteCache(config)
    return ResponseCache(config)


class _CacheKeyBuilder:
    """Shared cache key generation logic."""

    @staticmethod
    def make_key(
        model_id: str,
        method: str,
        messages: list[dict[str, str]] | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = -1,
        output_format: str | dict | None = None,
        think: bool = False,
        tools: list[dict] | None = None,
    ) -> str:
        """Build a deterministic cache key from call
        parameters.

        Returns a hex SHA-256 digest.
        """
        payload = {
            "model_id": model_id,
            "method": method,
            "messages": messages,
            "prompt": prompt,
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
            "output_format": output_format,
            "think": think,
        }
        if tools is not None:
            payload["tools"] = tools
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()


class ResponseCache(_CacheKeyBuilder):
    """Thread-safe in-memory LRU cache for LLM responses.

    Entries are evicted in least-recently-used order when
    ``max_size > 0`` and the limit is exceeded.
    """

    def __init__(self, config: CacheConfig):
        self._max_size = config.max_size
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Return cached value or ``None`` on miss.

        On hit the entry is moved to the end
        (most-recently-used).
        """
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)
            return self._store[key]

    def put(self, key: str, value: Any) -> None:
        """Store *value* under *key*, evicting LRU if
        needed."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = value
                return
            self._store[key] = value
            if self._max_size > 0 and len(self._store) > self._max_size:
                self._store.popitem(last=False)

    @property
    def size(self) -> int:
        """Number of entries currently in the cache."""
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._store.clear()


class SqliteCache(_CacheKeyBuilder):
    """Thread-safe SQLite-backed persistent cache.

    Stores responses as JSON in a local SQLite database.
    Survives process restarts. LRU eviction is applied
    on ``put()`` when ``max_size > 0``.
    """

    def __init__(self, config: CacheConfig):
        self._max_size = config.max_size
        self._path = config.path
        self._lock = threading.Lock()
        # check_same_thread=False is safe here: every public method
        # acquires self._lock before touching the connection, so
        # concurrent access is serialised at the Python level.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  key TEXT PRIMARY KEY,"
            "  value TEXT NOT NULL,"
            "  accessed INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        self._conn.commit()
        self._approx_size: int = self._conn.execute(
            "SELECT COUNT(*) FROM cache"
        ).fetchone()[0]

    def get(self, key: str) -> Any | None:
        """Return cached value or ``None`` on miss."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value, accessed FROM cache WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE cache SET accessed = accessed + 1 WHERE key = ?",
                (key,),
            )
            self._conn.commit()
            return json.loads(row[0])

    def put(self, key: str, value: Any) -> None:
        """Store *value* under *key*, evicting LRU if
        needed."""
        serialized = json.dumps(value, ensure_ascii=False)
        with self._lock:
            exists = (
                self._conn.execute(
                    "SELECT 1 FROM cache WHERE key = ?", (key,)
                ).fetchone()
                is not None
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO cache "
                "(key, value, accessed) VALUES (?, ?, 0)",
                (key, serialized),
            )
            if not exists:
                self._approx_size += 1
                if self._max_size > 0 and self._approx_size > self._max_size:
                    excess = self._approx_size - self._max_size
                    self._conn.execute(
                        "DELETE FROM cache WHERE key IN ("
                        "  SELECT key FROM cache "
                        "  ORDER BY accessed ASC LIMIT ?"
                        ")",
                        (excess,),
                    )
                    self._approx_size = self._max_size
            self._conn.commit()

    @property
    def size(self) -> int:
        """Number of entries currently in the cache."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
            return row[0]

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._conn.execute("DELETE FROM cache")
            self._conn.commit()
            self._approx_size = 0
