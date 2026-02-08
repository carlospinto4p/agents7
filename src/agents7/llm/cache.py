"""In-memory LRU response cache for deterministic LLM calls."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from agents7.exceptions import LLMException


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for in-memory LRU response cache.

    Attributes:
        max_size: Maximum number of cached entries. 0 = unlimited.
    """

    max_size: int = 128

    def __post_init__(self):
        if self.max_size < 0:
            raise LLMException(f"max_size must be >= 0, got {self.max_size}")


class ResponseCache:
    """Thread-safe in-memory LRU cache for LLM responses.

    Entries are evicted in least-recently-used order when
    `max_size > 0` and the limit is exceeded.
    """

    def __init__(self, config: CacheConfig):
        self._max_size = config.max_size
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

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
    ) -> str:
        """Build a deterministic cache key from call parameters.

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
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        """Return cached value or `None` on miss.

        On hit the entry is moved to the end (most-recently-used).
        """
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)
            return self._store[key]

    def put(self, key: str, value: Any) -> None:
        """Store *value* under *key*, evicting LRU if needed."""
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
