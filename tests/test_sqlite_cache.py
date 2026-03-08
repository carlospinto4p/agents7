"""Unit tests for SqliteCache backend."""

from agents7.llm.cache import CacheConfig, SqliteCache, create_cache


# --- CacheConfig backend validation ---


def test_config_default_backend():
    """Default backend is memory."""
    config = CacheConfig()
    assert config.backend == "memory"


def test_config_sqlite_backend():
    """sqlite backend is accepted."""
    config = CacheConfig(backend="sqlite", path="test.db")
    assert config.backend == "sqlite"


def test_config_invalid_backend():
    """Invalid backend raises LLMException."""
    import pytest
    from agents7.exceptions import LLMException

    with pytest.raises(LLMException):
        CacheConfig(backend="redis")


# --- create_cache factory ---


def test_create_cache_memory():
    """Memory backend returns ResponseCache."""
    from agents7.llm.cache import ResponseCache

    config = CacheConfig(backend="memory")
    cache = create_cache(config)
    assert isinstance(cache, ResponseCache)


def test_create_cache_sqlite(tmp_path):
    """SQLite backend returns SqliteCache."""
    config = CacheConfig(backend="sqlite", path=str(tmp_path / "test.db"))
    cache = create_cache(config)
    assert isinstance(cache, SqliteCache)


# --- SqliteCache operations ---


def test_put_and_get(tmp_path):
    """Store and retrieve a value."""
    config = CacheConfig(backend="sqlite", path=str(tmp_path / "c.db"))
    cache = SqliteCache(config)
    cache.put("k1", {"response": "hello"})
    assert cache.get("k1") == {"response": "hello"}


def test_miss_returns_none(tmp_path):
    """Missing key returns None."""
    config = CacheConfig(backend="sqlite", path=str(tmp_path / "c.db"))
    cache = SqliteCache(config)
    assert cache.get("missing") is None


def test_size(tmp_path):
    """Size tracks entry count."""
    config = CacheConfig(backend="sqlite", path=str(tmp_path / "c.db"))
    cache = SqliteCache(config)
    assert cache.size == 0
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    assert cache.size == 2


def test_clear(tmp_path):
    """Clear removes all entries."""
    config = CacheConfig(backend="sqlite", path=str(tmp_path / "c.db"))
    cache = SqliteCache(config)
    cache.put("k1", "v1")
    cache.clear()
    assert cache.size == 0


def test_lru_eviction(tmp_path):
    """Entries are evicted when max_size exceeded."""
    config = CacheConfig(
        max_size=2,
        backend="sqlite",
        path=str(tmp_path / "c.db"),
    )
    cache = SqliteCache(config)
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    cache.put("k3", "v3")  # should evict k1
    assert cache.size == 2
    assert cache.get("k1") is None
    assert cache.get("k3") == "v3"


def test_overwrite_existing(tmp_path):
    """Overwriting a key updates the value."""
    config = CacheConfig(backend="sqlite", path=str(tmp_path / "c.db"))
    cache = SqliteCache(config)
    cache.put("k1", "v1")
    cache.put("k1", "v2")
    assert cache.get("k1") == "v2"
    assert cache.size == 1


def test_persistence(tmp_path):
    """Data persists across instances."""
    db_path = str(tmp_path / "persist.db")
    config = CacheConfig(backend="sqlite", path=db_path)

    cache1 = SqliteCache(config)
    cache1.put("k1", {"data": "persisted"})

    cache2 = SqliteCache(config)
    assert cache2.get("k1") == {"data": "persisted"}


def test_make_key_shared():
    """SqliteCache shares make_key with ResponseCache."""
    key = SqliteCache.make_key(
        model_id="ollama::qwen3",
        method="chat",
    )
    assert isinstance(key, str)
    assert len(key) == 64  # SHA-256 hex
