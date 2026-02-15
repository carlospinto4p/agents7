"""Unit tests for in-memory LRU response cache (no LLM required)."""

from unittest.mock import AsyncMock, Mock

import pytest

from agents7.agents.labeler import Labeler
from agents7.exceptions import LLMException
from agents7.llm.base import LLMEngine
from agents7.llm.cache import CacheConfig, ResponseCache
from agents7.llm.streaming import AsyncStreamResult, StreamResult

from helpers import chat_chunk


def _make_engine(cache: CacheConfig | None = None):
    """Create an LLMEngine with a mocked provider."""
    engine = object.__new__(LLMEngine)
    engine.provider = "ollama"
    engine.model_name = "test-model"
    engine.model_id = "ollama::test-model"
    engine.retry_config = None
    engine._cache = ResponseCache(cache) if cache is not None else None
    engine.llm_provider = Mock()
    return engine


# ===================================================================
# 1. CacheConfig validation
# ===================================================================


# --- CacheConfig ---


def test_defaults():
    """Default max_size is 128."""
    cfg = CacheConfig()
    assert cfg.max_size == 128


def test_custom_max_size():
    """Custom max_size is stored correctly."""
    cfg = CacheConfig(max_size=256)
    assert cfg.max_size == 256


def test_negative_max_size():
    """Negative max_size raises LLMException."""
    with pytest.raises(LLMException, match="max_size"):
        CacheConfig(max_size=-1)


def test_zero_max_size_allowed():
    """max_size=0 means unlimited, no error."""
    cfg = CacheConfig(max_size=0)
    assert cfg.max_size == 0


# ===================================================================
# 2. ResponseCache core operations
# ===================================================================


# --- ResponseCache ---


def test_put_and_get():
    """Stored value can be retrieved."""
    cache = ResponseCache(CacheConfig(max_size=10))
    cache.put("k1", {"response": "hello"})
    assert cache.get("k1") == {"response": "hello"}


def test_miss_returns_none():
    """Cache miss returns None."""
    cache = ResponseCache(CacheConfig(max_size=10))
    assert cache.get("nonexistent") is None


def test_lru_eviction():
    """Oldest entry is evicted when max_size exceeded."""
    cache = ResponseCache(CacheConfig(max_size=2))
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_get_refreshes_lru_order():
    """Accessing an entry moves it to most-recently-used."""
    cache = ResponseCache(CacheConfig(max_size=2))
    cache.put("a", 1)
    cache.put("b", 2)
    # Access "a" to refresh it
    cache.get("a")
    # Now "b" is LRU, inserting "c" should evict "b"
    cache.put("c", 3)

    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3


def test_size_property():
    """Size reflects the number of entries."""
    cache = ResponseCache(CacheConfig(max_size=10))
    assert cache.size == 0
    cache.put("k1", "v1")
    assert cache.size == 1
    cache.put("k2", "v2")
    assert cache.size == 2


def test_clear():
    """Clear removes all entries."""
    cache = ResponseCache(CacheConfig(max_size=10))
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    cache.clear()
    assert cache.size == 0
    assert cache.get("k1") is None


def test_unlimited_no_eviction():
    """max_size=0 never evicts."""
    cache = ResponseCache(CacheConfig(max_size=0))
    for i in range(500):
        cache.put(f"k{i}", i)
    assert cache.size == 500
    assert cache.get("k0") == 0
    assert cache.get("k499") == 499


# ===================================================================
# 3. make_key
# ===================================================================


# --- MakeKey ---


def test_same_params_same_key():
    """Identical parameters produce the same key."""
    k1 = ResponseCache.make_key(
        model_id="ollama::qwen3",
        method="chat",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0,
        seed=None,
        max_tokens=-1,
        output_format=None,
        think=False,
    )
    k2 = ResponseCache.make_key(
        model_id="ollama::qwen3",
        method="chat",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0,
        seed=None,
        max_tokens=-1,
        output_format=None,
        think=False,
    )
    assert k1 == k2


def test_different_messages_different_key():
    """Different messages produce different keys."""
    k1 = ResponseCache.make_key(
        model_id="ollama::qwen3",
        method="chat",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0,
    )
    k2 = ResponseCache.make_key(
        model_id="ollama::qwen3",
        method="chat",
        messages=[{"role": "user", "content": "bye"}],
        temperature=0,
    )
    assert k1 != k2


def test_different_model_different_key():
    """Different model_id produces a different key."""
    k1 = ResponseCache.make_key(
        model_id="ollama::qwen3",
        method="chat",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0,
    )
    k2 = ResponseCache.make_key(
        model_id="ollama::llama3",
        method="chat",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0,
    )
    assert k1 != k2


# ===================================================================
# 4. LLMEngine cache integration (non-streaming)
# ===================================================================


# --- EngineCacheIntegration ---


def test_cache_hit_skips_llm_call():
    """Second identical call returns cached result."""
    engine = _make_engine(cache=CacheConfig())
    engine.llm_provider.chat.return_value = {
        "response": "hello",
        "role": "assistant",
        "thinking": None,
        "metrics": {},
    }

    result1 = engine.chat(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0,
    )
    result2 = engine.chat(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0,
    )

    assert result1 == result2
    assert engine.llm_provider.chat.call_count == 1


def test_nonzero_temp_bypasses_cache():
    """Non-zero temperature never caches."""
    engine = _make_engine(cache=CacheConfig())
    engine.llm_provider.chat.return_value = {
        "response": "hello",
    }

    engine.chat(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
    )
    engine.chat(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
    )

    assert engine.llm_provider.chat.call_count == 2


def test_no_cache_config_no_caching():
    """Without CacheConfig, every call hits the LLM."""
    engine = _make_engine(cache=None)
    engine.llm_provider.chat.return_value = {
        "response": "hello",
    }

    engine.chat(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0,
    )
    engine.chat(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0,
    )

    assert engine.llm_provider.chat.call_count == 2


@pytest.mark.asyncio
async def test_cache_hit_skips_async_llm_call():
    """Async: second identical call returns cached result."""
    engine = _make_engine(cache=CacheConfig())
    engine.llm_provider.chat_async = AsyncMock(
        return_value={"response": "hello"}
    )

    result1 = await engine.chat_async(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0,
    )
    result2 = await engine.chat_async(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0,
    )

    assert result1 == result2
    assert engine.llm_provider.chat_async.call_count == 1


def test_generate_cache_hit():
    """Generate: second identical call returns cached result."""
    engine = _make_engine(cache=CacheConfig())
    engine.llm_provider.generate.return_value = {
        "response": "generated",
    }

    result1 = engine.generate(prompt="hi", temperature=0)
    result2 = engine.generate(prompt="hi", temperature=0)

    assert result1 == result2
    assert engine.llm_provider.generate.call_count == 1


# ===================================================================
# 5. Streaming cache integration
# ===================================================================


# --- StreamingCacheIntegration ---


def test_stream_cached_after_iteration():
    """After iterating a stream, result is stored in cache."""
    engine = _make_engine(cache=CacheConfig())

    chunks = [
        chat_chunk(content="Hello"),
        chat_chunk(content=" world"),
        chat_chunk(
            done=True,
            eval_count=10,
            prompt_eval_count=5,
            total_duration=2_000_000_000,
        ),
    ]
    stream = StreamResult(iter(chunks), "chat")
    engine.llm_provider.chat_stream.return_value = stream

    result = engine.chat_stream(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0,
    )
    # Consume the stream to trigger _on_complete
    collected = list(result)

    assert collected == ["Hello", " world"]
    assert engine._cache.size == 1


def test_stream_cache_hit_returns_cached():
    """Cache hit returns a pre-filled StreamResult."""
    engine = _make_engine(cache=CacheConfig())

    chunks = [
        chat_chunk(content="Hello"),
        chat_chunk(
            done=True,
            eval_count=5,
            prompt_eval_count=2,
            total_duration=1_000_000_000,
        ),
    ]
    stream = StreamResult(iter(chunks), "chat")
    engine.llm_provider.chat_stream.return_value = stream

    result1 = engine.chat_stream(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0,
    )
    list(result1)  # consume to cache

    result2 = engine.chat_stream(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0,
    )
    # result2 should be from_cached, not calling provider
    assert engine.llm_provider.chat_stream.call_count == 1
    collected = list(result2)
    assert collected == ["Hello"]
    assert result2.response == "Hello"


# ===================================================================
# 6. from_cached
# ===================================================================


# --- FromCached ---


def test_sync_from_cached():
    """StreamResult.from_cached yields full response."""
    cached = {
        "response": "cached response",
        "thinking": "some thought",
        "role": "assistant",
        "metrics": {"tokens": 10, "prompt_tokens": 5},
    }
    stream = StreamResult.from_cached(cached)

    collected = list(stream)
    assert collected == ["cached response"]
    assert stream.response == "cached response"
    assert stream.thinking == "some thought"
    assert stream.role == "assistant"
    assert stream.metrics["tokens"] == 10


@pytest.mark.asyncio
async def test_async_from_cached():
    """AsyncStreamResult.from_cached yields full response."""
    cached = {
        "response": "async cached",
        "thinking": None,
        "role": "assistant",
        "metrics": {"tokens": 7},
    }
    stream = AsyncStreamResult.from_cached(cached)

    collected = []
    async for chunk in stream:
        collected.append(chunk)

    assert collected == ["async cached"]
    assert stream.response == "async cached"
    assert stream.thinking is None
    assert stream.role == "assistant"


def test_sync_from_cached_no_thinking():
    """from_cached with no thinking sets thinking to None."""
    cached = {
        "response": "hello",
        "role": None,
        "metrics": {},
    }
    stream = StreamResult.from_cached(cached)
    list(stream)
    assert stream.thinking is None


# ===================================================================
# 7. YAML config parsing
# ===================================================================


# --- CacheYAMLConfig ---


def test_yaml_with_cache(tmp_path):
    """Parses cache section into CacheConfig."""
    yaml_content = (
        "name: CachedLabeler\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "    cache:\n"
        "        max_size: 256\n"
        "system_prompt: Test prompt.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
    )
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content)

    agent = Labeler.from_yaml(str(yaml_file))

    assert agent.engine._cache is not None
    assert agent.engine._cache._max_size == 256


def test_yaml_without_cache(tmp_path):
    """No cache section means engine._cache is None."""
    yaml_content = (
        "name: NoCacheLabeler\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Test prompt.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
    )
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content)

    agent = Labeler.from_yaml(str(yaml_file))

    assert agent.engine._cache is None


def test_yaml_empty_cache_uses_defaults(tmp_path):
    """Empty cache: {} enables caching with defaults."""
    yaml_content = (
        "name: DefaultCacheLabeler\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "    cache: {}\n"
        "system_prompt: Test prompt.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
    )
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content)

    agent = Labeler.from_yaml(str(yaml_file))

    assert agent.engine._cache is not None
    assert agent.engine._cache._max_size == 128
