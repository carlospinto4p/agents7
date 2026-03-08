"""
Example: Persistent SQLite Cache

Demonstrates how to configure an LLMEngine with a SQLite-backed
cache that persists across process restarts. Repeated calls
with the same parameters return cached results instantly.
"""

from agents7.agents.labeler import Labeler
from agents7.llm import LLMEngine
from agents7.llm.cache import CacheConfig
from agents7.models.demo import PersonInfo


if __name__ == "__main__":
    # Create engine with SQLite cache
    engine = LLMEngine(
        provider="ollama",
        model_name="qwen2.5:3b",
        cache=CacheConfig(
            max_size=100,
            backend="sqlite",
            path="demo_cache.db",
        ),
    )

    extractor = Labeler(
        name="CachedExtractor",
        version=1,
        engine=engine,
        system_prompt=("Extract person information from the text."),
        output_model=PersonInfo,
    )

    text = "Sarah, a 28-year-old data scientist from London."

    # First call: hits the LLM
    print("[Call 1 — cache miss]")
    r1 = extractor.act({"text": text})
    print(f"  Result: {r1.model}")
    print(f"  Duration: {r1.metrics.total_duration:.2f}s")

    # Second call: served from SQLite cache
    print("\n[Call 2 — cache hit]")
    r2 = extractor.act({"text": text})
    print(f"  Result: {r2.model}")
    print(f"  Duration: {r2.metrics.total_duration:.2f}s")

    print(f"\nCache size: {engine._cache.size} entries")
    print("Cache file: demo_cache.db (persists across runs)")
