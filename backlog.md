
# Backlog - Agents7

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.


### Scan — 17th May 2026

#### Test Coverage

- [ ] Add tests for guard retry loop exhaustion in `agents/guarded.py:506-570`: parse failure on retry, all retries exhausted, `max_guard_retries=0` (HIGH, M)
- [ ] Add integration tests: `GuardedChatter` with real guard, `Pipeline+GuardedChatter` steps, `SceneChatter` system prompt generation, memory persistence across restarts (HIGH, M)
- [ ] Add tests for `agents/config_loader.py`: YAML inheritance (`_resolve_inheritance()`), variable substitution, multi-file pipeline configs (HIGH, M)
- [ ] Add edge-case tests for `ResponseCache.make_key()` in `llm/cache.py`: empty messages, `None` values, special characters (HIGH, L)
- [ ] Add tests for `BenchmarkResult.select_best()`, `TestCase`, `CaseResult` in `model_selection.py` (HIGH, L)
- [ ] Add tests for `Pipeline.act_in_batch()` edge cases: empty input, all-fail, mixed condition partitioning, mapper failures (MED, M)
- [ ] Add edge-case tests for `BatchProcessor._chunk_inputs()` in `agents/batch.py:212-222`: `requests_per_prompt=1`, single item, empty list (MED, M)
- [ ] Add tests for `LLMEngine` initialization and provider fallback in `llm/base.py` (MED, M)
