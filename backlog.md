
# Backlog - Agents7

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.


### 10th May 2026 — /scan

#### Test coverage

- [x] `agents7_framework/src/agents7/agents/memory.py` — No dedicated unit tests; concurrent access and `max_size` eviction never tested — HIGH — M
- [x] `agents7_framework/src/agents7/agents/pipeline.py` — `act_stream()` / `act_stream_async()` untested; step failure mid-pipeline never verified — HIGH — M (premise no longer holds — test_pipeline_stream.py already covers act_stream, act_stream_async, and mid-pipeline failures)
- [ ] `agents7_framework/src/agents7/agents/guarded.py` — Retry loop on guard rejection untested; no test for guard that always rejects — HIGH — M
- [ ] `agents7_framework/src/agents7/llm/openai_compat.py` — No dedicated unit tests for error handling, retry logic, or streaming chunk parsing — HIGH — L
- [ ] `agents7_framework/src/agents7/agents/batch.py:42-80` — Async concurrency test has timing dependencies; flaky in CI — HIGH — M
- [ ] `agents7_framework/src/agents7/agents/parsing.py` — No dedicated test file; `ResponseParser.parse()` only tested indirectly — MED — M
- [ ] `agents7_framework/src/agents7/llm/cache.py:63-100` — Cache key builder incomplete; no hash-collision or malformed-dict tests — MED — M
- [ ] `agents7_framework/src/agents7/llm/retry.py:62-100` — Missing negative-status tests (400, 401, 403) for `_is_retryable()` — MED — S
- [ ] `agents7_framework/src/agents7/llm/dispatcher.py:30-120` — `_cache_store_callback` never tested in isolation — MED — M
- [ ] `agents7_framework/src/agents7/agents/context_guard.py:53-160` — Guard prompt templates untested; no test for injection via `persona_summary` — MED — M
- [ ] `agents7_framework/src/agents7/agents/guarded_scene.py` — No specific tests for auto-generated system prompt combining persona + scene — MED — S
- [ ] `agents7_framework/src/agents7/agents/utils.py:87-120` — `get_examples_str()` untested for malformed examples (missing keys, type mismatches) — MED — S
- [ ] `agents7_framework/src/agents7/eval/metrics.py:79-140` — Error isolation in batch evaluation never tested (partial batch completion) — MED — M
- [ ] `agents7_framework/src/agents7/export.py` — No dedicated unit tests for DataFrame export with mixed types, NaN, datetimes — MED — M
- [ ] `agents7_framework/src/agents7/agents/config_loader.py` — Circular YAML references and missing field defaults untested — MED — M
