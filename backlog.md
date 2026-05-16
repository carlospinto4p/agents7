
# Backlog - Agents7

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.


### 10th May 2026 — /scan

#### Security

- [ ] `agents7_ui/backend/routers/run.py:35-37` — Pipeline agent paths resolved without bounds check; path traversal possible via `../../` — MED — M
- [ ] `agents7_ui/backend/main.py:12-13` — `CORS_ORIGINS` env var used without URL validation; add whitelist check — MED — S
- [ ] `agents7_ui/backend/routers/eval.py:96-99` — Raw `input_data` logged via `logger.warning`; may expose PII — MED — M
- [ ] `agents7_ui/backend/routers/engine.py:18` — Raw `str(exc)` returned in error response; return generic message — MED — M
- [ ] `agents7_ui/backend/routers/chat.py:56` — No rate limiting on session endpoints; session enumeration possible — MED — S
- [ ] `agents7_framework/src/agents7/llm/cache.py` — `check_same_thread=False` on SQLite connection is unnecessary given the existing lock — MED — M

#### Performance

- [ ] `agents7_ui/frontend/src/pages/Engine.svelte:20` — `fetchModels()` called in `$effect` without dependency guard; fires on every state change — HIGH — S
- [ ] `agents7_ui/frontend/src/pages/Catalog.svelte:8-12` — Three `.filter()` calls on agents array run reactively; wrap in `$derived` — HIGH — S
- [ ] `agents7_ui/backend/routers/engine.py:13-16` — Two sequential `httpx` calls; batch with `asyncio.gather` — MED — S
- [ ] `agents7_framework/src/agents7/llm/cache.py:87-106` — `json.dumps(output_format, sort_keys=True)` called on every cache key build; cache the result — MED — S
- [ ] `agents7_framework/src/agents7/agents/pipeline.py:506-519` — `list(await asyncio.gather(...))` is redundant; `gather` already returns a list — MED — S
- [ ] `agents7_ui/frontend/src/pages/Runner.svelte:165-177` — Manual slot-based concurrency; replace with `Promise.all` over batched chunks — MED — S
- [ ] `agents7_ui/frontend/src/pages/Runner.svelte:202-213` — Count aggregation recalculated on every render; use simpler `$derived` — MED — S
- [ ] `agents7_framework/src/agents7/agents/guarded.py:557-570` — Guard retry loop is fully sequential; check whether guard can run on prior response — MED — M

#### Refactoring

- [ ] `agents7_ui/backend/routers/run.py:17` + `chat.py:23` — `_import_agent_class`/`_import_class` duplicated across routers; extract to shared util — MED — S
- [ ] `agents7_ui/backend/routers/run.py:9` — `_AGENT_TYPE_MAP` defined in multiple routers; centralise in one place — MED — S
- [ ] `agents7_ui/backend/routers/chat.py:76` + `run.py:84` + `eval.py:95` — Broad `except Exception` catches without specific types; narrow them — MED — S
- [ ] `agents7_framework/src/agents7/agents/parsing.py:65,75,106` — Overly broad `except Exception` swallows parse/validation errors silently; catch `ParseError`, `ValidationError` specifically — MED — M
- [ ] `agents7_framework/src/agents7/model_selection.py:352` — Multiple broad `except Exception` blocks with inconsistent handling; standardise pattern — MED — M
- [ ] `agents7_framework/src/agents7/testing.py:92` — Bare `except Exception` in response parsing fallback; add specific type — MED — S
- [ ] `agents7_framework/src/agents7/metrics.py:1` — Backward-compatibility re-export duplicates interface; remove or consolidate — HIGH — S
- [ ] `agents7_framework/src/agents7/llm/streaming.py:11` — `_StreamBase` replicates state management logic from `_AgentStreamBase`; extract shared base — MED — M
- [ ] `agents7_framework/src/agents7/agents/detector.py` vs `classifier.py` vs `regressor.py` — Duplicated `_validate_output_model` and inheritance; extract to mixin — MED — M

#### Test coverage

- [ ] `agents7_framework/src/agents7/agents/memory.py` — No dedicated unit tests; concurrent access and `max_size` eviction never tested — HIGH — M
- [ ] `agents7_framework/src/agents7/agents/pipeline.py` — `act_stream()` / `act_stream_async()` untested; step failure mid-pipeline never verified — HIGH — M
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
