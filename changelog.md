
## Changelog - Agents7



### v3.26.0 - 7th June 2026

- Updated `GuardedChatter` fix fallback: when retries are
  exhausted and the guard supplied a `fix`, the chatter now
  regenerates once using the fix as guidance (in full persona
  voice) and guards the result, instead of returning the
  guard's one-line fix verbatim. Falls back to the verbatim fix
  if the regeneration is still rejected.
  - Added `_regenerate_from_fix` / `_regenerate_from_fix_async`
    and `_add_fix_guidance`.
  - Adds one chatter call + guard check only on the
    exhausted-retry path, and only when a `fix` exists.
- Updated `docs/chatter.md`: documented the fix-guided
  regeneration step.


### v3.25.0 - 7th June 2026

- Added language-aware forbidden vocabulary to the guard:
  - `ContextGuard.forbidden_terms`: list injected into the
    prompt (via a `$forbidden_terms` placeholder), defaulting
    to the previous bilingual ES/EN list so the default prompt
    is byte-identical.
  - `forbidden_terms_for_language()` in `agents/context_guard.py`:
    starter lists for Spanish, English, French, German,
    Portuguese, Italian; unknown/empty falls back to the
    default.
  - YAML loaders derive `forbidden_terms` from
    `persona.language` unless `guard.forbidden_terms` is set;
    `ContextGuard.from_yaml` accepts the field.
  - `GuardedChatter._add_feedback` now lists the guard's
    `forbidden_terms` instead of a hardcoded Spanish set.
- Updated `docs/chatter.md`: documented language-aware
  forbidden vocabulary.


### v3.24.1 - 7th June 2026

- Updated `Chatter.__post_init__`: a non-empty `scene` with no
  `persona` now raises `AgentConfigurationError` instead of
  silently dropping the scene (the scene frames the persona's
  role-play and is unusable on its own). An empty `scene`
  without a persona is still allowed.
- Updated `docs/chatter.md`: noted the scene-requires-persona
  rule.


### v3.24.0 - 7th June 2026

- Added fail-closed guarding to `GuardedChatter` (and
  `GuardedSceneChatter`):
  - `guard_fail_open` (default `True`): when `False`, an
    unparseable guard verdict no longer passes through —
    the response is replaced with a fallback deflection.
  - `guard_fallback_response`: the in-character deflection
    used when fail-closed and the guard gives no verdict
    (empty uses a generic default).
  - `_is_accepted` now honours `guard_fail_open`; the
    parse-failure `GuardEvent` reports `accepted` truthfully.
  - YAML loaders accept both new fields.
- Updated `docs/chatter.md`: documented fail-open vs
  fail-closed guarding.


### v3.23.0 - 7th June 2026

- Added `guard_buffer_stream` field to `GuardedChatter` (and
  `GuardedSceneChatter`): when `True`, streaming buffers the
  full response, runs the guard, and yields only the guarded
  text, so unvalidated content never reaches the caller.
  Defaults to `False` (live streaming, guard runs after).
  - `GuardedChatterStreamResult` / `GuardedChatterAsyncStreamResult`
    override iteration to drain-then-guard in buffered mode.
  - YAML loaders accept `guard_buffer_stream`.
- Updated docs:
  - `docs/chatter.md`: documented `guard_buffer_stream` and the
    streaming guard trade-off.
  - `README.md`: corrected the agent hierarchy diagram
    (`GuardedChatter` extends `Chatter`, not `PersonalityChatter`).


### v3.22.0 - 7th June 2026

- Added `Scene.to_summary()` in `agents/persona.py`: renders a
  compact scene summary (setting, situation, rules) for the guard
  prompt.
- Updated `GuardedSceneChatter`:
  - Added `__post_init__` that folds the scene context into the
    guard's `persona_summary`, so direct Python construction now
    matches the YAML loader. Idempotent; an explicit
    `persona_summary` overrides the persona description but the
    scene context is still appended.
- Updated `_derive_persona_summary` (`agents/config_loader.py`)
  to reuse `Scene.to_summary()`.
- Updated `docs/chatter.md`: scene-aware guarding now applies to
  both construction paths.


### v3.21.43 - 7th June 2026

- Updated `docs/chatter.md`:
  - Replaced the stale ASCII agent hierarchy with a Mermaid
    `classDiagram` showing inheritance plus persona/scene/guard
    composition.
  - Corrected `GuardedChatter` to extend `Chatter` (not
    `PersonalityChatter`).
  - Added a `GuardedSceneChatter` section, including the
    YAML-only scene enrichment of the guard's `persona_summary`.


### v3.21.42 - 7th June 2026

- Rotated changelog: archived 2 entries to `changelog/2026.md`.

### v3.21.41 - 6th June 2026

- Cleaned up stale Streamlit references left by the Agent Builder's
  migration to the `agents7_ui` SvelteKit + FastAPI app:
  - `docs/agent_builder.md`: replaced the 413-line Streamlit wizard
    guide with a redirect stub pointing to `agents7_ui/`.
  - `src/agents7/eval/metrics.py` and `README.md`: reworded
    "usable without Streamlit" → "headless evaluation primitives".


### v3.21.40 - 4th June 2026

- Rotated changelog: archived 7 entries to `changelog/2026.md`.

### v3.21.39 - 4th June 2026

- Synced `.claude/rules/committing.md` from the programme registry: step 6 now scopes `uv.lock` regeneration to code-related bumps only — non-code patch bumps (`.claude/` config, docs, changelog, rule syncs) skip `uv lock`.

### v3.21.38 - 3rd June 2026

- Synced `.claude/rules/testing.md` from the programme registry: added the SQLite-backed fixtures pointer to the session-scoped template pattern (see the shared `testing-python` rule).

### v3.21.37 - 31st May 2026

- Added `[build-system]` to `pyproject.toml`; `uv sync --all-extras` now handles the editable install automatically.


### v3.21.36 - 17th May 2026

- Added unit tests:
  - `tests/test_batch.py`: 3 edge-case tests for `BatchProcessor._chunk_inputs()` (empty list, single item, `requests_per_prompt=1`).
  - `tests/test_llm_engine.py`: 10 tests for `LLMEngine` initialization, provider lowercasing, fallback wiring, `supports_warmup()`, and `_create_provider()` for all three providers and the unknown-provider error path.


### v3.21.35 - 17th May 2026

- Added unit tests:
  - `tests/test_cache.py`: edge-case tests for `ResponseCache.make_key()` — `messages=[]` vs `None`, `prompt=None` vs `""`, Unicode/special-character content.
  - `tests/test_model_selection.py`: additional tests for `TestCase`, `CaseResult`, and `BenchmarkResult` — empty fields, nested input, `passed=False`, run number, immutability.
  - `tests/test_pipeline.py`: edge-case tests for `Pipeline.act_in_batch()` — empty input, all-fail, mapper failure per item, all-items-skipped-by-condition.


### v3.21.34 - 17th May 2026

- Added unit tests:
  - `tests/test_guarded_chatter.py`: parse failure on retry (sync + async), GuardedChatter as pipeline step, GuardedChatter memory persistence.
  - `tests/test_config_loader.py`: `load_yaml_settings()` with `base_config` inheritance chain, `build_engine()` resolving `${VAR}` in `api_key`, error on unset env var.


### v3.21.33 - 17th May 2026

- Rotated changelog: archived 84 old entries to `changelog/2026.md`.



### v3.21.32 - 17th May 2026

- Added unit tests in `tests/test_chatter.py`:
  - `test_from_yaml_invalid_memory_backend`: invalid backend raises `AgentConfigurationError`.
  - `test_from_yaml_circular_inheritance`: circular `base_config` raises `AgentConfigurationError`.
  - `test_from_yaml_malformed_yaml`: malformed YAML raises `yaml.YAMLError`.


### v3.21.31 - 17th May 2026

- Removed deprecated `src/agents7/export.py` re-export shim:
  - Migrated all callers to `agents7.eval.export` directly.
  - Updated `tests/test_export.py`, `integration_tests/report_utils.py`, `examples/features/02_export.py`, `README.md`, and `agents_catalog/tagger_agent/tests/test_tagger_agent.py`.


### v3.21.30 - 17th May 2026

- Added return type annotations to previously unannotated public/private methods:
  - `llm/ollama.py`: `_with_think_fallback()` and `_with_think_fallback_async()` → `dict`
  - `agents/labeler.py`: `_preprocess()` → `Any`
  - `agents/utils.py`: `run_with_semaphore()` → `Any`
  - `models/base.py`: `ClassificationProperty.validate_category_reason_link()` → `"ClassificationProperty"`


### v3.21.29 - 17th May 2026

- Perf: `llm/cache.py` `_CacheKeyBuilder.make_key()`: removed module-level `_FMT_CACHE` dict and pre-serialization of `output_format`; now passes all fields through a single `json.dumps(..., sort_keys=True)` call.


### v3.21.28 - 17th May 2026

- Added Table of Contents to `README.md` with links to all 34 sections and subsections.


### v3.21.27 - 16th May 2026

- Added unit tests:
  - `tests/test_export_dataframe.py`: mixed primitive types (int/float/bool),
    Optional field set to None within valid model, datetime field stringification.
  - `tests/test_config_loader.py`: `parse_yaml_examples()` (absent, valid,
    not-a-list, non-dict item, missing/extra keys), `cast_optional_fields()`
    (present fields cast, absent fields skipped, source not mutated),
    `load_yaml_settings()` edge cases (empty `prompt_params`, malformed examples).


### v3.21.26 - 16th May 2026

- Added unit tests (`agents7_framework/tests/`):
  - `test_context_guard.py`: 3 tests for `persona_summary` injection — placeholder fully replaced, injection attempts embedded verbatim, self-referential `$persona_summary` not re-substituted.
  - `test_guarded_scene.py`: 4 tests verifying all scene/persona fields in auto-generated system prompt (situation, time_period, goal, occupation).
  - `test_utils.py`: 8 tests for `get_examples_str()` — happy path, empty list, non-list input, non-dict entry, missing keys, extra keys.
  - `test_metrics.py`: 1 test for partial batch completion — rows with `model=None` are individually marked as errors without affecting passing rows.


### v3.21.25 - 16th May 2026

- Added tests in `agents7_framework/tests/`:
  - `test_cache.py`: key-order invariance, non-serializable `output_format`, and distinct-payload collision tests for `_CacheKeyBuilder.make_key`.
  - `test_retry.py`: `_is_retryable()` coverage for non-retryable status codes (400, 401, 403), generic 429, and plain exceptions.
  - `test_dispatcher.py`: `_cache_store_callback` isolation tests (store, overwrite, key isolation).


### v3.21.24 - 16th May 2026

- Added unit tests in `tests/test_openai_compat.py`:
  - `test_chat_sync_propagates_api_error`: error propagation from sync client.
  - `test_chat_async_propagates_api_error`: error propagation from async client.
  - `test_oai_chunk_empty_choices`: `_OAIChunk` with empty choices list.
  - `test_oai_chunk_delta_with_reasoning`: `reasoning_content` in delta.
  - `test_adapt_stream_sync_filters_empty_chunks`: empty-chunk filtering.
  - `test_adapt_stream_sync_yields_usage_chunk`: usage-only chunk passthrough.
  - `test_chat_stream_returns_stream_result`: `chat_stream()` integration.
  - `test_chat_stream_async_returns_async_stream_result`: `chat_stream_async()` integration.


### v3.21.23 - 16th May 2026

- Added tests in `tests/test_guarded_chatter.py`:
  - `test_guard_call_count_equals_initial_plus_retries`: guard called `max_guard_retries+1` times when always rejecting.
  - `test_chat_zero_retries_uses_fix`: `max_guard_retries=0` skips retry loop and applies fix immediately.
  - `test_chat_zero_retries_no_fix_uses_last`: `max_guard_retries=0` with no fix uses last response.
  - `test_chat_async_always_rejects_no_fix`: async path, all retries exhausted, no fix available.
  - `test_broke_character_hint_in_feedback`: BROKE_CHARACTER reason produces AI-vocabulary hint in retry message.
- Fixed `tests/test_batch.py` `test_async_multi_chunk_preserves_order`: replaced timing-dependent `asyncio.sleep` delays with `asyncio.Event` synchronization to eliminate CI flakiness.


### v3.21.22 - 16th May 2026

- Added unit tests in `tests/test_memory.py`:
  - `test_json_cache_max_size_evicts_oldest`: verifies `_MAX_CACHE` eviction evicts the oldest entry.
  - `test_json_concurrent_saves`: verifies thread-safe concurrent saves on `JsonFileMemory`.


### v3.21.21 - 16th May 2026

- Refactored `metrics.py`: removed backward-compat re-export shim; updated all callers to import directly from `agents7.eval.metrics`:
  - `agents7_framework/tests/test_metrics.py`
  - `agents7_framework/tests/test_eval_report.py`
  - `agents7_framework/integration_tests/report_utils.py`
  - `agents_catalog/tagger_agent/tests/test_tagger_agent.py`
  - `agents7_framework/README.md`


### v3.21.20 - 16th May 2026

- Refactored `testing.py`: narrowed bare `except Exception` in `_wrap_for_batch` to `except (ValueError, SyntaxError)`.


### v3.21.19 - 16th May 2026

- Refactored `model_selection.py`: added `logger.debug()` to engine-init and agent-creation `except Exception` blocks, standardising debug-logging across all four catch sites.


### v3.21.18 - 16th May 2026

- Narrowed broad `except Exception` in `agents/parsing.py`:
  - `parse()` now catches `(ValidationError, json.JSONDecodeError)` for both parse attempts instead of `Exception`


### v3.21.17 - 16th May 2026

- Performance:
  - `agents/pipeline.py`: removed redundant `list()` wrapper around `asyncio.gather()` return value.
  - `agents7_ui/frontend/src/pages/Runner.svelte`: replaced slot-based concurrency with batched `Promise.all` chunks.
  - `agents7_ui/frontend/src/pages/Runner.svelte`: simplified count aggregation from `$derived.by` + three `$derived` accessors to three direct `$derived` filters.


### v3.21.16 - 16th May 2026

- Extracted `_row_input()` helper in `eval/metrics.py`: removes four
  inline `row.get("input", row)` repetitions in `_build_eval_result`,
  `run_evaluation`, `run_evaluation_async`, and `_fallback_sequential`.


### v3.21.15 - 11th May 2026

- Removed `.claude/rules/ui.md`: stale Streamlit UI guidelines; project has moved to `agents7_ui` (Svelte + FastAPI).
- Added `.gitignore` entry for `.claude/scheduled_tasks.lock` (Claude Code runtime artifact).


### v3.21.14 - 10th May 2026

- Added `tests/test_guarded_scene.py`:
  - `test_from_yaml_persona_summary_includes_situation`: verifies `situation` appears in derived guard persona_summary.
  - `test_from_yaml_persona_summary_explicit_override`: verifies explicit `guard.persona_summary` in YAML overrides the derived value.
  - `test_from_yaml_persona_summary_no_scene_omits_scene_keys`: verifies persona_summary has no scene keys when no scene is in YAML.


### v3.21.13 - 10th May 2026

- Added `from_yaml()` tests in:
  - `tests/test_detector.py`: `test_detector_from_yaml_returns_instance`.
  - `tests/test_classifier.py`: `test_classifier_from_yaml_returns_instance`.
  - `tests/test_regressor.py`: `test_regressor_from_yaml_returns_instance`.
  - `tests/test_context_guard.py`: `test_context_guard_from_yaml_returns_instance`.
- Fixed `ContextGuard.from_yaml()` in `agents/context_guard.py`: now extracts `persona_summary` from the raw YAML via `_resolve_inheritance`, making YAML-based construction work end-to-end.
- Fixed `BatchProcessor._flatten()` in `agents/batch.py`: skips chunks with `model=None` (malformed LLM response) instead of crashing with `AttributeError`.
- Added malformed-chunk tests in `tests/test_batch.py`:
  - `test_malformed_chunk_is_skipped`.
  - `test_valid_chunks_preserved_around_malformed`.
- Items 4 and 5 from the `/scan` backlog (`_compare_output()` `None`-field tests and `asyncio.TimeoutError` dispatcher test) were already implemented; marked done.


### v3.21.12 - 10th May 2026

- Added tests in `tests/test_guarded_chatter.py`:
  - `test_chat_stream_guard_accepted`: chunks yielded and guard accepts.
  - `test_chat_stream_guard_rejected_uses_fix`: fix applied in `_finalize`.
  - `test_chat_stream_finalize_fires_callback`: `guard_callback` fires after stream.
  - `test_chat_stream_async_guard_accepted`: async chunks yielded and guard accepts.
  - `test_chat_stream_async_guard_rejected_uses_fix`: fix applied in `_finalize_async`.
  - `test_chat_stream_async_finalize_fires_callback`: callback fires after async stream.


