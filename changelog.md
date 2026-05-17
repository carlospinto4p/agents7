
## Changelog - Agents7

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


### v3.21.11 - 9th May 2026

- Refactored `agents7_framework/src/agents7/agents/`:
  - Merged `chatter_config_loader.py` into `config_loader.py` — all chatter-family loader functions (`load_chatter_yaml_settings`, `load_guarded_yaml_settings`, `load_guarded_scene_yaml_settings`, `validate_chatter_yaml_config`, `_derive_persona_summary`, `_load_base_chatter_config`, `_build_guard`) now live in a single module.
  - Deleted `chatter_config_loader.py`.
  - Updated all consumers (`chatter.py`, `guarded.py`, `guarded_scene.py`, `personality.py`, `scene.py`, `tests/test_guarded_chatter.py`, `tests/test_guarded_scene.py`) to import from `config_loader`.


### v3.21.10 - 9th May 2026

- Refactored `agents/`:
  - `detector.py`: moved `from_yaml()` to base class using `cls(**config)` and `typing.Self` return type — subclasses now inherit it automatically.
  - `classifier.py`: removed redundant `from_yaml()` override and `load_yaml_settings` import.
  - `regressor.py`: removed redundant `from_yaml()` override and `load_yaml_settings` import.


### v3.21.9 - 9th May 2026

- Refactored `src/agents7/agents/`:
  - `labeler.py`: replaced `self._set_attr()` calls in `_initialize_parser` and `_initialize_handlers` with `object.__setattr__()` to make the frozen-bypass explicit at each call site.
  - `detector.py`: same replacement in `__post_init__`.
  - `classifier.py`: same replacement in `__post_init__`.
- Refactored `integration_tests/`:
  - `test_multi_model.py`: narrowed `except Exception` to `except (OSError, subprocess.TimeoutExpired)` in `is_model_available`.
  - `conftest.py`: narrowed `except Exception` to `except (OSError, subprocess.TimeoutExpired)` in `_check_ollama`.


### v3.21.8 - 9th May 2026

- Performance improvements:
  - `memory.py`: Use `OrderedDict.popitem(last=False)` for O(1) LRU eviction (was `pop(next(iter())))`); use `OrderedDict` for `_cache`.
  - `chatter.py`: Avoid redundant history scan when `memory` is `None`; use local `history` variable for turn count.
  - `chatter_config_loader.py`: Eliminate intermediate string in `_derive_persona_summary()` — build one list and `join` once.
  - `agents7_ui/Runner.svelte`: Parallelize batch requests with 4-slot semaphore + `Promise.allSettled()`; combine three `$derived` filter passes into single `$derived.by()` loop; replace two `results.filter()` calls in `runBatch()` with one loop.
- Fixed global `format-python.sh` hook to use `uvx ruff` instead of `uv run ruff`.


### v3.21.7 - 9th May 2026

- Updated `.claude/rules/committing.md` to v1.1: added a
  "When to Skip Tests" section with an explicit allowlist
  (markdown, version bump, lock file, `.claude/` config,
  `CLAUDE.md`) of diffs where tests can be safely skipped.
  Canonical source for the rule — pushed to 15 linked
  projects via `push_to_projects.py`.


### v3.21.6 - 9th May 2026

- Added unit tests for scan-identified coverage gaps:
  - `src/agents7/model_selection.py`: `_benchmark_model` now raises `ValueError` when `runs < 1`; updated `tests/test_model_selection.py` accordingly.
  - `tests/test_memory.py`: 4 new `SqliteMemory` edge-case tests — concurrent same-conversation writes, `clear()` on missing id, corrupted DB file.
  - `tests/test_parsing.py`: 4 new `_fix_malformed_json()` pass-through tests — trailing comma, unclosed bracket, missing quotes, whitespace stripping.
  - `tests/test_chatter.py`: 5 new edge-case tests — chat after `reset()`, `max_history=0` behaviour, `conversation_id` isolation in memory, very long messages.
  - `tagger_agent/tests/test_rules_classifier.py`: 6 new tests asserting `confidence`, `rule`, and `detail` fields for name lookup, keyword substring, and fallback rules.


### v3.21.5 - 9th May 2026

- Added unit tests:
  - `tests/test_base_agent.py`: 12 tests for `describe()`, `_set_attr()`, and `resolved_system_prompt` edge cases.
  - `tests/test_agent_streaming.py`: 11 tests for `AgentStreamResult.from_failed()`, `AgentAsyncStreamResult.from_failed()`, and Detector streaming.
  - `tests/test_models_base.py`: 18 tests for `LLMMetrics`, `AgentResponse`, `ChatResponse`, `Property`, `ClassificationProperty`, and `_extend_enum_with_not_applicable()`.
  - `tests/test_metrics.py`: 14 new tests for timing stats, `_build_eval_result`, `_extract_fields`, `_fallback_sequential`, async batch failure fallback, and row-level async errors.


### v3.21.4 - 9th May 2026

- Added unit tests:
  - `tests/test_parsing.py`: 16 tests for `ResponseParser.parse()`, `_fix_malformed_json()`, enum name fallback, and `run_validators()`.
  - `tests/test_parse_retry.py`: 6 direct `ParseRetryHandler` unit tests (zero retries, success, exhaustion, temperature override, async).
  - `tests/test_dispatcher.py`: 10 tests for `EngineDispatcher` — cache hit/miss, fallback, cost injection, async, timeout.


### v3.21.3 - 9th May 2026

- Refactored `src/agents7/agents/`:
  - `classifier.py`: `_extra_prompt_sections()` delegates NOT_APPLICABLE guidance to `super()`, eliminating duplication with `Detector`.
  - `export.py`: Added `DeprecationWarning`; canonical import location is `agents7.eval.export`.
- Updated `tests/`:
  - `test_eval_report.py`, `test_export_dataframe.py`: Switched to `agents7.eval.export` directly.
  - `test_export.py`: Suppressed deprecation warning via `warnings.filterwarnings`.


