
## Changelog - Agents7

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


### v3.21.2 - 9th May 2026

- Regrouped 9 historical changelog entries flagged by `/changelog-review`:
  - `agents7_framework/changelog.md` v3.21.1: 3 `src/agents7/llm/` and 3 `src/agents7/agents/` bullets each grouped under their module parent.
  - `agents7_framework/changelog.md` v3.18.25: 4 `agents/` bullets grouped under one parent.
  - `agents7_framework/changelog.md` v3.17.7: 3 `agents/` bullets grouped under one parent.
  - `agents7_framework/changelog.md` v3.15.12: 3 `tests/` bullets grouped under one parent.
  - `agents7_framework/changelog.md` v3.15.6: 3 `llm/` bullets grouped under one parent.
  - `agents7_framework/changelog.md` v3.15.5: 3 `agents/` bullets grouped under one parent.
  - `agents7_framework/changelog/2026.md` v1.2.7: 3 `.claude/rules/` bullets grouped under one parent.
  - `agents7_framework/changelog/2026.md` v1.3.0: 3 `_pages/` bullets grouped under one parent.
- Closes the resume-pointer from session 2026.05.09: completes the 28-entry historical changelog regrouping pass started across 10 projects.


### v3.21.1 - 9th May 2026

- `src/agents7/llm/`:
  - Refactored `retry.py`:
    - Extracted `_retry_impl()`: shared sync retry loop accepting a `sleep_fn` callable; `retry_sync()` now delegates to it.
    - Added `_all_provider_errors()`: helper returning all known provider error types (ollama + openai).
  - Added `LLMEngine.supports_warmup()` in `base.py`.
  - Refactored `dispatcher.py`:
    - Added `_dispatch_errors()` helper; replaced four broad `except Exception` clauses with specific provider/LLM error types so programming bugs propagate instead of triggering fallback.
- `src/agents7/agents/`:
  - Refactored `base.py`:
    - Added `_set_attr()`: replaces `object.__setattr__()` calls across all agent subclasses.
    - `warm_up()` / `warm_up_async()`: use `engine.supports_warmup()` instead of hard-coded `provider != "ollama"` string.
  - Refactored `labeler.py`:
    - Extracted `_prepare_input()`: consolidates `_sanitize_chat_kwargs` + `_preprocess` + `_build_act_messages` called in all four `act*()` methods.
    - Added `_validate_input()`: raises `AgentConfigurationError` for non-dict / non-list-of-dicts inputs before preprocessing.
  - Refactored `detector.py`:
    - `__post_init__()` caches `_prompt` and `_prompt_for_batch`; properties now return cached values instead of recomputing on every access.
- Updated `object.__setattr__()` → `self._set_attr()` in `chatter.py`, `classifier.py`, `context_guard.py`, `detector.py`, `labeler.py`.


### v3.21.0 - 4th May 2026

- Refactored `agents/` — composition over inheritance for Chatter personality/scene:
  - Added `agents/persona.py`: new leaf module with `PhysicalTraits`, `Persona`, `Scene`, and `_build_scene_system_prompt`. No imports from other agent modules, preventing circular dependency.
  - Updated `agents/chatter.py`: added `persona: Persona | None`, `scene: Scene | None`, and `extra_instructions: str` as optional plugin fields. `__post_init__` auto-generates the system prompt when `persona` is set; added `_build_plugin_prompt()`.
  - Updated `agents/personality.py`: replaced data class definitions with re-exports from `persona.py`; removed `__post_init__` and `_build_system_prompt` overrides (now handled by `Chatter`).
  - Updated `agents/scene.py`: removed `_build_scene_system_prompt` (moved to `persona.py`); removed `_build_system_prompt` override.
  - Updated `agents/guarded.py`: `GuardedChatter` now extends `Chatter` directly instead of `PersonalityChatter`.
  - Updated `agents/guarded_scene.py`: removed `_build_system_prompt` override and invalid `_build_scene_system_prompt` import; imports `Scene` from `persona.py`.


### v3.20.3 - 4th May 2026

- Refactored `src/agents7/agents/detector.py`:
  - Removed redundant `self._validate_output_model()` call from `__post_init__()`; `Labeler.__post_init__()` already calls it via polymorphism before `_initialize_parser()` runs.


### v3.20.2 - 4th May 2026

- Removed GitLab token from `agents7_ui/docker-compose.yml`:
  - Removed `secrets:` block from `backend.build`.
  - Removed top-level `secrets:` section.
  - `docker compose up --build` now works with no credentials.


### v3.20.1 - 4th May 2026

- Refactored `src/agents7/agents/labeler.py` `__post_init__()`:
  - Extracted `_validate_output_model()`: guards `output_model` is a `BaseModel` subclass.
  - Extracted `_initialize_parser()`: builds `output_model_list`, schemas, and `ResponseParser`.
  - Extracted `_initialize_handlers()`: builds `ParseRetryHandler` and `BatchProcessor`.


### v3.20.0 - 4th May 2026

- Removed `cpinto` dependency from `agents7_framework`:
  - `src/agents7/logs.py`: replaced `get_app_logger` with `logging.getLogger`.
  - `src/agents7/settings.py`: replaced `load_setting` with inline `os.getenv` helper `_load_setting`.
  - `src/agents7/agents/config_loader.py`: replaced `load_yaml` with local `_load_yaml` helper using `yaml.safe_load` (UTF-8).
  - `src/agents7/agents/pipeline.py`: replaced `load_yaml` with inline `yaml.safe_load` (UTF-8).
  - `src/agents7/agents/batch.py`: replaced `identity_func` with `lambda x: x`.
  - `pyproject.toml`: removed `cpinto>=0.4.10` dependency and GitLab private index config.
  - `agents7_ui/Dockerfile.backend`: removed BuildKit secret mount and GitLab token env vars.
- Added `.dropboxignore` at repo root to prevent `.venv/` from syncing to Dropbox.


### v3.19.0 - 3rd May 2026

- Removed `agents7.ui` Streamlit-based Agent Builder UI:
  - Deleted `src/agents7/ui/` and all submodules.
  - Removed `[ui]` optional dependency group (`streamlit`, `polars`, `plotly`).
  - Removed `agents7-ui` CLI entry point.
  - Removed `Dockerfile` (was UI-only; new UI lives in `agents7_ui/`).
  - Updated `README.md` and `CLAUDE.md` to remove UI references.


### v3.18.48 - 3rd May 2026

- Parallelised `BatchProcessor.act_in_batch()` in `src/agents7/agents/batch.py`:
  - Added `batch_settings` parameter (mirrors async variant).
  - Replaced sequential list-comprehension with `ThreadPoolExecutor.map()`; chunks run concurrently, results returned in submission order.


### v3.18.47 - 3rd May 2026

- Optimised `src/agents7/agents/chatter.py` `_trim_history()`: replaced O(excess) scan over removed messages with O(1) arithmetic to compute the assistant-turn count.


### v3.18.46 - 3rd May 2026

- Optimised `src/agents7/llm/cache.py`:
  - `SqliteCache.put()`: moved LRU eviction query inside the `if not exists` block — updates no longer trigger an eviction check.
  - `_CacheKeyBuilder.make_key()`: added `_FMT_CACHE` (id-keyed dict) to avoid re-serialising the same `output_format` schema dict on every call.


### v3.18.45 - 3rd May 2026

- Fixed `agents/memory.py` `JsonFileMemory._conv_path()`:
  - Guards against all-dot identifiers (`"."`, `".."`): maps them
    to `"_"` to avoid creating special directory-entry filenames.
  - Added `path.parent == self._dir` assertion as belt-and-suspenders
    against future sanitization regressions.
- Added tests in `tests/test_memory.py`:
  - `test_json_conv_path_dotdot_sanitized`
  - `test_json_conv_path_traversal_chars_replaced`


### v3.18.44 - 3rd May 2026

- Fixed `agents/config_loader.py` `_resolve_inheritance()`: rejects
  absolute `base_config` paths with `AgentConfigurationError`, preventing
  injection of arbitrary system files (e.g. `base_config: /etc/passwd`).
- Updated `tests/test_config_inheritance.py`:
  - Changed all test fixtures from absolute `base_config` paths to
    relative ones (the correct real-world pattern).
  - Added `test_absolute_base_config_raises`.


### v3.18.43 - 3rd May 2026

- Added input validation in `agents/utils.py` `import_from_path()`:
  rejects paths containing characters outside `[A-Za-z0-9_.]` with
  `AgentConfigurationError`, blocking arbitrary module loading when
  the path originates from user-controlled config.
- Added tests in `tests/test_utils.py`:
  - `test_import_from_path_valid`
  - `test_import_from_path_unsafe_slash_raises`
  - `test_import_from_path_unsafe_space_raises`
  - `test_import_from_path_unsafe_leading_dot_raises`


### v3.18.42 - 3rd May 2026

- Added `_resolve_env_var()` in `agents/config_loader.py`:
  resolves `${VAR_NAME}` references to environment variables.
  `build_engine()` now calls it for `api_key` and `base_url`,
  so API keys can be stored as env var references instead of
  plaintext in YAML configs.
- Added tests in `tests/test_config_loader.py`:
  - `test_resolve_env_var_plain_string`
  - `test_resolve_env_var_resolves_set_var`
  - `test_resolve_env_var_missing_raises`
  - `test_resolve_env_var_partial_reference_unchanged`


### v3.18.41 - 3rd May 2026

- Fixed `agents/context_guard.py` `_DEFAULT_GUARD_PROMPT`:
  - Added explicit REJECT example for "No soy un modelo de
    lenguaje, soy David" — covers the negation pattern that
    qwen3:4b was accepting (rej=0) despite "modelo de lenguaje"
    being prohibited vocabulary.
  - Added IMPORTANT note: scene rules are binding — violating
    a scene constraint (e.g. source protection) is OFF_TOPIC
    even if the content relates to the scene subject.
  - Added source-protection REJECT/ACCEPT examples: responses
    that reveal source-identifying details are OFF_TOPIC; correct
    deflections are IN_CHARACTER.


### v3.18.40 - 3rd May 2026

- Fixed `agents/guarded.py` `_add_feedback`: added reason-specific hints to the retry feedback message.
  - `BROKE_CHARACTER`: explicitly prohibits AI/LLM vocabulary in the regenerated response.
  - `PROMPT_INJECTION`: explicitly instructs the character to dismiss or ignore the injected persona/instruction.


### v3.18.39 - 3rd May 2026

- Fixed `benchmarks/benchmark_guarded_scene.py` bad signals:
  - Case 10 (`scene_break`): removed `'la capital'` — fires on redirect phrases like "con la capital de Australia ya nos vamos a perder el hilo" rather than actual violations.
  - Case 5 (`injection`): replaced `'pirata'` with `'soy un pirata'` and `'soy pirata'` — bare `'pirata'` fires when the character dismisses the injection ("aquí no se hace el pirata"), not just when it complies.


### v3.18.38 - 3rd May 2026

- Updated `tests/test_labeler.py`:
  - `test_batch_parse`: replaced weak `assert result is not None` with `assert result.items[1].value == "b"` to verify both parsed items.


### v3.18.37 - 3rd May 2026

- Added unit tests in `tests/test_jitter.py`:
  - `test_inject_typo_4char_swap_modifies`: swap on a 4-char word modifies the string.
  - `test_inject_typo_4char_drop_no_change`: drop on a 4-char word is silently skipped (`len > 4` guard).
  - `test_inject_typo_5char_drop_fires`: drop on a 5-char word does shorten the string.


### v3.18.36 - 3rd May 2026

- Added unit tests in `tests/test_streaming.py`:
  - `test_extract_chat_no_message_attr`: chat chunk with no `message` attribute yields empty strings.
  - `test_extract_chat_message_is_none`: chat chunk with `message=None` yields empty strings.
  - `test_extract_generate_no_response_attr`: generate chunk with no `response` attribute yields empty string.
  - `test_extract_chat_message_missing_content`: chat chunk whose message has no `content` attribute yields empty.


### v3.18.35 - 3rd May 2026

- Added unit tests in `tests/test_model_selection.py`:
  - `test_benchmark_model_runs_zero`: verifies `_benchmark_model` with `runs=0` returns a vacuous pass with no case results.
  - `test_find_best_model_runs_zero_passed_through`: verifies `runs=0` is forwarded to `_benchmark_model`.


### v3.18.34 - 3rd May 2026

- Added unit tests in `tests/test_memory.py`:
  - `test_json_load_corrupted_file`: verifies `JsonFileMemory.load()` raises `json.JSONDecodeError` on a corrupted JSON file.


### v3.18.33 - 3rd May 2026

- Added unit tests:
  - `tests/test_ollama_utils.py`: 6 tests covering `check_connection()` and
    `list_available_models()` error paths (`httpx.HTTPError`,
    `ollama.ResponseError`, success).
  - `tests/test_config_loader.py`: 12 tests covering `validate_yaml_config()`,
    `build_engine()` (minimal, retry, cache, fallback), and
    `load_yaml_settings()` (keys, prompt strip, params, optional fields).
  - `tests/test_utils.py`: 13 tests covering `run_with_semaphore()`,
    `resolve_strenum()`, and `model_class_to_string()`.


### v3.18.32 - 3rd May 2026

- Added unit tests:
  - `tests/test_model_selection.py`: `test_benchmark_model_engine_init_failure` — verifies `_benchmark_model()` returns `ModelScore(passed=False)` when `LLMEngine` init raises.
  - `tests/test_memory.py`: `test_sqlite_concurrent_read_write` — verifies `SqliteMemory` serialises concurrent saves/loads from 10 threads without corruption.


### v3.18.31 - 3rd May 2026

- Added `tests/test_eval_report.py`:
  - `test_report_timing_section_rendered`: verifies `## Response Times` section renders correctly when timing data is present.
  - `test_report_per_case_timing`: verifies per-case duration is appended to each case header.
  - `test_report_timing_omitted_when_absent`: verifies timing section is absent when `timing` key is missing from metrics.
  - `test_report_timing_omitted_when_zero`: verifies timing section is absent when `total_s` is zero.
  - `test_report_missing_metric_fields`: verifies report uses zero defaults when metrics keys are absent.


### v3.18.30 - 3rd May 2026

- Added `tests/test_enum_parsing.py`:
  - `test_parse_fallback_converts_enum_names`: verifies `parse()` falls back to enum-name conversion when direct validation fails.
  - `test_parse_direct_path_succeeds`: verifies `parse()` succeeds without fallback when values are already correct.
  - `test_parse_raises_when_fallback_also_fails`: verifies `parse()` raises `ParseError` when both paths fail.


### v3.18.29 - 2nd May 2026

- Refactored `src/agents7/`:
  - Added `eval/` subpackage grouping evaluation concerns:
    - `eval/metrics.py`: canonical home for `EvalResult`,
      `run_evaluation`, `run_evaluation_async`,
      `compute_metrics`, `build_confusion_matrix`,
      `_resolve_enum`, `_extract_fields`, `_compare_field`.
    - `eval/export.py`: canonical home for `export_jsonl`,
      `export_csv`, `export_dataframe`, `export_eval_report`,
      `export_benchmark_report`.
    - `eval/__init__.py`: re-exports all public symbols.
  - `metrics.py` and `export.py` are now backward-compat
    shims that re-export from `agents7.eval.*`.
  - Updated internal imports in `testing.py`,
    `ui/evaluation/metrics.py`, `ui/evaluation/runner.py`,
    and `model_selection.py` to use canonical paths.


### v3.18.28 - 2nd May 2026

- Refactored `metrics.py` and `export.py`:
  - Added `_resolve_enum()` to `metrics.py`: shared helper
    that returns `val.name` for enum members, otherwise `val`.
  - Updated `_extract_fields()` in `metrics.py` to use
    `_resolve_enum`; simplified to a dict comprehension.
  - Updated `_format_value()` in `export.py` to import and
    use `_resolve_enum`; eliminates the duplicated
    `hasattr(val, "name")` pattern.


### v3.18.27 - 2nd May 2026

- Refactored `agents/config_loader.py`:
  - Removed chatter-family loaders; now contains only shared
    utilities and the labeler-family loader (`load_yaml_settings`).
- Added `agents/chatter_config_loader.py`: chatter-family loaders
  extracted from `config_loader.py`:
  - `validate_chatter_yaml_config`
  - `load_chatter_yaml_settings`
  - `_derive_persona_summary`
  - `_load_base_chatter_config`
  - `_build_guard`
  - `load_guarded_yaml_settings`
  - `load_guarded_scene_yaml_settings`
- Updated callers:
  - `agents/chatter.py`
  - `agents/personality.py`
  - `agents/scene.py`
  - `agents/guarded.py`
  - `agents/guarded_scene.py`
  - `tests/test_guarded_chatter.py`
  - `tests/test_guarded_scene.py`


### v3.18.26 - 2nd May 2026

- Refactored `agents/detector.py`:
  - Extracted `_validate_output_model()` method so subclasses can
    override it without hacking the MRO.
- Refactored `agents/classifier.py`:
  - Overrides `_validate_output_model()` instead of using
    `super(Detector, self).__post_init__()` to skip Detector's check.
  - `__post_init__` now calls `super().__post_init__()` normally;
    `_possible_reasons` is inherited from `Detector.__post_init__`.


### v3.18.25 - 2nd May 2026

- `agents/`:
  - Refactored `personality.py`:
    - Added `PersonalityChatter._build_system_prompt()` override point so subclasses can change prompt construction without duplicating `__post_init__` plumbing.
  - Refactored `scene.py`:
    - Extracted `_build_scene_system_prompt()` module-level helper (shared by `SceneChatter` and `GuardedSceneChatter`).
    - `SceneChatter` now overrides `_build_system_prompt()` instead of calling `Chatter.__post_init__()` directly; fragile super-chain skip removed.
  - Refactored `guarded_scene.py`:
    - `GuardedSceneChatter` now overrides `_build_system_prompt()` using the shared helper instead of inlining duplicate scene-prompt logic; fragile direct `Chatter.__post_init__()` call removed.
  - Updated `classifier.py`:
    - Added a comment explaining why `super(Detector, self).__post_init__()` is necessary (`ClassificationProperty` is not a `Property` subclass).


### v3.18.24 - 2nd May 2026

- Refactored `llm/base.py`: extracted dispatch pipeline into `llm/dispatcher.py`.
  - Added `llm/dispatcher.py` (`EngineDispatcher`): owns cache helpers, cost injection, fallback handlers, and `execute`/`execute_async`/`stream`/`stream_async` dispatch methods.
  - `LLMEngine` is now a thin orchestrator: creates providers and `EngineDispatcher`, exposes public API, and proxies `_cache`/`_fallback_provider` as read-only properties.
  - Updated `src/agents7/testing.py` (`create_mock_engine`): constructs `EngineDispatcher` instead of directly setting `_cache`/`_fallback_provider`.
  - Updated `tests/helpers.py` (`make_engine`): same fix.


### v3.18.23 - 2nd May 2026

- Refactored `agents/labeler.py`: split into focused collaborators.
  - Added `agents/retry.py` (`ParseRetryHandler`): encapsulates sync/async parse-retry logic with `retry_sync()` and `retry_async()`.
  - Added `agents/batch.py` (`BatchProcessor`): encapsulates sync/async batch execution, chunking, prompting, parsing, and flattening.
  - `Labeler.__post_init__` now creates `_retry_handler` and `_batch_processor` and delegates accordingly; all public methods unchanged.
  - Updated `tests/test_parse_retry.py`: `test_parse_retry_message_content` now calls `ParseRetryHandler._retry_message` directly.


### v3.18.22 - 2nd May 2026

- Fixed bare `except Exception:` blocks that swallowed errors silently:
  - `agents/pipeline.py`: added `exc_info=True` to all four mapper-failure `logger.error` calls; converted f-string log messages to `%`-style.
  - `metrics.py`: added `logger.warning` with `exc_info=True` before batch→sequential fallback in `run_evaluation` and `run_evaluation_async`; added missing `logger` import.
  - `llm/base.py`: added `logger.warning` with `exc_info=True` in `_fallback_sync` and `_fallback_async` before re-raising the primary error.
  - `agents/parsing.py`: added `logger.debug` for the direct-parse→enum-conversion fallback path; added `exc_info=True` to both `logger.error` calls.


### v3.18.21 - 2nd May 2026

- Optimised `ui/_pages/1_Define.py` and `ui/_pages/2_Configure.py`: wrapped `discover_schema()` and `generate_prompt()` calls with `ThreadPoolExecutor` — offloads blocking LLM I/O off the Streamlit worker thread.


### v3.18.20 - 2nd May 2026

- Optimised `agents/memory.py` (`JsonFileMemory`): added `_MAX_CACHE = 100` cap with insertion-order eviction — prevents unbounded `_cache` dict growth in long UI sessions.


### v3.18.19 - 2nd May 2026

- Optimised `ui/app.py` and `ui/_pages/3_Test.py`: replaced unbounded `prompt_versions` list with `deque(maxlen=50)` — caps memory at 50 saved versions and auto-evicts oldest when full. Updated indexed deletion to use `del d[i]`.


### v3.18.18 - 2nd May 2026

- Optimised `export.py` (`_append_confusion_matrix`): replaced per-row `lines.append()` loop with a single `lines.extend()` generator expression — avoids N individual list appends for confusion matrix rows.


### v3.18.17 - 2nd May 2026

- Optimised `ui/helpers.py` (`_state_cache_key`): removed `sort_keys=True` from `json.dumps` — all fields in the state dict are constructed in stable insertion order, so recursive key sorting on every Streamlit interaction was unnecessary overhead.


### v3.18.16 - 2nd May 2026

- Optimised `ui/_pages/__init__.py` (`safe_index`): replaced `list.index()` O(n) lookup with a `{value: index}` dict comprehension — O(1) for large option lists on every Streamlit re-render.


### v3.18.15 - 2nd May 2026

- Optimised `agents/chatter.py` (`Chatter`):
  - Added `_turns` counter initialised in `__post_init__` (counting assistant messages in loaded history).
  - `_append_assistant` increments `_turns` in O(1) instead of scanning history.
  - `_trim_history` decrements `_turns` by the number of assistant messages removed from the front.
  - `reset()` zeroes `_turns`.
  - `turns` property now returns `_turns` directly — O(1) instead of O(n).


### v3.18.14 - 2nd May 2026

- Optimised `ui/comparison/runner.py` (`run_comparison`):
  - Runs version A and version B `_run_single()` calls concurrently via `ThreadPoolExecutor(max_workers=2)`, halving wall-clock time for each comparison input.


### v3.18.13 - 2nd May 2026

- Optimised `llm/cache.py` (`_CacheKeyBuilder.make_key`):
  - Pre-serialises `output_format` dict with `sort_keys=True` once instead of sorting the entire nested payload on every LLM call.
  - Removes `sort_keys=True` from the outer `json.dumps`, relying on stable insertion order for all fixed-key fields.


### v3.18.12 - 2nd May 2026

- Fixed `llm/cache.py` and `agents/memory.py`:
  - Added inline comment explaining that `check_same_thread=False` is safe because all public methods acquire `self._lock` before touching the SQLite connection.
- Fixed `agents/memory.py` `JsonFileMemory._conv_path()`: replaced `str.isalnum()` check with an explicit allowlist `frozenset` for clearer intent and stronger path sanitisation.


### v3.18.11 - 2nd May 2026

- Fixed `benchmarks/benchmark_guarded_scene.py`: set stdout to UTF-8 via
  `sys.stdout.reconfigure(encoding="utf-8")` in `main()` — prevents
  `UnicodeEncodeError` when a model response contains emoji on Windows.


### v3.18.10 - 2nd May 2026

- Updated `benchmarks/benchmark_guarded_scene.py`:
  - Added `--model` / `--guard-model` CLI args to override
    models without editing the YAML config.
  - Added `## Full Responses` section to output — prints the
    complete response text for every case for human validation.
  - Removed 60-char snippet column from the summary table.
- Updated `.claude/skills/validate-guarded-scene/SKILL.md`:
  - Multi-model support: runs benchmark sequentially across a
    configurable list of models (default: `gemma4:e4b`,
    `qwen3:8b`).
  - Updated report template to include full response text per
    case per model in `## Results — <model>` sections.
  - Updated Phase 3 state-tracking to chain models via wakeup
    prompts.


### v3.18.9 - 2nd May 2026

- Fixed `benchmarks/benchmark_guarded_scene.py`:
  - Removed `"tokio"` from case 9 bad_signals — the signal was too broad and caught valid echo-redirects like "¿Tokio? No sé a qué viene eso." Genuine engagement is still caught by `"sushi"`, `"ramen"`, `"yakitori"`, and `"te recomiendo"`.
- Updated `agents/context_guard.py` (`_DEFAULT_GUARD_PROMPT`):
  - Added ACCEPT example for echo-redirects: character mentions the off-topic word only to dismiss it, with no food/travel content → IN_CHARACTER.


### v3.18.8 - 2nd May 2026

- Fixed `agents/context_guard.py` (`_DEFAULT_GUARD_PROMPT`):
  - Added constraint to Decision Rules: `fix` responses must not use prohibited AI/LLM vocabulary (IA, AI, language model, modelo de lenguaje, etc.) even when expressing confusion. Addresses benchmark v2 case 8 where the guard's own fix echoed "modelo de lenguaje".
- Added `tests/test_context_guard.py`:
  - `test_context_guard_prompt_fix_vocabulary_constraint`: verifies the fix-vocabulary constraint is present in the prompt.


### v3.18.7 - 2nd May 2026

- Moved `benchmarks/benchmark_guarded_scene.py` (was `examples/chatter/`): benchmark scripts are developer validation tools, not usage examples — moved to a dedicated `benchmarks/` directory. Updated `_CONFIG` path and run-command in docstring. Updated `.claude/skills/validate-guarded-scene/SKILL.md` to reference the new location.


### v3.18.6 - 2nd May 2026

- Added `.claude/skills/validate-guarded-scene/SKILL.md`: new skill for full `GuardedSceneChatter` validation — pre-checks guard prompt and scene config, runs the benchmark as a background task, polls via `ScheduleWakeup`, writes a dated report to `docs/benchmarks/`, backlogs new findings, and commits the report.


### v3.18.5 - 2nd May 2026

- Updated `agents/context_guard.py` (`_DEFAULT_GUARD_PROMPT`):
  - Added four OFF_TOPIC examples (two REJECT, two ACCEPT) covering geography trivia and tourism questions, so the guard correctly flags scene-break responses.
- Updated `examples/configs/demo_guarded_scene_chatter.yml`:
  - Added scene rule: character redirects off-topic questions (tourism, trivia, general knowledge) back to the story instead of engaging.
- Added `tests/test_context_guard.py`:
  - `test_context_guard_prompt_has_off_topic_examples`: verifies the OFF_TOPIC examples are present in the guard prompt.


### v3.18.4 - 2nd May 2026

- Fixed `agents/context_guard.py` (`ContextGuard`):
  - Added `_agent_instruction()` override: tells the model to respond with a JSON object only and not to reproduce the schema definition. Addresses guard parse failures where the model echoed the textual schema back instead of producing JSON.
- Added `tests/test_context_guard.py`:
  - `test_context_guard_agent_instruction_requires_json`: verifies the instruction explicitly mentions JSON and prohibits schema reproduction.


### v3.18.3 - 2nd May 2026

- Rotated changelog: archived 29 entries to `changelog/2026.md`, keeping 30.



### v3.18.2 - 2nd May 2026

- Fixed `agents/config_loader.py`:
  - `_derive_persona_summary()`: now accepts an optional `Scene`; when provided, includes setting, situation, and scene rules in the summary so the guard can evaluate scene-specific OFF_TOPIC and rule violations.
  - `_build_guard()`: now accepts an optional `scene` parameter and passes it to `_derive_persona_summary`.
  - `load_guarded_scene_yaml_settings()`: passes the loaded `Scene` to `_build_guard` — previously the guard had no scene context at all.
- Updated `agents/context_guard.py` (`_DEFAULT_GUARD_PROMPT`):
  - Renamed `## Character Summary` to `## Character and Scene Context`.
  - Tightened BROKE_CHARACTER criterion: now specifies AI/LLM-specific vocabulary explicitly instead of the vague "acknowledges being artificial in ANY way."
  - Fixed PROMPT_INJECTION false positives: criterion now requires the character **complied** with the injection; refusing/ignoring/being confused is explicitly IN_CHARACTER.
  - Added clearer examples distinguishing compliance from confusion/refusal for both BROKE_CHARACTER and PROMPT_INJECTION.
- Added `tests/test_guarded_scene.py`:
  - `test_derive_persona_summary_includes_scene_context`: verifies scene setting and rules appear in the derived summary.
  - `test_derive_persona_summary_without_scene_has_no_scene_keys`: verifies plain persona summary has no scene keys.
  - `test_guard_retry_succeeds_early`: verifies the retry loop terminates as soon as the guard accepts.
  - `test_guard_input_has_correct_structure`: verifies guard receives `user_message`, `assistant_response`, `recent_history`.
  - `test_guard_called_once_on_immediate_accept`: verifies no extra guard calls when first check passes.
  - `test_from_yaml_builds_guarded_scene`: extended with assertion that YAML-loaded guard includes scene context.
- Added `tests/test_context_guard.py`:
  - `test_context_guard_prompt_includes_scene_summary`: scene-aware persona_summary renders inside the prompt.
  - `test_context_guard_prompt_uses_scene_context_header`: verifies new prompt header.
- Added `examples/chatter/benchmark_guarded_scene.py`: benchmark script for `GuardedSceneChatter` covering 10 cases across five categories (normal, source, injection, char_break, scene_break).


### v3.18.1 - 26th April 2026

- Added `examples/README.md`: documents all 17 examples across
  `agents/`, `chatter/`, `features/`, and `pipelines/` — expected
  result and model-choice analysis for each.


### v3.18.0 - 26th April 2026

- Optimised `llm/streaming.py`:
  - `_StreamBase._finish()`: pre-computes `_response_cache` and `_thinking_cache` once on stream completion.
  - `response` and `thinking` properties: return cached strings on repeated access after stream is done.
  - `_restore_from_cached()`: populates both caches for pre-filled instances.
- Optimised `llm/cache.py` (`SqliteCache`):
  - Replaced per-call `SELECT COUNT(*)` with an in-memory `_approx_size` counter, incremented only for new keys.
  - Eviction query runs only when the counter exceeds `max_size`; `clear()` resets the counter.
- Optimised `agents/memory.py`:
  - `SqliteMemory.save()`: appends only new messages when the conversation has grown; falls back to full replace only when messages are removed or on first save.
  - `SqliteMemory.clear()`: removes the conversation from the in-memory length tracker.
  - `JsonFileMemory`: switched from a single flat JSON file to per-conversation files in a directory. Each conversation is stored as `{dir}/{id}.json`. Default path changed from `"memory.json"` to `"memory/"`.
- Optimised `agents7.ui` — `ui/helpers.py`:
  - Added `get_cached_agent()`: wraps `build_agent_from_state()` with `@st.cache_resource`, keyed on an MD5 hash of the agent-construction fields.
  - Updated `_pages/2_Configure.py`, `3_Test.py`, and `4_Evaluate.py` to call `get_cached_agent()` instead of rebuilding the agent on every Streamlit interaction.


### v3.17.7 - 25th April 2026

- `agents/`:
  - Refactored `guarded.py`:
    - Extracted `_run_guard_retry()` and `_run_guard_retry_async()` as parametric loop helpers taking `check_fn` / `generate_fn` callables.
    - Reduced `_guard_and_fix()` and `_guard_and_fix_async()` to thin delegates that wire the correct sync/async callables.
  - Refactored `labeler.py`:
    - Inlined `_retry_parse_loop()` and `_retry_parse_loop_async()` into `_retry_parse_sync()` / `_retry_parse_async()`, removing the delegation layer and the `chat_fn` parameter abstraction.
  - Fixed `detector.py`, `classifier.py`, `regressor.py`:
    - Converted `from_yaml()` docstrings from NumPy style (`Args:` / `Returns:`) to project-standard Sphinx/reST (`:param:` / `:return:`).


### v3.17.6 - 25th April 2026

- Refactored `agents/scene.py` — `_load_scene_yaml()`:
  - Replaced manual `_resolve_inheritance` / `validate_chatter_yaml_config` / persona / extra_instructions setup with a call to `_load_base_chatter_config()`.
  - Replaced 6 inline `if "field" in raw:` blocks with a single `cast_optional_fields()` call.


### v3.17.5 - 25th April 2026

- Refactored `agents/config_loader.py`:
  - Added `_load_base_chatter_config()`: shared setup for Chatter-family YAML loaders (resolve, validate, persona, extra_instructions).
  - Added `_build_guard()`: shared ContextGuard construction for guarded loaders.
  - Simplified `load_guarded_yaml_settings()` and `load_guarded_scene_yaml_settings()` to use the new helpers, removing ~35 lines of duplicated code.


### v3.17.4 - 25th April 2026

- Removed `from __future__ import annotations` from all remaining 35 source files:
  - Fixed forward-reference return annotations in `agents/`, `llm/`, and `ui/` modules by quoting class names where the class is not yet in scope at definition time.


### v3.17.3 - 25th April 2026

- Removed `from __future__ import annotations` from:
  - `agents/chatter.py`
  - `agents/context_guard.py`
  - `agents/guarded.py`
  - `agents/guarded_scene.py`
- Quoted self-referential return types in `from_yaml` classmethods in `context_guard.py`, `guarded.py`, and `guarded_scene.py`.


### v3.17.1 - 12th April 2026

- `examples/`:
  - Added `chatter/06_guarded_scene_chatter.py`: demo
    combining David García (Spanish journalist) with a
    Madrid café interview scene and the context guard.
  - Added `configs/demo_guarded_scene_chatter.yml`:
    matching YAML config.


### v3.17.0 - 12th April 2026

- `src/agents7/agents/`:
  - Added `guarded_scene.py` with `GuardedSceneChatter`:
    combines `SceneChatter`'s scene-based role-play framing
    with `GuardedChatter`'s context guard and retry/fix
    pipeline.
  - Updated `__init__.py` to export `GuardedSceneChatter`.
  - Updated `config_loader.py`: added
    `load_guarded_scene_yaml_settings()` for loading
    `GuardedSceneChatter` from YAML.


### v3.16.0 - 12th April 2026

- `src/agents7/export.py`:
  - Added `export_benchmark_report()`: writes a Markdown
    report from a `BenchmarkResult` with model scores and
    per-case tables.
- `src/agents7/model_selection.py`:
  - Added `report_path` parameter to `find_best_model()`:
    when set, auto-writes a Markdown benchmark report after
    benchmarking completes.


### v3.15.17 - 12th April 2026

- `integration_tests/test_multi_model.py`:
  - Rewritten to single-call connectivity test per model.
    Now covers all 5 models used in configs: `qwen2.5:3b`,
    `qwen3:1.7b`, `qwen3:4b`, `gemma3:12b`, `gemma4:e4b`.
  - Asserts structure (model parsed, metrics present) not
    classification correctness.
- `.claude/rules/testing.md`:
  - Updated integration test scope: every model in
    `examples/configs/*.yml` must have a connectivity test.


### v3.15.16 - 12th April 2026

- `integration_tests/`:
  - Removed `test_guarded_chatter.py` — tested feature
    behavior (injection resistance, character breaks) not
    LLM integration. Same `engine.chat()` path as Chatter.
    Reclaims ~180s of the 5-minute test budget.


### v3.15.15 - 12th April 2026

- `integration_tests/`:
  - Removed `test_scene_chatter.py` — SceneChatter uses
    the same `engine.chat()` path as Chatter; no new
    integration to verify. Feature-level behavior tests
    don't belong in integration tests.


### v3.15.14 - 12th April 2026

- `integration_tests/test_scene_chatter.py`:
  - Added 6 integration tests for `SceneChatter` using
    `demo_scene_chatter.yml` (qwen2.5:3b). Tests cover:
    in-character responses, modern concept rejection,
    other character awareness, and goal pursuit.


### v3.15.13 - 12th April 2026

- `integration_tests/integration_helpers.py`:
  - Added `run_single_case()` and `run_batch_and_report()`
    to eliminate duplicated test boilerplate.
  - Renamed from `helpers.py` to avoid shadowing the
    unit-test `helpers.py` on the pytest path.
- `integration_tests/`:
  - Simplified `test_detector.py`, `test_classifier.py`,
    and `test_regressor.py` to use shared helpers.


### v3.15.12 - 11th April 2026

- `tests/`:
  - `conftest.py`:
    - Removed plain helper functions (`mock_metrics`,
      `make_agent_mock`, `MockModel`, `make_response`).
      Now contains only pytest fixtures.
  - `helpers.py`:
    - Moved UI test helpers from `conftest.py`. Now the
      single location for all non-fixture test utilities.
  - Updated `test_ui_comparison.py` and
    `test_ui_evaluation.py` to import from `helpers`.


### v3.15.11 - 11th April 2026

- `tests/helpers.py`:
  - Added `chat_response_dict()`, `make_chatter()`,
    `make_guard()`, and `PERSONA_SUMMARY` constant.
- `tests/`:
  - Removed duplicated `chat_response_dict` from 4 files,
    `make_chatter` from 3 files, and `make_guard` from
    2 files. All now import from `helpers.py`.


### v3.15.10 - 11th April 2026

- `agents/config_loader.py`:
  - Moved `load_guarded_yaml_settings()` and
    `_derive_persona_summary()` from `guarded.py`,
    consolidating all YAML loading in one module.
- `agents/guarded.py`:
  - Removed `_load_guarded_yaml` and
    `_derive_persona_summary` (now in `config_loader`).


### v3.15.9 - 11th April 2026

- `agents/guarded.py`:
  - Extracted `_try_accept()` and `_use_fix_or_text()`
    from `_guard_and_fix` / `_guard_and_fix_async`,
    eliminating duplicated verdict-decision and fallback
    logic between the sync and async variants.


### v3.15.8 - 11th April 2026

- `agents/chatter.py`:
  - Moved `__iter__`, `__aiter__`, and `_finalize_async`
    into `_ChatterStreamBase`, eliminating duplication
    from all four stream wrapper subclasses.
- `agents/guarded.py`:
  - Simplified `GuardedChatterStreamResult` and
    `GuardedChatterAsyncStreamResult` to only override
    finalize hooks. Removed redundant `_guard`/`_user_msg`
    slots and constructor parameters.


### v3.15.7 - 11th April 2026

- `llm/base.py`:
  - Extracted `_create_provider()` factory method from
    `LLMEngine.__init__`, replacing nested if/elif
    dispatch with a clean factory that also handles
    fallback provider creation.
  - Added `ValueError` for unknown provider names.


### v3.15.6 - 11th April 2026

- `llm/`:
  - `base.py`:
    - Moved `_prompt_to_messages()`, `generate()`,
      `generate_async()`, `generate_stream()`, and
      `generate_stream_async()` from abstract to default
      implementations in `LLMProvider`.
  - `anthropic.py`:
    - Removed 6 methods now inherited from `LLMProvider`.
  - `openai_compat.py`:
    - Removed 6 methods now inherited from `LLMProvider`.


### v3.15.5 - 11th April 2026

- `agents/`:
  - `config_loader.py`:
    - Added `cast_optional_fields()`: shared helper that
      replaces repetitive `if "field" in config` blocks
      across YAML loaders.
  - `personality.py`:
    - Replaced 6 field-casting blocks with
      `cast_optional_fields()` call.
  - `guarded.py`:
    - Replaced 8 field-casting blocks with
      `cast_optional_fields()` call.


### v3.15.4 - 11th April 2026

- `.claude/rules/`:
  - Decoupled `/refactor` rule: canonical
    `refactoring.md` is now procedural only.
  - Added `refactoring-areas.md` with
    project-specific code smells to watch.
- `.claude/skills/refactor/`:
  - Updated `SKILL.md` to read both canonical
    procedure and per-project areas.


### v3.15.3 - 11th April 2026

- `.claude/rules/`:
  - Decoupled `/optimize` rule: canonical
    `optimization.md` is now procedural only.
  - Added `optimization-areas.md` with
    project-specific performance areas.
- `.claude/skills/optimize/`:
  - Updated `SKILL.md` to read both canonical
    procedure and per-project areas.


### v3.15.2 - 10th April 2026

- `.claude/rules/`:
  - Decoupled `/improvements` rule: canonical
    `improvements.md` is now procedural only.
  - Added `improvement-areas.md` with
    project-specific areas to watch.
- `.claude/skills/improvements/`:
  - Updated `SKILL.md` to read both canonical
    procedure and per-project areas.


### v3.15.1 - 5th April 2026

- `examples/chatter/05_scene_chatter.py`:
  - Added interactive example: medieval tavern keeper
    scenario testing scene rules, context awareness, and
    character goal.
- `backlog.md`:
  - Added SceneChatter integration test item.


### v3.15.0 - 5th April 2026

- New module `agents7.agents.scene`:
  - `SceneChatter`: extends `PersonalityChatter` with a
    structured `Scene` that frames the conversation as a
    role-play in a specific context (setting, situation,
    time period, rules, goal, other characters).
- New model `Scene` in `agents7.agents.personality`:
  - `Scene`: frozen dataclass with `to_prompt()` and
    `from_dict()` for YAML loading.
- `agents7.agents.__init__`:
  - Exported `Scene` and `SceneChatter`.
- `tests/test_scene.py`:
  - 17 unit tests covering `Scene`, `SceneChatter`, and
    YAML loading.
- `examples/configs/demo_scene_chatter.yml`:
  - YAML example: medieval tavern keeper scenario.
- `docs/`:
  - Updated `chatter.md`: added SceneChatter section.
  - Updated `design.md`: added SceneChatter to hierarchy.
- `README.md`:
  - Added SceneChatter section with usage example.


### v3.14.10 - 5th April 2026

- `changelog.md`:
  - Merged root `changelog.md` into `agents7_framework/changelog.md`
    — eliminated duplicate/diverged changelog files.
  - Rotated: kept last 30 entries, archived older entries
    into `changelog/2026.md` and `changelog/2025.md`.
  - Deleted root `changelog.md`.


