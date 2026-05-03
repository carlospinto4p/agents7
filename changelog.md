
## Changelog - Agents7

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

- Refactored `agents/personality.py`:
  - Added `PersonalityChatter._build_system_prompt()` override point so subclasses can change prompt construction without duplicating `__post_init__` plumbing.
- Refactored `agents/scene.py`:
  - Extracted `_build_scene_system_prompt()` module-level helper (shared by `SceneChatter` and `GuardedSceneChatter`).
  - `SceneChatter` now overrides `_build_system_prompt()` instead of calling `Chatter.__post_init__()` directly; fragile super-chain skip removed.
- Refactored `agents/guarded_scene.py`:
  - `GuardedSceneChatter` now overrides `_build_system_prompt()` using the shared helper instead of inlining duplicate scene-prompt logic; fragile direct `Chatter.__post_init__()` call removed.
- Updated `agents/classifier.py`:
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

- Refactored `agents/guarded.py`:
  - Extracted `_run_guard_retry()` and `_run_guard_retry_async()` as parametric loop helpers taking `check_fn` / `generate_fn` callables.
  - Reduced `_guard_and_fix()` and `_guard_and_fix_async()` to thin delegates that wire the correct sync/async callables.
- Refactored `agents/labeler.py`:
  - Inlined `_retry_parse_loop()` and `_retry_parse_loop_async()` into `_retry_parse_sync()` / `_retry_parse_async()`, removing the delegation layer and the `chat_fn` parameter abstraction.
- Fixed `agents/detector.py`, `agents/classifier.py`, `agents/regressor.py`:
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

- `tests/conftest.py`:
  - Removed plain helper functions (`mock_metrics`,
    `make_agent_mock`, `MockModel`, `make_response`).
    Now contains only pytest fixtures.
- `tests/helpers.py`:
  - Moved UI test helpers from `conftest.py`. Now the
    single location for all non-fixture test utilities.
- `tests/`:
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

- `llm/base.py`:
  - Moved `_prompt_to_messages()`, `generate()`,
    `generate_async()`, `generate_stream()`, and
    `generate_stream_async()` from abstract to default
    implementations in `LLMProvider`.
- `llm/anthropic.py`:
  - Removed 6 methods now inherited from `LLMProvider`.
- `llm/openai_compat.py`:
  - Removed 6 methods now inherited from `LLMProvider`.


### v3.15.5 - 11th April 2026

- `agents/config_loader.py`:
  - Added `cast_optional_fields()`: shared helper that
    replaces repetitive `if "field" in config` blocks
    across YAML loaders.
- `agents/personality.py`:
  - Replaced 6 field-casting blocks with
    `cast_optional_fields()` call.
- `agents/guarded.py`:
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


