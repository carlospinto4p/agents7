
## Changelog - Agents7

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


### v3.14.9 - 5th April 2026

- `.claude/rules/`:
  - Updated `versioning.md`: added changelog
    rotation section (30-version limit, yearly
    archives in `changelog/YYYY.md`).


### v3.14.8 - 5th April 2026

- `.claude/rules/`:
  - Updated `versioning.md`: added changelog
    rotation section (30-version limit, yearly
    archives in `changelog/YYYY.md`).


### v3.14.7 - 5th April 2026

- `.claude/rules/versioning.md`:
  - Added "Reading changelogs" rule: never read the full
    file, use `limit: 5` or `Grep` to minimize token cost.


### v3.14.6 - 5th April 2026

- `integration_tests/test_guarded_chatter.py`:
  - Added 7 integration tests for `GuardedChatter` with
    `gemma4:e4b`: 2 normal, 2 injection, 2 break, and
    1 guard callback test.
  - Tests retry once on failure to handle LLM
    non-determinism. 120s per-test timeout.


### v3.14.5 - 5th April 2026

- `.claude/`:
  - Updated `backlog` skill (v1.4.0): tables now
    always include Priority and Effort columns.
- `docs/benchmarks/`:
  - Added Run 5 to GuardedChatter benchmark: think vs
    no-think across qwen3:4b, gemma4:e4b, gemma4:e2b.
  - qwen3:4b root cause: chatter complies with injections
    regardless of think mode (4/6 both). Unsuitable.
  - gemma4:e4b: think=true is 6/6, think=false is 5/6.
  - gemma4:e2b: think=false is 6/6, think=true is 5/6.
  - Updated recommendation with per-model think settings.
- `examples/configs/demo_guarded_chatter.yml`:
  - Updated guard config comment with think guidance and
    e2b alternative.


### v3.14.4 - 5th April 2026

- `.claude/hooks/`:
  - Fixed stdin consumption: all hooks now
    capture stdin before piping to python.


### v3.14.3 - 5th April 2026

- `.claude/`:
  - Updated `backlog` skill (v1.3.0): auto-cleans
    completed items before display, shows per-section
    tables when backlog has multiple sections.
  - Updated `backlog` rule: added auto-cleanup
    section.
- `docs/benchmarks/`:
  - Added Run 4 to GuardedChatter benchmark: `gemma4:e2b`
    variants. `e2b (same)` scores 6/6 (15.7s avg) — ties
    with `e4b` quality at half the VRAM. Cross-model e2b/e4b
    combos are slow (52–157s) due to model swap overhead.
  - Updated recommendation: use same model for both roles,
    `e4b` for best precision, `e2b` for VRAM-constrained
    setups.


### v3.14.2 - 5th April 2026

- `.claude/`:
  - Updated `backlog` skill (v1.1.0): auto-cleans
    completed items when 5+ accumulate.
  - Updated `backlog` rule: added auto-cleanup
    section.


### v3.14.1 - 5th April 2026

- `docs/benchmarks/`:
  - Added Run 3 to GuardedChatter benchmark: `gemma4:e4b`
    guard with `qwen3:4b` (5/6, slow) and `gemma3:4b`
    (4/6, fast but too compliant) as chatters. Confirms
    `gemma4:e4b` for both roles remains the best combo.
- `backlog.md`:
  - Discarded qwen3.5 re-benchmark item.
  - Marked cross-model guard benchmark as done.


### v3.14.0 - 4th April 2026

- `agents7.models.base`:
  - Added optional `confidence` field to
    `ClassificationProperty` (0.0–1.0 float, defaults to
    `None`). All classifier agents can now carry a
    confidence score without breaking existing code.
- `tests/test_classifier.py`:
  - Added tests for confidence default, assignment, and
    out-of-range validation.


### v3.13.7 - 4th April 2026

- `docs/`:
  - Added `benchmark_guarded_chatter.md`: full benchmark
    of 6 model combos for `GuardedChatter`. `gemma4:e4b`
    scored 6/6 with 2.2s avg guard time.
  - Updated `chatter.md` model selection section with
    benchmark findings and link.
- `examples/configs/demo_guarded_chatter.yml`:
  - Changed both chatter and guard to `gemma4:e4b`
    (benchmark winner, avoids model-swap overhead).


### v3.13.6 - 4th April 2026

- Module `agents7.agents.guarded`:
  - Fixed `AttributeError` crash when guard LLM returns
    unparseable output — `_guard_check` now returns
    `(verdict | None, duration)` tuple; `None` verdict
    triggers fail-open (response passes through).
  - Added `_is_accepted()` helper for None-safe checks.
  - Added `duration` and `parse_failed` fields to
    `GuardEvent` for observability.
  - YAML loader now passes `think` from the `guard:`
    section to `ContextGuard`.
- `examples/`:
  - `configs/demo_guarded_chatter.yml`: added
    `think: true` to guard config for better reasoning.
  - `chatter/04_guarded_chatter.py`: callback now shows
    guard timing and parse-failure events.
- `docs/chatter.md`:
  - Added model selection guidance (4B+ recommended).
- Tests:
  - Added `test_callback_on_parse_failure` test.


### v3.13.5 - 4th April 2026

- `.claude/`:
  - Added `hooks/block-raw-python.sh`: enforces `uv run python` over bare `python`.


### v3.13.4 - 4th April 2026

- Module `agents7.agents.context_guard`:
  - Rewrote default guard prompt with stricter rejection
    criteria, explicit character-break signals (e.g.
    "I am not an AI" is still a break), few-shot examples,
    and a "when in doubt, REJECT" policy.
- `examples/configs/demo_guarded_chatter.yml`:
  - Changed guard model from `qwen2.5:3b` to `qwen3:4b`
    with comment explaining small models miss breaks.
- `docs/chatter.md`:
  - Added "Model selection" section recommending 4B+
    models for the guard.


### v3.13.3 - 4th April 2026

- Module `agents7.agents.guarded`:
  - Added `GuardEvent` dataclass: emitted on each guard
    check with attempt, accepted, reason, response_text,
    fix, and used_fix fields.
  - Added `guard_callback` field to `GuardedChatter`:
    optional callable invoked with a `GuardEvent` on every
    guard check for debugging and observability.
  - Added `_emit()` helper wired into `_guard_and_fix()`
    and `_guard_and_fix_async()`.
- Exported `GuardEvent` from `agents7.agents`.
- `examples/chatter/04_guarded_chatter.py`:
  - Rewrote to use `guard_callback` for verbose output
    showing each verdict, retry, and fix in real time.
- `docs/chatter.md`:
  - Added "Debugging with `guard_callback`" section with
    `GuardEvent` field reference.
- Tests:
  - Added 3 callback tests in `test_guarded_chatter.py`.


### v3.13.2 - 4th April 2026

- `examples/`:
  - Renamed all example scripts with numbered prefixes
    sorted by hierarchy (base classes first, then
    subclasses and complex workflows):
    - `agents/`: `01_labeler`, `02_detector`,
      `03_classifier`, `04_regressor`.
    - `chatter/`: `01_chatter`, `02_chatter_memory`,
      `03_personality_chatter`, `04_guarded_chatter`.
    - `features/`: `01_streaming`, `02_export`,
      `03_labeler_params`, `04_sqlite_cache`,
      `05_config_inheritance`.
    - `pipelines/`: `01_pipeline`,
      `02_conditional_pipeline`.
- Documentation updates:
  - `README.md`: updated examples section with numbered
    names and grouped by directory.
  - `docs/chatter.md`: updated example references.


### v3.13.1 - 4th April 2026

- `examples/`:
  - Added `configs/demo_guarded_chatter.yml`: YAML config
    for a guarded David persona with ContextGuard.
  - Added `chatter/guarded_chatter.py`: example showing
    normal conversation, prompt injection attempts, and
    character break attempts with GuardedChatter.
- Documentation updates:
  - `docs/chatter.md`: added example reference, updated
    hierarchy diagram.
  - `docs/design.md`: updated hierarchy to include
    `ContextGuard` and `GuardedChatter`.
  - `README.md`: updated agents layer hierarchy diagram.


### v3.13.0 - 4th April 2026

- New module `agents7.agents.context_guard`:
  - `GuardReason` — enum: `IN_CHARACTER`, `PROMPT_INJECTION`,
    `OFF_TOPIC`, `BROKE_CHARACTER`.
  - `GuardVerdict(Property[bool])` — output model with `value`,
    `reason`, and optional `fix` field for suggested rewrites.
  - `ContextGuard(Detector)` — post-processing guard that
    evaluates conversational responses against a persona.
    Auto-generates system prompt from `persona_summary`.
- New module `agents7.agents.guarded`:
  - `GuardedChatter(PersonalityChatter)` — wraps every response
    in a `ContextGuard` check. Rejected responses are retried
    with feedback (up to `max_guard_retries`); if all retries
    fail, the guard's suggested `fix` is used. Rejected
    responses never enter conversation history.
  - `GuardedChatterStreamResult` / `GuardedChatterAsyncStreamResult`
    — streaming wrappers that run the guard after stream
    completion. `chat_response.response` is the authoritative
    (possibly corrected) text.
  - YAML loading with nested `guard:` section;
    `persona_summary` auto-derived from persona.
- `exceptions.py`:
  - Added `GuardRejectionError`, `GuardConfigurationError`.
- Exported from `agents7.agents`:
  - `ContextGuard`, `GuardReason`, `GuardVerdict`,
    `GuardedChatter`, `GuardedChatterStreamResult`,
    `GuardedChatterAsyncStreamResult`.
- `docs/chatter.md`:
  - Added `GuardedChatter` section with guard pipeline,
    retry flow, streaming, YAML config, and cost notes.
- Tests:
  - `tests/test_context_guard.py`: 13 tests.
  - `tests/test_guarded_chatter.py`: 15 tests.


### v3.12.16 - 4th April 2026

- Module `agents7.agents.personality`:
  - Updated `Persona.to_prompt()`: instructions now explicitly
    counteract the LLM's "helpful assistant" tone — the agent
    won't offer help, use customer-service language, or ask
    what it can do for the user. Responds as the character
    naturally would.
- `docs/chatter.md`:
  - Updated auto-generated prompt section to document the
    anti-servile instructions.


### v3.12.15 - 4th April 2026

- `docs/`:
  - Added `chatter.md`: dedicated documentation for `Chatter`
    and `PersonalityChatter` — fields, methods, streaming,
    memory backends, persona configuration, YAML examples.
  - Updated `design.md`: expanded agent hierarchy to include
    `Chatter` and `PersonalityChatter` branches.


### v3.12.14 - 4th April 2026

- `.claude/rules/`:
  - Normalized `versioning.md` to enhanced canonical
    with detailed sub-bullet guidance.


### v3.12.13 - 3rd April 2026

- `.claude/rules/`:
  - Normalized `committing.md` to canonical template.


### v3.12.12 - 3rd April 2026

- `.claude/rules/`:
  - Updated `committing.md`: added one-cmd-per-bash, removed no-coauthor.


### v3.12.11 - 3rd April 2026

- `.claude/rules/`:
  - Updated `committing.md`: added post-change workflow
    section.


### v3.12.10 - 3rd April 2026

- `CLAUDE.md`:
  - Normalized to canonical template: added missing
    shared sections, removed low-value sections.


### v3.12.9 - 3rd April 2026

- `.claude/`:
  - Migrated `/refine-tagger-agent` from command to skill (v1.0.0)
    for version tracking.


### v3.12.8 - 3rd April 2026

- `.claude/`:
  - Migrated `/self-refinement` from command to skill
    (v1.0.0) for version tracking.


### v3.12.7 - 3rd April 2026

- `.claude/`:
  - Migrated `/improvements` from command to skill (v1.0.0)
    for version tracking.


### v3.12.6 - 3rd April 2026

- `.claude/`:
  - Migrated `/optimize` from command to skill (v1.0.0)
    for version tracking.


