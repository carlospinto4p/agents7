
## Changelog - Agents7

### v1.2.7 - 15th February 2026

- `.claude/rules/app.md`: Fixed optional dependency name from
  `[app]` to `[ui]` and source path from `app/` to `ui/`.
- `.claude/rules/testing.md`: Replaced stale references from
  another project (databases, `db/`) with agents7-
  specific examples (Ollama, detectors, classifiers). Fixed
  integration test commands.
- `CLAUDE.md`: Removed redundant lessons, fixed `.claude/rules/`
  path reference.
- `backlog.md`: Added checkboxes and section headers per backlog
  rules.
- `integration_tests/`: Removed standalone `pyproject.toml` and
  `uv.lock`; integration tests now run from the root project.
- `.claude/commands/`: Added `refactor.md` and
  `self-refinement.md` command files.


### v1.2.6 - 8th February 2026

- Module `agents`:
  - `utils.py`: New shared helpers `extract_enum_members()` and
    `run_with_semaphore()`.
  - `detector.py`, `classifier.py`: Use `extract_enum_members`
    instead of duplicated inline enum extraction.
  - `labeler.py`: Break up `_load_yaml_settings` into
    `_validate_yaml_config`, `_build_engine`, and
    `_parse_yaml_examples`. Use `run_with_semaphore` instead
    of nested `_limited()`.
  - `pipeline.py`: Use `run_with_semaphore` instead of nested
    `_limited()`.
- Module `llm`:
  - `base.py`: Extract `_cache_lookup` and `_cache_store` helpers
    to deduplicate cache logic across 8 methods.
  - `streaming.py`: Extract `_StreamBase` mixin to deduplicate
    shared state and helpers between `StreamResult` and
    `AsyncStreamResult`.
- Module `settings`:
  - Add `frozen=True` to `BatchSettings` and `OllamaSettings`.
  - Add docstring to `OllamaSettings`.
- Replace `ValueError`/`TypeError` with custom
  `AgentConfigurationError`/`PropertyConfigurationError`
  across `labeler.py`, `detector.py`, `classifier.py`,
  `utils.py`.
- Move inline imports to module level in `models/base.py`
  and `pipeline.py`.
- Fix double-backtick docstrings to single backticks across
  9 UI/evaluation files.
- Update test assertions in `test_labeler.py` for custom
  exceptions.
- `.claude/CLAUDE.md`: Add "Propose Manual Review" rule.


### v1.2.5 - 8th February 2026

- Module `ui`:
  - New submodule `comparison/`:
    - `models.py`: `PromptVersion` dataclass for storing
      prompt snapshots (system_prompt, examples, model_name).
    - `runner.py`: `ComparisonResult` dataclass and
      `run_comparison()` function for running two prompt
      versions side-by-side against the same inputs.
    - `__init__.py`: Exported `PromptVersion`,
      `ComparisonResult`, and `run_comparison`.
  - `app.py`: Added `prompt_versions` to session state init.
  - Page `3_Test.py`: Restructured into 4 tabs (Single
    Input, Batch Test, A/B Compare, History). Added prompt
    version management (save/load/delete) and A/B comparison
    tab with side-by-side results and agreement metrics.
- New file `tests/test_ui_comparison.py`: Unit tests for
  `PromptVersion`, `ComparisonResult`, `_build_version_state`,
  and `run_comparison`.
- `README.md`: Updated Test step description.
- `backlog.md`: Removed completed item.


### v1.2.4 - 8th February 2026

- Module `ui`:
  - New file `evaluation/export.py`:
    - `format_results_csv()`: Converts evaluation results to a
      flat CSV string with per-field expected/actual/passed
      columns.
    - `format_results_jsonl()`: Converts evaluation results to
      JSONL (one JSON object per line, excludes heavy `response`).
  - Page `4_Evaluate.py`: Added "Download Results" section with
    CSV and JSONL download buttons after per-case results.
  - Page `5_Export.py`: Exports `eval_results.csv` and
    `eval_results.jsonl` to the workspace when evaluation results
    exist in session state.
  - `evaluation/__init__.py`: Exported `format_results_csv` and
    `format_results_jsonl`.
- New file `tests/test_ui_export.py`: 20 unit tests for CSV and
  JSONL formatters.
- `README.md`: Updated Evaluate and Export step descriptions.
- `backlog.md`: Removed completed item.


### v1.2.3 - 8th February 2026

- Module `ui`:
  - `app.py`: Added `--host` and `--port` CLI flags to the
    `agents7-ui` entry point. Refactored `main()` into a CLI
    launcher and `_render_app()` for the Streamlit rendering.
  - `__init__.py`: `launch()` now accepts optional `host` and
    `port` parameters forwarded to the CLI.
- `README.md`: Documented `--host` / `--port` usage in the
  Agent Builder UI section.
- `backlog.md`: Removed completed item.


### v1.2.2 - 8th February 2026

- Module `settings`:
  - Replaced bare `DEFAULT_MAX_CONCURRENT` constant with
    `BatchSettings` dataclass using `load_setting` pattern.
    Default `max_concurrent` is now overridable via the
    `BATCH_MAX_CONCURRENT` environment variable.
- Module `agents`:
  - Class `Labeler`:
    - `act_in_batch_async`: Replaced `max_concurrent` parameter
      with `batch_settings: BatchSettings | None`.
  - Class `Pipeline`:
    - `act_in_batch_async`: Replaced `max_concurrent` parameter
      with `batch_settings: BatchSettings | None`.
- Tests updated to use `BatchSettings` instances.
- `README.md`: Updated "Batch Concurrency Control" section with
  `BatchSettings` usage.


### v1.2.1 - 8th February 2026

- Module `settings`:
  - New `DEFAULT_MAX_CONCURRENT = 5` constant for library-wide
    default concurrency limit.
- Module `agents`:
  - Class `Labeler`:
    - `act_in_batch_async`: Default `max_concurrent` changed from
      `None` (unlimited) to `DEFAULT_MAX_CONCURRENT` (5).
  - Class `Pipeline`:
    - `act_in_batch_async`: Default `max_concurrent` changed from
      `None` (unlimited) to `DEFAULT_MAX_CONCURRENT` (5).
- `README.md`: Updated "Batch Concurrency Control" table with new
  default.


### v1.2.0 - 8th February 2026

- Module `agents`:
  - Class `Labeler`:
    - `act_in_batch_async`: New `max_concurrent` parameter
      (`int | None`, default `None`) to limit concurrent LLM
      requests via `asyncio.Semaphore`.
  - Class `Pipeline`:
    - `act_in_batch_async`: New `max_concurrent` parameter
      with the same semantics. Items within each step now run
      concurrently via `asyncio.gather` (previously sequential).
- `tests/test_labeler.py`: 3 new tests for semaphore behavior
  (unlimited, limited, sequential).
- `tests/test_pipeline.py`: 2 new tests for concurrent pipeline
  batch processing with semaphore.
- `README.md`: Added "Batch Concurrency Control" section.


### v1.1.1 - 8th February 2026

- Docker support:
  - New `Dockerfile` for the Streamlit Agent Builder UI
    (Python 3.12-slim, installs `agents7[ui]` via uv).
  - New `docker-compose.yml` running the app container and
    connecting to Ollama on the host via `host.docker.internal`.
  - New `.env.example` documenting required `GITLAB_TOKEN`.
  - New `.dockerignore` to keep the build context small.
- `README.md`: Added "Docker" section with usage instructions.
- Module `ui`:
  - Renamed `pages/` to `_pages/` to prevent Streamlit from
    auto-detecting the directory as multipage navigation. Only the
    custom sidebar radio navigation is now active.


### v1.1.0 - 8th February 2026

- Module `ui` (new, optional dependency `agents7[ui]`):
  - New file `state.py`: `AgentBuilderState`, `EnumMember`,
    `FieldDefinition` dataclasses capturing the full wizard state.
  - New submodule `codegen/`:
    - `model_generator.py`: `generate_model_code()` produces `.py`
      source; `create_model_class()` builds live Pydantic classes
      in-memory for all 4 agent types (Classifier, Detector,
      Regressor, Labeler).
    - `yaml_generator.py`: `generate_yaml()` produces YAML config.
  - New submodule `evaluation/`:
    - `runner.py`: `run_evaluation()` executes an agent against
      labeled data, returns `EvalResult` per row.
    - `metrics.py`: `compute_metrics()` for accuracy/per-field
      stats; `build_confusion_matrix()` for classifiers.
  - `ollama_utils.py`: `check_connection()` and
    `list_available_models()` for Ollama auto-detection.
  - `helpers.py`: `build_agent_from_state()` constructs a live
    agent from UI state without writing files.
  - `app.py`: Streamlit entry point with sidebar navigation.
  - `pages/`: 5-step wizard (Define, Configure, Test, Evaluate,
    Export) with live code preview, batch testing, CSV upload,
    and workspace export.
- `pyproject.toml`:
  - Added `[project.optional-dependencies] ui` group
    (streamlit, pandas, plotly).
  - Added `agents7-ui` script entry point.
- `.gitignore`: Added `workspace/`.
- `CLAUDE.md`: Noted optional UI module in project overview.
- `README.md`: Added "Agent Builder UI" documentation section.
- New test files:
  - `tests/test_ui_codegen.py`: 25 unit tests for state,
    code generation, dynamic class creation, YAML generation.
  - `tests/test_ui_evaluation.py`: 19 unit tests for evaluation
    runner, metrics, and confusion matrix.


### v1.0.4 - 8th February 2026

- `integration_tests/`:
  - Trimmed parametrized test data for faster execution: detector
    11 → 4 cases, classifier 5 → 3 cases, regressor 5 → 4 cases
    (one per category/reason). Multi-model tests unchanged.


### v1.0.3 - 8th February 2026

- `integration_tests/`:
  - New `conftest.py` with session-scoped fixture to `chdir` to the
    project root, so YAML config paths resolve correctly.
- `CLAUDE.md`: Updated "After Every Change" workflow to require running
  both unit tests and integration tests before committing.


### v1.0.2 - 8th February 2026

- Tests:
  - Separated integration tests (LLM-calling) from unit tests.
  - New `integration_tests/` standalone project with its own
    `pyproject.toml`, containing tests extracted from
    `test_detector.py`, `test_classifier.py`, `test_regressor.py`,
    and moved `test_async.py`, `test_multi_model.py`.
  - `tests/` now contains only fast unit tests (no Ollama required).
- `README.md`: Updated development section with both test suites.
- `CLAUDE.md`: Clarified unit test workflow in "After Every Change".


### v1.0.1 - 8th February 2026

- `changelog.md`: Fixed inconsistent spacing between version entries
  (v1.0.0 through v0.5.0 had one blank line instead of two).
- `.claude/rules/versioning.md`: Documented the two-blank-lines rule
  for changelog entries.


### v1.0.0 - 1st February 2026

- Module `exceptions`:
  - New `PipelineConfigurationError` exception class.
- Module `agents`:
  - New file `pipeline.py`:
    - `PipelineStep` frozen dataclass: wraps an agent and an
      optional mapper function (`Callable[[AgentResponse], dict]`,
      defaults to `model.model_dump()`).
    - `PipelineResponse` Pydantic model: contains the list of
      step responses, the final model, and a `success` flag.
    - `Pipeline` frozen dataclass: chains multiple agents
      sequentially. Supports `act()`, `act_async()`,
      `act_in_batch()`, `act_in_batch_async()`, and
      `from_yaml()`. Short-circuits on parse failure or mapper
      error.
  - `Pipeline`, `PipelineResponse`, `PipelineStep` exported
    from `agents7.agents`.
  - `import_from_path` return type widened from
    `type[BaseModel]` to `type`.
- Module `models.demo`:
  - New `PersonBio` model for bio generation demo.
- New file `tests/test_pipeline.py`:
  - 21 unit tests (all mocked, no LLM) covering `PipelineStep`
    validation, `Pipeline` init, default mapper, sync/async
    single and batch execution, short-circuit behaviour, and
    YAML configuration parsing.
- Examples:
  - New `pipeline.py` example demonstrating a two-step pipeline
    (PersonExtractor → BioGenerator) with custom mapper, single
    and batch modes.
  - New `configs/demo_bio_generator.yml` agent configuration.
- `README.md`: Added "Agent Pipelines" documentation section,
  updated architecture diagram and agent list.


### v0.8.0 - 1st February 2026

- Module `agents`:
  - Class `Labeler`:
    - New `validators` field: optional list of callables that run
      after the LLM response is parsed. Each receives a BaseModel
      and returns it (possibly modified); raising rejects the model
      (sets it to `None`). Pipeline runs in list order; first
      failure short-circuits.
    - New `_run_validators` method applies the validator pipeline.
    - `act()`, `act_async()`, `act_in_batch()`, and
      `act_in_batch_async()` now pass parsed models through
      validators. In batch mode, validators run per individual item.
    - `_load_yaml_settings` parses optional `validators:` list of
      dotted import paths.
  - Subclasses `Detector`, `Classifier`, `Regressor` inherit
    validators for free via `Labeler`.
- Module `models`:
  - Class `AgentResponse`:
    - `model` field type changed from `BaseModel` to
      `BaseModel | None` (fixes pre-existing type mismatch with
      `_parse_model_response` returning `None`).
- New file `tests/test_validators.py`:
  - 14 unit tests (all mocked, no LLM) covering `_run_validators`
    pipeline, `act`/`act_async` integration, batch validation,
    partial rejection, and YAML config parsing.
- `README.md`: Added "Validation Hooks" documentation section.


### v0.7.0 - 1st February 2026

- Module `llm`:
  - New file `cache.py`:
    - `CacheConfig` frozen dataclass for configuring in-memory LRU cache
      (`max_size`, default 128, 0 = unlimited).
    - `ResponseCache` thread-safe LRU store with `make_key`, `get`,
      `put`, `size`, and `clear`.
  - Class `LLMEngine`:
    - New `cache` parameter for optional `CacheConfig`.
    - All 8 call methods (`chat`, `chat_async`, `generate`,
      `generate_async`, plus 4 streaming variants) check the cache
      when `temperature == 0`.
    - Module-level `_cache_store_callback` used with
      `functools.partial` to store streaming results after iteration.
  - `CacheConfig` exported from `agents7.llm`.
  - `StreamResult` and `AsyncStreamResult`:
    - New `_on_complete` callback invoked after iteration finishes.
    - New `_as_dict()` method to serialize state for caching.
    - New `from_cached()` classmethod to create pre-filled instances
      from cached data.
    - `__iter__` / `__aiter__` return cached parts when `_done` is True.
- Module `agents`:
  - Class `Labeler`:
    - `_load_yaml_settings` parses optional `cache:` section in
      engine config.
- New file `tests/test_cache.py`:
  - 27 unit tests (all mocked, no LLM) covering `CacheConfig`,
    `ResponseCache`, `make_key`, engine integration, streaming cache,
    `from_cached`, and YAML config parsing.
- `tests/test_streaming.py`: Added `_cache = None` to mock engine
  helper for compatibility.
- `README.md`: Added "Caching" documentation section, updated
  architecture features line.


### v0.6.0 - 1st February 2026

- Module `llm`:
  - New file `streaming.py`:
    - `StreamResult` wrapper that yields `str` content chunks from a
      sync ollama streaming iterator. Exposes `.response`, `.thinking`,
      `.role`, and `.metrics` after iteration.
    - `AsyncStreamResult` async counterpart using `__aiter__`.
  - Class `LLMProvider`:
    - 4 new abstract methods: `chat_stream`, `chat_stream_async`,
      `generate_stream`, `generate_stream_async`.
  - Class `LLMEngine`:
    - 4 new delegation methods that bypass retry logic (streaming
      calls cannot be retried mid-stream).
  - `StreamResult` and `AsyncStreamResult` exported from `agents7.llm`.
- Module `llm.ollama`:
  - Class `OllamaProvider`:
    - 4 new concrete streaming methods using `stream=True`.
- New file `tests/test_streaming.py`:
  - 15 unit tests (all mocked, no LLM) covering `StreamResult`,
    `AsyncStreamResult`, and `LLMEngine` delegation.
- `README.md`: Added "Streaming" documentation section, updated
  architecture line.


### v0.5.0 - 1st February 2026

- Module `llm`:
  - New file `retry.py`:
    - `RetryConfig` frozen dataclass for configuring exponential backoff.
    - `retry_sync` and `retry_async` helpers that wrap LLM calls.
  - Class `LLMEngine`:
    - New `retry` parameter for optional `RetryConfig`.
    - All 4 delegation methods (`chat`, `chat_async`, `generate`,
      `generate_async`) now route through retry helpers.
  - `RetryConfig` exported from `agents7.llm`.
- Module `agents`:
  - Class `Labeler`:
    - `_load_yaml_settings` parses optional `retry:` section in
      engine config.
- New file `tests/test_retry.py`:
  - 15 unit tests covering RetryConfig validation, sync/async retry
    logic, and YAML config parsing.
- `README.md`: Added "Retry / Resilience" documentation section.


### v0.4.8 - 1st February 2026

- New file `tests/test_async.py`:
  - 7 async integration tests for `act_async` and
    `act_in_batch_async` across Detector, Classifier, and Regressor.
- Added `pytest-asyncio` dev dependency and `asyncio_mode = "auto"`
  pytest config in `pyproject.toml`.


### v0.4.7 - 1st February 2026

- Tests: Fixed YAML fixture paths in `test_detector.py`,
  `test_classifier.py`, and `test_regressor.py` to use
  project-root-relative paths instead of `../examples/`.


### v0.4.6 - 1st February 2026

- `.claude/`: Added `README.md` update step to the
  "After Every Change" workflow in `CLAUDE.md`.


### v0.4.5 - 1st February 2026

- `.claude/`: Added Ollama and full test suite steps to the
  "After Every Change" workflow in `CLAUDE.md`.


### v0.4.4 - 1st February 2026

- `.claude/`: Added `uv pip install -e .` + `git add uv.lock` step
  to the "After Every Change" workflow in `CLAUDE.md`.


### v0.4.3 - 1st February 2026

- New file `tests/test_labeler.py`:
  - 27 pure unit tests for the `Labeler` agent (no LLM required).
  - Covers initialization, YAML loading, prompt generation,
    output model list, JSON parsing, and enum mapping.


### v0.4.2 - 1st February 2026

- `.claude/`: Added cleanup lesson for `.tmp` files to `CLAUDE.md`.


### v0.4.1 - 1st February 2026

- New example `examples/labeler_params.py` demonstrating `prompt_params`
  usage with both YAML static values and Python callable params.
- New YAML config `examples/configs/demo_labeler_params.yml`.


### v0.4.0 - 1st February 2026

- Module `exceptions`:
  - New `PromptParameterError` exception class.
- Module `agents`:
  - `BaseAgent`:
    - New `prompt_params` field for `$variable` placeholders in
      `system_prompt`.
    - New `resolved_system_prompt` property that resolves placeholders
      using `string.Template.safe_substitute()`.
    - Supports static strings and callables (evaluated each access).
    - Validation in `__post_init__` raises `PromptParameterError` for
      invalid values.
  - `Labeler`:
    - `prompt` and `prompt_for_batch` now use `resolved_system_prompt`.
    - `_load_yaml_settings()` parses optional `prompt_params` key,
      casting values to `str`.
  - `Detector`, `Classifier`, `Regressor`:
    - `prompt` and `prompt_for_batch` now use `resolved_system_prompt`.
- New test file `tests/test_prompt_params.py` with 19 unit tests.


### v0.3.5 - 1st February 2026

- `CLAUDE.md`: Moved from `.claude/CLAUDE.md` to project root.
  - Updated internal links to reference `.claude/rules/` paths.
  - Fixed lesson learned about file location.


### v0.3.4 - 25th January 2026

- Module `agents`:
  - `Labeler`: Refactored JSON parsing helpers for cleaner code:
    - Made `_fix_malformed_json` a static method.
    - Extracted `_get_enum_mappings`, `_apply_enum_mappings` methods.
    - Moved imports to module level.
- Documentation:
  - Added Model Compatibility section to README.
- `.claude/`:
  - Added code style guidelines (static methods, imports, no nested functions).


### v0.3.3 - 25th January 2026

- Module `llm`:
  - `OllamaProvider`: Added fallback for models that don't support
    structured output format (retries without format parameter).
- Module `agents`:
  - `Labeler`: Added JSON parsing fixes for common LLM output issues:
    - Handles markdown code blocks (```json...```)
    - Handles double-brace malformation ({"{"...)
    - Converts enum names to values when models return names instead
- Tests:
  - Added `test_multi_model.py` with 9 tests for multi-model support.


### v0.3.2 - 25th January 2026

- Examples:
  - Fixed YAML syntax in `demo_temperature_detector.yml` (typo in keys).
  - Improved temperature detector prompt to reason conceptually instead of
    matching keywords, with better indirect indicator handling.
  - Improved `detector.py` with human-readable output format.
- `.claude/`:
  - Added lesson: always read `rules/committing.md` before commits.


### v0.3.1 - 25th January 2026

- Documentation:
  - Updated `README.md` with current architecture (Classifier, Regressor,
    ClassificationProperty, Property[T]), examples using batch processing,
    and custom model definitions.
- Examples:
  - Updated `classifier.py` and `labeler.py` to use `act_in_batch` for
    efficient batch processing instead of looping with `act()`.
  - Improved `detector.py` with human-readable output format.
- `.claude/`:
  - Added lesson learned: prefer batch methods for list inputs.
  - Added lesson learned: include `.claude/` changes in changelog.
  - Added lesson learned: keep README aligned with changes.
  - Added `rm *.tmp*` permission to settings.json.
  - Updated workflow: auto-push committed changes to origin.


### v0.3.0 - 25th January 2026

- Module `models`:
  - Added `ClassificationProperty` base class for classification tasks.
    - Supports `category` and `reason` StrEnum fields.
    - Validates category-reason relationships via `CATEGORY_REASONS` mapping.
    - Auto-adds `NOT_APPLICABLE` to both enums.
  - Added `Sentiment` demo model with category-reason constraints.
- Module `agents`:
  - Added `Classifier` agent for categorical classification tasks.
    - Requires `ClassificationProperty` output model.
    - Includes `possible_categories` and `category_reasons_constraints` properties.
    - Prompts include category-reason mapping for LLM guidance.
  - Added `Regressor` agent for numeric value extraction tasks.
    - Uses existing `Property[T]` output model.
    - Specialized prompts for value extraction.
- Examples:
  - Added `classifier.py` example with sentiment analysis.
  - Added `demo_sentiment_classifier.yml` configuration.
- Tests:
  - Added `test_classifier.py` with 14 tests.
  - Added `test_regressor.py` with 11 tests.


### v0.2.0 - 25th January 2026

- Module `exceptions`:
  - Added `PropertyConfigurationError` for Property/Detector configuration errors.
- Module `models`:
  - Class `Property`:
    - Automatically adds `NOT_APPLICABLE` member to reason enums if not present.
    - Uses `__init_subclass__` to extend reason enum before Pydantic processes it.
- Module `agents`:
  - Class `Detector`:
    - Now uses `PropertyConfigurationError` instead of `TypeError`.
    - Removed explicit `NOT_APPLICABLE` validation (now automatic).
- Tests:
  - Added tests for automatic `NOT_APPLICABLE` addition.
  - Added test for `PropertyConfigurationError`.


### v0.1.0 - 25th January 2026

- Module `models`:
  - Class `Property`:
    - Now a generic class `Property[T]` for type-safe value extraction.
    - Improved documentation with usage examples.
- Module `agents`:
  - Class `Detector`:
    - Now validates that the reason enum has a `NOT_APPLICABLE` member.
    - Raises `ValueError` if the reason enum lacks this fallback.
- Module `models.demo`:
  - Class `Temperature`: Updated to use `Property[float]`.
- Tests:
  - Added test for `NOT_APPLICABLE` enforcement in Detector.
  - Updated COMFORTABLE test case for batch consistency.


### v0.0.19 - 25th January 2026

- Examples:
  - Labeler example improved with entity extraction use case.
  - Added `PersonInfo` model for clear person data extraction demo.
  - Updated `demo_labeler.yml` with extraction rules and examples.
  - Rewritten `labeler.py` with diverse test cases.
- Tests:
  - Replaced ambiguous COMFORTABLE temperature test case with explicit statement.


### v0.0.18 - 25th January 2026

- Module `agents`:
  - Class `Labeler`:
    - Improved `prompt_for_batch` to align with single mode prompt structure.
    - Added clear instruction for independent batch processing.
- Tests:
  - Replaced ambiguous test cases with clearer temperature statements.
  - Batch tests now verify LLM-dependent content successfully.


### v0.0.17 - 25th January 2026

- Module `agents`:
  - Class `Labeler`:
    - Fixed chunking bug in `act_in_batch` and `act_in_batch_async`:
      slice was `(i+1)+n` instead of `(i+1)*n`.
    - Methods `act_in_batch` and `act_in_batch_async` now return one
      `AgentResponse` per input item (flattened), instead of one per chunk.


### v0.0.16 - 25th January 2026

- Documentation:
  - `README.md`: Added comprehensive documentation with architecture overview,
    quick start examples, and development instructions.
  - `.claude/CLAUDE.md`: Added Claude Code guidelines with workflow,
    versioning, commit conventions, and lessons learned sections.
  - `.claude/settings.json`: Added permission settings for git, python,
    and bash commands.


### v0.0.15 - 5th October 2025

- Module `agents`:
  - Class `Labeler`:
    - Added the properties:
      - `prompt_for_batch`.
      - `output_model_list`.
    - Methods `act_in_batch` and `act_in_batch_async`:
      - Now they use the `prompt_for_batch` property.
      - Added the parameter `requests_per_prompt`.
    - Method `act_in_batch_async`:
      - Renamed the parameter `print_tqdm` to `with_tqdm`.
    - Added the methods:
      - `_act_in_batch_async_request`.
      - `_act_in_batch_request`.
  - Class `Detector`:
    - Added the properties:
      - `prompt_for_batch`.


### v0.0.14 - 28th September 2025

- Module `agents`:
  - Class `BaseAgent`: methods `act` response type changed to
    `models.AgentResponse`.
- Module `models`:
  - Submodule `base`: Added the models:
    - `AgentResponse`.
    - `LLMMetrics`.
  - Models also added to `__init__.py`.


### v0.0.13 - 28th September 2025

- Module `agents`:
  - Classes `Labeler` and `Detector`: fixed a bug in the method 
    `act_in_batch`: was returning a list with the output models
    instead of a list of the full responses.
- Added tests for the `Detector` class.


### v0.0.12 - 14th September 2025

- Module `agents`:
  - Class `Detector`: removed the "Reasons" part of the `prompt`.
  - Function `utils.model_class_to_string`:
    - Fix a bad rendering of the default values of the fields when 
      they are not defined.
    - Improved strings for descriptions.
    - Added a "Possible values" section for Enum fields.


### v0.0.11 - 14th September 2025

- Module `agents`:
  - Class `Labeler`:
    - Added the parameter `examples`.
    - Added the property `prompt`.
    - Added the `classmethod` `_load_yaml_settings`.
    - Added the `__posrt_init__` method.
    - With these changes, the prompt will better include 
      instructions, examples, and the output schema description.
  - Class `Detector`:
    - Added the properties `prompt` and `possible_reasons`.
    - Fix in the method `from_yaml`, it wasn't returning a 
      `Detector` instance.
  - Added the scrip `utils` with functions:
    - `get_examples_str`.
    - `import_from_path`.
    - `model_class_to_string`.
- `detector` example improved.


### v0.0.10 - 31st August 2025

- Fix in the function `llm.ollama.parse_ollama_response`,
  was not properly parsing the `thinking` response from different
  models.
- Dependency on `ollama` updated to `0.5.3`.


### v0.0.9 - 24th August 2025

- Module `agents`:
  - Added `detector.Detector` into `__init__`.
  - Class `labeler.Labeler`:
    - Added async methods `act_async` and `act_in_batch_async`.
    - Added the method `_parse_model_response`.


### v0.0.8 - 13th August 2025

- Added the class `agents.detector.Detector`, which is a simple 
  `Labeler` with a defined output model.
- Module `models`
  - Submodule `base`:
    - Just the class `Property` remains, with simplified logic. 
      No need for `PropertyReason` anymore.
  - Submodule `demo`:
    - Added the classes `Temperature` and `TemperaturePropertyReason`.
- Added a folder for `examples`, showcasing the usage of a `labeler` 
  and a `detector` agents.


### v0.0.7 - 3rd August 2025

- Module `models.base`:
  - Class `PropertyReason`: It is not an `Enum` anymore.
  - Class `Property`: logic of the method `__init_subclass__` improved.
- Function `agents.labeler.Labeler.act_in_batch` receives the
  parameter `iter_func`, which defaults to `tqdm.tqdm`.


### v0.0.6 - 3rd August 2025

- Class `models.PropertyReason`: added a field `NOT_APPLICABLE`.
  All properties should have a reason, and this is the default
  when no property-reason is applicable.


### v0.0.5 - 27th July 2025

- Added the module `models.base` with the classes `Property` and
  `PropertyReason`. Added to `models.__init__.py`.


### v0.0.4 - 20th July 2025

- Module `agents`:
  - Class `base.BaseAgent`:
    - Added the parameter `version`.
    - Added the class method `from_yaml`.
    - It is now a `frozen` dataclass.
  - Class `labeler.Labeler`:
    - Implemented the class method `from_yaml`.
    - It is now a `frozen` dataclass.
- Added the module `models` with just `demo.DemoModel`.
- Added a dependency to `pyyaml`.


### v0.0.3 - 13th July 2025

- Added the module `agents`, with:
  - The type `Agent` and the class `Labeler` in `__init__`.
  - Added the submodule `base` with the class `BaseAgent`.
  - Added the submodule `labeler` with the class `Labeler`.
- Added the module `logs`.
- Fix in `llm.ollama.parse_ollama_chat_response`.
- Added a dependency to `pydantic` and `tqdm`.


### v0.0.2 - 6th July 2025

- Module `llm`:
  - Submodule `ollama`:
    - Added methods `chat` and `chat_async` to the `OllamaProvider` class.
    - Better parsing of the response for the `generate` method.
    - Improved the function `parse_ollama_response`. 
    - Added the function `parse_ollama_chat_response`.
  - Module `base`:
    - Added the improvements in `ollama` to the `LLMEngine` and 
      `LLMProvider` classes. Doc updated.
    - Added the attribute `model_id` to both classes.
      It appears in the output of the `get_model_info` method.


### v0.0.1 - 29th June 2025

- Initial release.
- Added the module `llm`:
  - Implemented the interface using `LLMEngine` and `LLMProvider`.
  - Implemented `ollama` as a LLM provider.
