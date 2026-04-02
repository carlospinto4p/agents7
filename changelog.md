
## Changelog - Agents7

### v3.12.2 - 2nd April 2026

- `CLAUDE.md`:
  - Added Shell Commands, Project Configuration, Versioning / Release,
    and Testing sections.


### v3.12.0 - 29th March 2026

- New module `agents7.agents.jitter`:
  - `apply_typos(text, rate, rng)` — introduces random
    human-like typos into text. Typo types: swap adjacent
    chars, double a char, drop a char. Preserves trailing
    punctuation. Only affects words >= 4 chars. Accepts
    optional `Random` instance for reproducibility.
- Module `agents7.agents.chatter`:
  - Added `typo_rate` field (0.0–1.0). When > 0, applies
    random typos to LLM responses in `chat()` and
    `chat_async()` before storing in history. 0.03 means
    ~3% of words get typos. Defaults to 0.0 (off).
- Module `agents7.agents.config_loader`:
  - Added `typo_rate` YAML config option for Chatter.
- Module `agents7.agents.personality`:
  - `typo_rate` also supported in PersonalityChatter
    YAML config.
- Tests:
  - 10 tests for jitter (zero/negative rate, high rate,
    short words, determinism, punctuation, word count,
    Chatter integration).


### v3.11.1 - 29th March 2026

- Module `agents7.agents.personality`:
  - Added `language` field to `Persona` (e.g.
    ``"Spanish"``). When set, the agent is instructed
    to ALWAYS respond in that language regardless of
    the user's language. Appears in both the character
    profile and the instructions section of the
    auto-generated system prompt.


### v3.11.0 - 29th March 2026

- New module `agents7.agents.personality`:
  - `PhysicalTraits` — frozen dataclass for physical
    appearance (height, build, hair, eyes, skin,
    distinguishing features). Renders via `to_text()`.
  - `Persona` — frozen dataclass for structured character
    definition: full_name, birthdate, nationality,
    occupation, physical traits, personality adjectives,
    backstory, interests, speaking style, and freeform
    `extra` dict. Renders via `to_prompt()`. Supports
    `from_dict()` for YAML loading.
  - `PersonalityChatter` — extends `Chatter` with a
    `Persona`. Auto-generates the system prompt from
    the persona data with instructions to stay in
    character. Supports `extra_instructions` for
    additional guidance. YAML loading via `from_yaml()`
    with a `persona:` section.
- Exported `Persona`, `PersonalityChatter`,
  `PhysicalTraits` from `agents7.agents`.
- Tests:
  - 16 tests covering PhysicalTraits, Persona (prompt
    generation, from_dict), PersonalityChatter (creation,
    auto-prompt, extra instructions, chat integration,
    YAML loading).


### v3.10.2 - 29th March 2026

- Module `agents7`:
  - Exported `RetryConfig`, `CacheConfig`, `MemoryConfig`
    from the top-level package for easier imports.


### v3.10.1 - 29th March 2026

- Module `agents7.llm.base`:
  - Added `timeout` parameter to all `LLMEngine`
    chat/generate methods (`chat`, `chat_async`,
    `generate`, `generate_async`). For async methods,
    enforced via `asyncio.wait_for` — raises
    `LLMException` on expiry. Timeout is excluded from
    cache keys and stripped before forwarding to
    providers.
  - Added `cost_per_1m_tokens` constructor parameter
    to `LLMEngine`. When set, `_inject_cost()` computes
    `cost_cents` from total tokens and injects it into
    the response metrics dict.
- Module `agents7.models.base`:
  - Added `cost_cents: float | None` field to
    `LLMMetrics`. Defaults to `None` when cost tracking
    is not configured.


### v3.10.0 - 29th March 2026

- New module `agents7.llm.anthropic`:
  - `AnthropicProvider` — native Claude provider using the
    official `anthropic` SDK. Supports all Claude models
    (Haiku, Sonnet, Opus).
  - Handles system messages correctly (extracted as
    separate `system` parameter per Anthropic API).
  - Extended thinking support via `think=True` (maps to
    Anthropic's `thinking` parameter with `budget_tokens`).
  - Configurable `default_max_tokens` (defaults to 4096).
  - Full streaming support with adapted chunks matching
    the Ollama interface (`_adapt_stream_sync`,
    `_adapt_stream_async`).
  - Fallback model support wired through `LLMEngine`.
  - Optional dependency: `uv pip install anthropic`.
- Module `agents7.llm.base`:
  - `LLMEngine` now accepts `provider="anthropic"` with
    kwargs `api_key`, `base_url`, `default_max_tokens`.
- Tests:
  - 11 tests for AnthropicProvider (param building,
    response parsing, thinking, engine integration,
    stream adapter). Auto-skipped when `anthropic`
    package not installed.


### v3.9.0 - 29th March 2026

- New module `agents7.testing`:
  - `create_mock_engine(responses)` — creates an
    ``LLMEngine`` with a mocked provider that returns
    predefined responses. Supports dict (input→response
    mapping), single string, or default fallback. Works
    with both sync and async agent methods.
  - `assert_accuracy_above(agent, eval_data, fields,
    min_accuracy)` — runs ``run_evaluation()`` and
    asserts accuracy meets the threshold. Returns the
    metrics dict on success.
- New module `agents7.simple`:
  - `simple_classifier(categories, system_prompt,
    engine)` — creates a ``Classifier`` from a list of
    category strings. Auto-generates ``StrEnum`` and
    ``ClassificationProperty`` subclasses. Supports
    custom reasons, NOT_APPLICABLE toggle, think mode,
    and extra kwargs (examples, prompt_params).
- Tests:
  - 8 tests for `agents7.testing` (mock engine default,
    single, dict, fallback, custom name, async; accuracy
    pass/fail).
  - 11 tests for `agents7.simple` (creation, custom
    params, output model, categories, NOT_APPLICABLE
    on/off, reasons, classify with mock, think, kwargs).


### v3.8.4 - 29th March 2026

- Module `agents7.agents.parsing`:
  - Cached `_enum_mappings` at `ResponseParser.__init__`
    time — avoids recomputing enum field metadata on every
    `parse()` call. Especially impactful in batch
    operations.
- Module `agents7.agents.memory`:
  - Added in-memory cache to `JsonFileMemory` — `save()`
    no longer re-reads the entire file on every call.
    Disk read happens once on first access via
    `_get_cache()`.
- Module `agents7.agents.utils`:
  - Replaced `+=` string concatenation with list-append +
    `"\n\n".join()` in `get_examples_str()`.


### v3.8.3 - 29th March 2026

- Module `agents7.llm.openai_compat`:
  - Added type hints to `_OAIChunk.__init__` (`chunk: Any`)
    and `_OAIMessage.__init__` (`content: str`,
    `role: str | None`, `thinking: str | None`).
- Module `agents7.ui`:
  - Added `-> None` return type to `launch()`,
    `main()`, `_render_app()`, `_home_page()`.
  - Added type hints to `_cached_model_code(state: dict)`
    and `_cached_yaml(state: dict)`.
- Tests:
  - Parametrized YAML validation tests in `test_labeler.py`
    (3 cases into `test_yaml_validation_errors`).
  - Parametrized fix-malformed-json tests (4 cases into
    `test_fix_malformed_json`).


### v3.8.2 - 29th March 2026

- Module `agents7.agents.labeler`:
  - Extracted `_build_act_messages()` — shared message
    construction for `act`, `act_async`, `act_stream`,
    `act_stream_async`.
  - Extracted `_finalize_response()` — shared validate,
    wrap, and log logic for `act` and `act_async`.
  - Extracted `_log_retry()`, `_retry_chat_kwargs()`, and
    `_try_parse()` from retry parse loops — reduces
    sync/async duplication in `_retry_parse_loop` and
    `_retry_parse_loop_async`.
  - Added `-> None` return type to `__post_init__`.
- Added `-> None` return type to `__post_init__` in:
  `detector.py`, `classifier.py`, `pipeline.py` (both
  `PipelineStep` and `Pipeline`), `cache.py`, `retry.py`.


### v3.8.1 - 29th March 2026

- Module `agents7.agents.streaming`:
  - Extracted `_AgentStreamBase` base class shared by
    `AgentStreamResult` and `AgentAsyncStreamResult`,
    consolidating duplicated `_finalize()`, `from_failed()`,
    `agent_response` property, and `__init__`.
- Module `agents7.agents.chatter`:
  - Extracted `_ChatterStreamBase` base class shared by
    `ChatterStreamResult` and `ChatterAsyncStreamResult`,
    consolidating duplicated `_finalize()`, `chat_response`
    property, and `__init__`.
  - Added return type hints to `__iter__`, `__aiter__`,
    and `_iterate`.
  - Removed `warm_up` / `warm_up_async` (now inherited
    from `BaseAgent`).
- Module `agents7.agents.base`:
  - Added `warm_up()`, `warm_up_async()`,
    `_log_warm_up_start()`, `_log_warm_up_done()` — shared
    by all agents (previously duplicated in `Labeler` and
    `Chatter`).
- Module `agents7.agents.labeler`:
  - Removed `warm_up`, `warm_up_async`,
    `_log_warm_up_start`, `_log_warm_up_done` (now inherited
    from `BaseAgent`).


### v3.8.0 - 28th March 2026

- Module `agents7.agents.chatter`:
  - Added `max_turns` field — limits the number of complete
    conversation turns. Raises `ChatEndedError` when reached.
  - Added `end_condition` field — callable that receives
    conversation history and returns True to end the chat.
  - Added `turns` property — count of completed turns.
  - Added `ended` property — whether the conversation has
    ended (by max_turns or end_condition).
  - All chat methods (`chat`, `chat_async`, `chat_stream`,
    `chat_stream_async`) check end conditions before
    processing.
  - `reset()` clears turn count, allowing new turns.
- Module `agents7.exceptions`:
  - Added `ChatEndedError` exception.
- Module `agents7.agents.config_loader`:
  - `load_chatter_yaml_settings()` supports `max_turns`
    and `end_condition` YAML fields.
- Tests:
  - 15 unit tests for end conditions.


### v3.7.0 - 28th March 2026

- New module `agents7.agents.memory`:
  - `MemoryConfig` frozen dataclass for memory backend
    configuration (backend, path).
  - `MemoryBackend` ABC with `load()`, `save()`, `clear()`,
    and `list_conversations()` methods.
  - `JsonFileMemory` — JSON file-backed conversation memory.
  - `SqliteMemory` — SQLite-backed conversation memory.
  - `create_memory()` factory function.
- Module `agents7.agents.chatter`:
  - `Chatter` now accepts optional `memory` (MemoryBackend)
    and `conversation_id` (str) fields.
  - History auto-loads from backend on init and auto-saves
    after each turn (chat, async, streaming).
  - `reset()` now also clears the memory backend.
  - `from_yaml()` supports `memory` section in YAML config.
- Module `agents7.agents.config_loader`:
  - `load_chatter_yaml_settings()` parses optional `memory`
    and `conversation_id` YAML fields.
- Tests:
  - 23 unit tests for memory backends.
  - 10 integration tests for Chatter + memory.


### v3.6.0 - 28th March 2026

- New module `agents7.agents.chatter`:
  - `Chatter` conversational agent extending `BaseAgent`
    for multi-turn chat interactions with configurable
    personality via system prompt.
  - Supports sync/async chat: `chat()`, `chat_async()`.
  - Supports sync/async streaming: `chat_stream()`,
    `chat_stream_async()`.
  - Manages conversation history with configurable
    `max_history` (automatic trimming).
  - Configurable `temperature` (default 0.7) and `think`
    mode.
  - `reset()` to clear conversation history.
  - `from_yaml()` for YAML-based configuration.
  - `ChatterStreamResult` and `ChatterAsyncStreamResult`
    streaming wrappers.
- New model `agents7.models.ChatResponse`:
  - Lightweight response model for Chatter (response,
    thinking, metrics — no structured output).
- Module `agents7.agents.config_loader`:
  - Added `load_chatter_yaml_settings()` and
    `validate_chatter_yaml_config()` for Chatter YAML
    loading.
- Examples:
  - Added `demo_chatter.yml` example configuration.
- Tests:
  - 27 unit tests for the Chatter agent.


### v3.5.0 - 21st March 2026

- New module `agents7.suggestions`:
  - `Suggestion` frozen dataclass for proposed corrections
    (agent, input_data, expected, note, timestamp).
  - `SuggestionQueue` file-based JSONL queue with add,
    list, count, remove, and clear operations. Default
    storage: `~/.agents_catalog/suggestions/<agent>.jsonl`.
- Module `agents7.exceptions`:
  - Added `SuggestionError` for invalid suggestion
    operations.
- Tests:
  - 20 unit tests for the suggestions module.


### v3.4.4 - 21st March 2026

- Docs `README.md`:
  - Updated library description to acknowledge that agents
    can be rule-based or hybrid, not only LLM-powered.
- Docs `docs/design.md`:
  - Added paragraph explaining when deterministic agents
    outperform LLM-based ones, with tagger_agent as
    a concrete example.


### v3.4.3 - 14th March 2026

- Module `agents7.llm.base`:
  - Extracted `_cache_check()` helper — shared cache-lookup
    logic used by `_execute`, `_execute_async`, `_stream`,
    and `_stream_async` (was duplicated 4 times).
- Module `agents7.agents.detector`:
  - Refactored prompt building into template methods:
    `_agent_instruction()`, `_batch_instruction()`, and
    `_extra_prompt_sections()`. `prompt` and
    `prompt_for_batch` now delegate to these.
- Module `agents7.agents.classifier`:
  - Overrides only the template methods instead of
    duplicating the full `prompt` / `prompt_for_batch`
    properties. Inherits `prompt` and `prompt_for_batch`
    from Detector unchanged.
- Module `agents7.agents.labeler`:
  - Extracted `_log_warm_up_start()` / `_log_warm_up_done()`
    helpers — `warm_up` and `warm_up_async` now share
    logging logic.


### v3.4.2 - 14th March 2026

- Module `agents7.agents.labeler`:
  - Consolidated `_retry_parse_sync` / `_retry_parse_async`
    into shared `_retry_parse_loop` /
    `_retry_parse_loop_async` that accept a chat callable.
  - New `_strip_role()` static method replaces 5 scattered
    `response.pop("role")` calls.
- Module `agents7.agents.pipeline`:
  - Standardized loop variable from `i` to `step_idx` in
    `act()`, `act_async()`, `act_stream()`, and
    `act_stream_async()`.


### v3.4.1 - 14th March 2026

- Module `agents7.metrics`:
  - `compute_metrics()` now includes a ``timing`` dict with
    ``total_s``, ``avg_s``, ``min_s``, ``max_s``, and
    ``per_case`` (list of per-case durations in seconds).
  - New internal `_compute_timing()` helper.
- Module `agents7.export`:
  - `export_eval_report()` includes a "Response Times"
    summary table (total, average, min, max) and per-case
    duration in seconds in the case headers.


### v3.4.0 - 14th March 2026

- Module `agents7.models.base`:
  - `Property` and `ClassificationProperty` now support
    `NOT_APPLICABLE = False` on subclasses to disable the
    automatic addition of NOT_APPLICABLE to enum fields.
    Default remains `True` (backward compatible).
  - `ClassificationProperty.validate_category_reason_link`
    uses safe `hasattr` checks for NOT_APPLICABLE name
    lookups.
- Module `agents7.agents.detector`:
  - `Detector` prompt conditionally includes NOT_APPLICABLE
    guidance based on `output_model.NOT_APPLICABLE`.
- Module `agents7.agents.classifier`:
  - `Classifier` prompt and constraints conditionally include
    NOT_APPLICABLE references.
- Module `agents7.models.demo`:
  - `Sentiment` model now sets `NOT_APPLICABLE = False`
    since NEUTRAL already covers the "no sentiment" case.
- Integration tests:
  - Classifier tests updated to reflect Sentiment model
    without NOT_APPLICABLE.
- Tests:
  - New `test_not_applicable_opt_out.py` (10 tests): opt-out
    for Property, ClassificationProperty, validation, and
    prompt generation.
  - `test_classifier.py`: updated assertions for Sentiment
    without NOT_APPLICABLE.
- `README.md`: documented NOT_APPLICABLE opt-out.


### v3.3.2 - 14th March 2026

- Module `agents7.export`:
  - New `export_eval_report(results, metrics, path, ...)`:
    generates a Markdown evaluation report with agent info,
    date, overall metrics, per-field accuracy, optional
    confusion matrix, and per-case detail.
  - Internal helper `_append_confusion_matrix()`.
- Integration tests:
  - `conftest.py`: new `reports_dir` fixture and
    `generate_eval_report()` helper that runs evaluation and
    writes a Markdown report to
    `integration_tests/reports/`.
  - `test_detector.py`: new `test_detector_eval_report`.
  - `test_classifier.py`: new `test_classifier_eval_report`
    (includes confusion matrix).
  - `test_regressor.py`: new `test_regressor_eval_report`.
- Tests:
  - New `test_eval_report.py` (8 tests): basic report, agent
    info, per-field accuracy, confusion matrix on/off, error
    handling, empty results, date stamp.
- `.gitignore`: added `integration_tests/reports/`.
- `README.md`: documented `export_eval_report()`.


### v3.3.1 - 14th March 2026

- Module `agents7.agents.labeler`:
  - New `preprocessor` field (optional, default None):
    callable that transforms input data before it reaches
    the LLM. Applied in `act()`, `act_async()`,
    `act_stream()`, `act_stream_async()`, and batch methods
    (per item).
  - New `_preprocess()` internal method.
- Module `agents7.agents.config_loader`:
  - YAML configs support `preprocessor` key (dotted import
    path).
- Tests:
  - New `test_preprocessor.py` (8 tests): default no-op,
    act/act_async/batch application, truncation, dict
    replacement, YAML loading.
- `README.md`: documented input preprocessor.


### v3.3.0 - 14th March 2026

- Module `agents7.agents.labeler`:
  - New `warm_up()`: sends a trivial `generate` call to
    preload the Ollama model into memory, avoiding cold-start
    latency on the first real `act()` call. No-op for
    non-Ollama providers.
  - New `warm_up_async()`: async version.
- Tests:
  - New `test_warm_up.py` (4 tests): sync/async warm-up,
    no-op for openai_compat.
- `README.md`: documented model warm-up.


### v3.2.0 - 14th March 2026

- Module `agents7.call_log` (new):
  - `CallRecord`: frozen dataclass capturing per-call
    metadata (trace_id, agent_name, agent_version, model_id,
    success, tokens, prompt_tokens, total_duration, error).
  - `JsonCallLogger`: writes `CallRecord` as JSON lines to a
    file or stream. Defaults to stdout.
  - `generate_trace_id()`: produces 12-char hex trace IDs.
- Module `agents7.agents.labeler`:
  - New `call_logger` field (optional, default None): callable
    invoked after each `act()` / `act_async()` with a
    `CallRecord`.
  - New `_emit_call_log()` internal method.
- Module `agents7.agents.config_loader`:
  - YAML configs support `call_logger` key (dotted import
    path to a callable).
- Tests:
  - New `test_call_log.py` (11 tests): CallRecord, trace ID
    generation, JsonCallLogger (file, stream, stdout),
    integration with act/act_async.
- `README.md`: documented structured call logging.


### v3.1.0 - 14th March 2026

- Module `agents7.agents.pipeline`:
  - New `Pipeline.act_stream(input_data, chat_kwargs)`:
    executes all steps except the last with `act()`, then
    streams the final step via `act_stream()`. Returns an
    `AgentStreamResult`.
  - New `Pipeline.act_stream_async(input_data, chat_kwargs)`:
    async version using `act_async()` + `act_stream_async()`.
  - On intermediate failure (model=None or mapper error),
    returns a pre-filled `AgentStreamResult` that yields
    nothing.
- Module `agents7.agents.streaming`:
  - New `AgentStreamResult.from_failed(response)`: creates a
    pre-filled failed result (used by pipeline streaming).
  - New `AgentAsyncStreamResult.from_failed(response)`: async
    version.
  - Both stream classes now handle `_stream=None` gracefully
    in `__iter__` / `__aiter__`.
- Module `agents7.export`:
  - New `export_dataframe(responses, fields)`: converts agent
    responses to a Polars `DataFrame`. Requires the `polars`
    package (optional).
- Tests:
  - New `test_pipeline_stream.py` (9 tests): two-step sync,
    single-step, intermediate failure, mapper failure,
    condition skip, async two-step, async failure,
    `from_failed` for both stream types.
  - New `test_export_dataframe.py` (7 tests): basic export,
    explicit fields, None model, empty, mixed, response
    column, no-fields fallback.
- `README.md`: documented pipeline streaming and
  `export_dataframe()`.


### v3.0.0 - 14th March 2026

- **BREAKING**: `pyproject.toml`: replaced `pandas` with
  `polars` in the `[ui]` optional dependency. Users
  installing `agents7[ui]` now get `polars>=1.0.0` instead
  of `pandas>=2.2.0`.
- Module `agents7.ui._pages.4_Evaluate`:
  - Migrated from pandas to polars: `pd.DataFrame` →
    `pl.DataFrame`, `pd.read_csv` → `pl.read_csv`,
    `.to_dict(orient="records")` → `.to_dicts()`.
  - Confusion matrix display uses an explicit `expected`
    column instead of a pandas index.


### v2.6.0 - 14th March 2026

- Module `agents7.metrics`:
  - `run_evaluation()` now uses `act_in_batch()` for
    efficient batch processing instead of calling `act()` in
    a loop. Falls back to sequential `act()` calls if the
    batch fails.
  - New `run_evaluation_async()`: async variant using
    `act_in_batch_async()`.
  - Internal refactor: extracted `_build_eval_result()` and
    `_fallback_sequential()` helpers.
- Module `agents7.agents.labeler`:
  - New `with_overrides(**kwargs)` method: creates a copy of
    the agent with selected fields replaced. Uses
    `dataclasses.replace()`. Works with all agent types
    (Labeler, Detector, Classifier, Regressor).
- Module `agents7.ui.evaluation.runner`:
  - Re-exports `run_evaluation_async` from core.
- Tests:
  - New `test_with_overrides.py` (11 tests): single/multiple
    field override, output model swap, engine swap, type
    preservation, independence.
  - `test_metrics.py`: rewritten to test batch-based
    evaluation, fallback, async, and progress callback.
  - `test_ui_evaluation.py`: updated mocks for batch-based
    `run_evaluation`.
- `README.md`: documented `with_overrides()`, batch
  evaluation, and `run_evaluation_async()`.


### v2.5.0 - 14th March 2026

- Module `agents7.llm`:
  - New `OpenAICompatProvider` (`openai_compat.py`): LLM
    provider for any OpenAI-compatible API (OpenAI, Azure,
    vLLM, LM Studio, etc.). Supports chat, generate,
    streaming (sync + async), and structured output via JSON
    schema. Requires optional `openai` package (`uv pip
    install agents7[openai]`).
  - `LLMEngine`: new `provider="openai_compat"` option with
    `base_url` and `api_key` kwargs. Fallback model support
    included.
  - `retry.py`: retries now also catch OpenAI `APIError`
    when the `openai` package is installed. Transport retry
    logic extended with generic `status_code` check for
    non-Ollama providers.
- Module `agents7.agents`:
  - `Labeler`: new `max_parse_retries` field (default 0).
    When parsing fails, re-prompts the LLM with the error
    message so it can self-correct. Works in `act()` and
    `act_async()`; batch methods do not retry.
  - `config_loader`: YAML configs support
    `max_parse_retries`, `base_url`, and `api_key` fields.
- `pyproject.toml`: new `[openai]` optional dependency
  extra.
- Tests:
  - New `test_openai_compat.py` (15 tests): provider
    creation, response parsing, parameter building, chat
    sync/async, generate, stream adapter, LLMEngine
    integration.
  - New `test_parse_retry.py` (10 tests): no-retry default,
    retry success, retry exhaustion, message accumulation,
    async retry, YAML loading.
- `README.md`: documented OpenAI-compatible provider,
  parse retry, updated requirements.
- `CLAUDE.md`: updated project overview to mention OpenAI
  support.
- `backlog.md`: added Improvements — 14th March 2026
  section with 10 items.


### v2.4.1 - 8th March 2026

- Examples:
  - New `config_inheritance.py`: demonstrates `base_config`
    YAML inheritance with `demo_base.yml` and
    `demo_labeler_inherited.yml`.
  - New `conditional_pipeline.py`: demonstrates conditional
    step execution with `condition` on `PipelineStep`.
  - New `streaming.py`: demonstrates `act_stream()` and
    `act_stream_async()` for token-by-token output.
  - New `sqlite_cache.py`: demonstrates persistent SQLite
    cache via `CacheConfig(backend="sqlite")`.
  - New `export.py`: demonstrates `export_jsonl()` and
    `export_csv()` for batch result export.
- `README.md`: added examples table linking all runnable
  scripts.


### v2.4.0 - 8th March 2026

- Module `agents7.agents.config_loader`:
  - New `base_config` key in YAML: inherits from a parent
    config file. Nested dicts (e.g. `engine`) are merged
    recursively. Chains supported; circular references
    detected.
  - New helpers `_deep_merge()` and `_resolve_inheritance()`.
- Module `agents7.agents.pipeline`:
  - `PipelineStep` gains optional `condition` field: a
    callable receiving the current input dict. When it
    returns `False`, the step is skipped and input passes
    unchanged to the next step.
  - `Pipeline.from_yaml()` supports `condition` key
    (dotted import path).
  - Batch modes (`act_in_batch`, `act_in_batch_async`)
    evaluate conditions per-item via new
    `_partition_by_condition()` helper.
- Tests:
  - New `test_config_inheritance.py` (11 tests): deep merge,
    single/chained inheritance, circular detection, relative
    paths.
  - `test_pipeline.py`: 8 new tests for conditional steps
    (validation, sync, async, batch with per-item filtering).
- `README.md`: documented `base_config` inheritance and
  conditional pipeline steps.


### v2.3.0 - 8th March 2026

- Module `agents7.agents.pipeline`:
  - New `Pipeline.from_agents(agents)` class method: wraps
    each agent in a `PipelineStep` with the default mapper,
    eliminating boilerplate for simple sequential chains.
- Module `agents7.export` (new):
  - `export_jsonl(responses, path)`: write agent responses to
    a JSONL file.
  - `export_csv(responses, path, fields=None)`: write agent
    responses to a CSV file with auto-detected or explicit
    field columns.
- Module `agents7.llm.cache`:
  - `CacheConfig` gains `backend` (``"memory"`` or
    ``"sqlite"``) and `path` fields.
  - New `SqliteCache` backend: persistent SQLite-backed cache
    that survives process restarts. LRU eviction via
    access counter.
  - New `create_cache(config)` factory function.
  - `_CacheKeyBuilder` base class shared between
    `ResponseCache` and `SqliteCache`.
- Module `agents7.agents.labeler`:
  - New `output_schema` property: returns the JSON schema dict
    for the output model.
  - New `enum_values()` method: returns a dict mapping StrEnum
    field names to their ``{name: value}`` member dicts.
  - New `describe()` method: returns a human-readable summary
    of the agent (name, version, model, output, enum fields).
- Tests:
  - New `test_export.py`: 7 tests for JSONL and CSV export.
  - New `test_sqlite_cache.py`: 13 tests for SQLite cache
    backend.
  - `test_pipeline.py`: 2 tests for `Pipeline.from_agents()`.
  - `test_labeler.py`: 5 tests for agent introspection
    methods.


### v2.2.0 - 8th March 2026

- Module `agents7.agents.streaming` (new):
  - `AgentStreamResult`: sync streaming wrapper that yields text
    chunks and exposes `agent_response` after iteration.
  - `AgentAsyncStreamResult`: async version.
- Module `agents7.agents.labeler`:
  - `Labeler.act_stream()`: stream agent responses with real-time
    text chunks, parsing structured output after completion.
  - `Labeler.act_stream_async()`: async version.
  - `act()`, `act_async()`, and batch methods now catch `ParseError`
    internally to preserve backward compatibility (return `model=None`).
- Module `agents7.exceptions`:
  - New `ParseError(LLMException)`: raised when response parsing fails.
  - New `ProviderError(LLMException)`: raised when the LLM provider errors.
  - New `ModelUnavailableError(ProviderError)`: raised on 404 model not found.
  - New `RateLimitError(ProviderError)`: raised on 429 rate limit.
- Module `agents7.llm.retry`:
  - Exhausted retries now raise specific exceptions (`ProviderError`,
    `RateLimitError`, `ModelUnavailableError`) instead of generic
    `LLMException`.
- Module `agents7.agents.parsing`:
  - `ResponseParser.parse()` now raises `ParseError` on failure
    instead of returning `None`.
- Module `agents7.metrics` (new):
  - `EvalResult`: dataclass for evaluation results.
  - `run_evaluation()`: run an agent against labeled data.
  - `compute_metrics()`: aggregate accuracy and per-field metrics.
  - `build_confusion_matrix()`: build confusion matrix for
    categorical fields.
- Module `agents7.ui.evaluation`:
  - `runner.py` and `metrics.py` now re-export from `agents7.metrics`.
- Tests:
  - New `test_agent_streaming.py`: 4 tests for sync/async agent
    streaming.
  - New `test_exceptions.py`: 6 tests for exception hierarchy.
  - New `test_metrics.py`: 12 tests for core evaluation metrics.
  - Updated `test_labeler.py`: `test_invalid_json_returns_none` →
    `test_invalid_json_raises_parse_error`.


### v2.1.4 - 8th March 2026

- Integration tests:
  - Fixed `RuntimeError: Event loop is closed` in async batch tests
    by setting `loop_scope="module"` on async test classes to match
    module-scoped agent fixtures.


### v2.1.3 - 8th March 2026

- Module `agents7.llm.ollama`:
  - Consolidated `_chat_call_kwargs` and `_generate_call_kwargs` into
    a single `_build_call_kwargs` method parameterized by primary key.
  - Extracted thinking-error string to class constant
    `_THINKING_NOT_SUPPORTED_MSG`.
- Tests:
  - New `make_engine()` helper in `tests/helpers.py`; replaced local
    `_make_engine()` in `test_cache.py` and `test_streaming.py`.
  - Consolidated integration test data (`PIPELINE_DATA`,
    `COMPANY_DATA`, `SIMPLE_TEMPERATURE_CASES`) into
    `integration_tests/data.py`.
  - Added `# --- Section ---` headers to `test_classifier.py`,
    `test_detector.py`, `test_regressor.py`.


### v2.1.2 - 8th March 2026

- Self-refinement pass:
  - `CLAUDE.md`: Fixed `pip` → `uv pip` in UI install command; added
    `model_selection` to Code Patterns; removed stale "File Locations"
    lesson.
  - `.claude/rules/testing.md`: Updated integration test skip mechanism
    to reflect `pytest_collection_modifyitems` in `conftest.py`.
  - `.claude/rules/optimization.md`: Replaced "DB batching" example with
    "LLM call batching".
  - `.claude/rules/backlog.md`: Added section header naming convention.
  - `.claude/rules/ui.md`: Trimmed to essential non-obvious rules.
  - `.claude/rules/improvements.md`: Clarified "unchecked items" trigger.
  - `.claude/settings.json`: Pre-commit test hook now skips when no `.py`
    files are staged.
- Cleanup: deleted 2 leftover `.tmp` files.


### v2.1.1 - 8th March 2026

- Module `agents7.agents.utils`:
  - New `resolve_strenum()`: unwraps Optional/Union to find StrEnum type.
  - New `enum_members()`: returns `{name: value}` for a StrEnum.
  - `extract_enum_members()` now uses `resolve_strenum` internally.
- Module `agents7.agents.parsing`:
  - Removed duplicated `_resolve_strenum` static method; now imports
    `resolve_strenum` and `enum_members` from `utils`.
- Module `agents7.model_selection`:
  - Moved function-level imports (`agents`, `types`, `ollama`, `LLMEngine`,
    `Property`, `ClassificationProperty`) to module level.
- Tests:
  - New `ConcurrencyTracker` helper in `tests/helpers.py`; replaced
    duplicate peak/current tracking in `test_labeler.py` and
    `test_pipeline.py`.
  - New `gen_chunk()` helper in `tests/helpers.py`; replaced local
    `_gen_chunk()` in `test_streaming.py`.
  - New `make_agent()` factory in `tests/helpers.py`; replaced
    `_make_detector/classifier/regressor` in `test_prompt_params.py`.
  - Moved `temperature_detector`, `sentiment_classifier`, and
    `temperature_regressor` fixtures to `tests/conftest.py`.


### v2.1.0 - 8th March 2026

- New module `agents7.model_selection`:
  - `find_best_model()`: Automatically finds the smallest Ollama model
    that passes all test cases for a given agent configuration.
  - `TestCase`: Frozen dataclass for defining input/expected pairs.
  - `ModelScore`: Per-model benchmark results (pass rate, duration, errors).
  - `BenchmarkResult`: Full benchmark result with best model and all scores.
  - `CaseResult`: Detailed result for each test case and run.
  - `ModelSelectionError`: Custom exception for model selection failures.
  - Supports multiple runs, batch mode testing, and early stopping.
  - Auto-discovers locally available Ollama models sorted by size.
  - Compares enum fields by name and regular fields by value.
- Tests:
  - New `tests/test_model_selection.py` with 35 unit tests covering
    comparison logic, candidate resolution, and benchmark orchestration.
- README: Added Model Selection section with usage example.
- Module `.claude`:
  - New `/improvements` command for suggesting new capabilities and features.
  - New `.claude/rules/improvements.md` with trigger conditions (low backlog,
    long sessions, noticed gaps) and guidance on what to look for.
  - Updated `.claude/rules/refactoring.md` to include `/improvements` in the
    periodic pass alongside `/refactor` and `/self-refinement`.


### v2.0.0 - 8th March 2026

- Module `agents7.agents.labeler`:
  - New `think` field (default `False`) to control thinking mode
    per agent. Can be overridden per call via `chat_kwargs`.
  - `_sanitize_chat_kwargs` now uses agent's `think` default
    instead of hardcoding `True`.
- Module `agents7.agents.detector`:
  - New `_not_applicable_guidance` property injected into prompts
    for Detector, Classifier, and Regressor. Instructs the model
    to not infer or speculate when classifying NOT_APPLICABLE.
- Module `agents7.agents.classifier`:
  - Added NOT_APPLICABLE guidance to prompt extra sections.
- Module `agents7.agents.regressor`:
  - Added NOT_APPLICABLE guidance to prompt extra sections.
- Module `agents7.agents.config_loader`:
  - Parse `think` from YAML agent config (optional, default true).
  - Parse `fallback_model` from YAML engine config.
- Module `agents7.llm.base`:
  - `LLMEngine` now accepts `fallback_model` parameter. When the
    primary model fails after retries, the engine retries the call
    with the fallback model. If both fail, the primary error is
    raised.
- New file `integration_tests/helpers.py`:
  - `assert_not_applicable()` utility for integration tests.
- Example YAML configs: Added model size recommendation comments
  referencing `docs/model-selection.md`.
- `README.md`: Added documentation for fallback model, thinking
  mode configuration, and updated YAML example.
- `docs/design.md`: Added sections on thinking mode, NOT_APPLICABLE
  prompt guidance, and fallback model. Updated structured output
  flow and design principles table.
- `docs/model-selection.md`: Updated thinking mode section to
  reflect `think=False` default. Added "How agents7 addresses
  these issues" section.
- `backlog.md`: Added design improvement items.


### v1.4.24 - 8th March 2026

- Moved `DESIGN.md` to `docs/design.md`.
- New file `docs/model-selection.md`: Explains why smaller LLMs
  outperform larger ones for structured extraction tasks.
- `README.md`: Updated design doc link and added link to
  model selection guide.


### v1.4.23 - 8th March 2026

- New file `DESIGN.md`: Library design document explaining the
  agent hierarchy, why reasons are required, NOT_APPLICABLE
  auto-injection, structured output flow, YAML configuration,
  batch processing, pipelines, and design principles.
- `README.md`: Added link to `DESIGN.md`.


### v1.4.22 - 8th March 2026

- Module `agents7.ui._pages.1_Define`:
  - Cache `generate_model_code()` with `@st.cache_data`.
- Module `agents7.ui._pages.2_Configure`:
  - Cache `generate_yaml()` with `@st.cache_data`.
- Module `agents7.ui._pages.4_Evaluate`:
  - Strip full `AgentResponse` from eval results before storing
    in session state to reduce memory usage.
- Module `agents7.ui.app`:
  - Cap `test_history` at 50 items using `collections.deque`.
- Integration tests:
  - Change agent fixtures to `scope="module"` to reuse
    instances across tests in the same file.


### v1.4.21 - 8th March 2026

- Module `agents7.agents.detector`:
  - Cache `possible_reasons` in `__post_init__` instead of
    calling `extract_enum_members()` on every property access.
- Module `agents7.agents.classifier`:
  - Cache `possible_categories` and `possible_reasons` in
    `__post_init__` instead of calling `extract_enum_members()`
    on every property access.
- Module `agents7.ui.ollama_utils`:
  - Cache `check_connection()` and `list_available_models()`
    with `@st.cache_resource` and 30-second TTL.
  - Move `streamlit` import to module level.
- Module `agents7.ui._pages.3_Test`:
  - Replace per-item `act()` loop with `act_in_batch()` in
    batch test tab.


### v1.4.20 - 8th March 2026

- Module `agents7.agents.labeler`:
  - Cache `output_model_list` (`create_model`) and both
    `model_json_schema()` results in `__post_init__` instead
    of regenerating on every property access and `act()` call.
- Module `agents7.llm.ollama`:
  - Reuse a single `AsyncClient` instance per `OllamaProvider`
    instead of creating a new one on every async call.


### v1.4.19 - 8th March 2026

- Module `.claude`:
  - `rules/shell.md`: Simplified shell rules, added `uv run`
    requirement and git commit formatting guidance.
  - `rules/optimization.md`: New periodic optimization guidelines
    tailored to the project (Python/Streamlit, no React/DB).
  - `commands/optimize.md`: New `/optimize` slash command for
    codebase-wide performance review.
- `CLAUDE.md`: Added optimization rules link.


### v1.4.18 - 1st March 2026

- Module `agents7.agents.labeler`:
  - Extracted `_chunk_inputs`, `_batch_chat_params`,
    `_parse_batch_response`, and `_flatten_batch_responses`
    helpers to reduce sync/async duplication in batch methods.
- Module `agents7.agents.pipeline`:
  - Added `_pipeline_result` static method and refactored
    `_process_step_result` to return `PipelineResponse | dict | None`,
    reducing duplication in `act`, `act_async`, and
    `_assemble_batch_results`.
- Module `agents7.agents.parsing`:
  - Extracted `_resolve_strenum` static method to simplify
    `_get_enum_mappings` enum type unwrapping logic.
- `backlog.md`: Marked items 11-13 as done.


### v1.4.17 - 1st March 2026

- Integration tests:
  - Moved `_OLLAMA_AVAILABLE` skip logic to `conftest.py` via
    `pytest_collection_modifyitems` hook, removing ~14 duplicated
    lines from each of 7 test files.
  - New `integration_tests/data.py` with shared `TEMPERATURE_DATA`
    and `SENTIMENT_DATA` constants. Removed duplicated data from
    `test_detector`, `test_regressor`, `test_classifier`,
    and `test_async`.
  - Consolidated agent fixtures (`temperature_detector`,
    `sentiment_classifier`, `temperature_regressor`) to
    `conftest.py`. Removed duplicated fixtures from 4 test files.
- Unit tests:
  - Moved `_batch_chat_response` to `tests/helpers.py` as
    `batch_chat_response` with optional `thinking` parameter.
  - Moved `DummyReason` and `DummyOutputWithEnum` to
    `tests/helpers.py`, removed from `test_labeler.py` and
    `test_prompt_params.py`.
  - Moved `_make_response` to `tests/helpers.py` as
    `make_agent_response`, removed from `test_pipeline.py`.
- `pyproject.toml`: Added `integration_tests` to `pythonpath` for
  shared data module imports.
- `backlog.md`: Marked items 5-10 from 1st March refactoring as done.


### v1.4.16 - 1st March 2026

- Module `agents7.llm.ollama`:
  - Extracted `_with_think_fallback` and `_with_think_fallback_async`
    helpers to eliminate duplicated think-retry logic across 8 methods.
  - Added `_generate_call_kwargs` builder and extended
    `_chat_call_kwargs` with `stream` parameter.
  - All 8 public methods now delegate to shared helpers.
- Module `agents7.llm.retry`:
  - Extracted `_retry_delay` helper that computes backoff delay, logs
    the warning, and raises `LLMException` when exhausted.
  - `retry_sync` and `retry_async` now use `_retry_delay` instead of
    duplicated inline logic.
- `backlog.md`: Marked items 1-3 from 1st March refactoring as done.


### v1.4.15 - 1st March 2026

- Module `.claude`:
  - `CLAUDE.md`: Integration tests now only required when relevant
    (not for docs-only changes). Added prompt_params, validators,
    and agent hierarchy to Code Patterns.
  - `rules/testing.md`: Updated integration test flow list (added
    pipeline, prompt_params). Dropped fixture parentheses rule.
    Updated skipif rule to require module-level `pytestmark`.
    Documented config-as-fixture convention.
  - `rules/shell.md`: Removed misleading version in commit example.
  - `rules/versioning.md`: Added `uv pip install` + `git add uv.lock`
    step to version workflow.
  - `rules/backlog.md`: Documented both category and date-based
    section header conventions.
  - `rules/ui.md`: Replaced generic `<package>` with `agents7`.
- Integration tests:
  - All 7 test modules now include `pytestmark` with
    `pytest.mark.skipif` for Ollama availability.


### v1.4.14 - 1st March 2026

- `README.md`:
  - Added "Integration Test Models" section documenting which Ollama
    models each integration test file uses, with sizes and pull commands.
  - Fixed integration test run command (removed unnecessary `cd`).


### v1.4.13 - 1st March 2026

- Configs:
  - New `demo_pipeline.yml`: 2-step pipeline config chaining
    PersonExtractor and BioGenerator.
- Integration tests:
  - New `test_pipeline.py`: 5 tests covering Pipeline YAML loading,
    single execution, intermediate step inspection, and batch processing.
  - New `test_prompt_params.py`: 5 tests covering prompt parameter
    resolution, company-scoped extraction, rejection of non-matching
    inputs, and batch processing with prompt_params.


### v1.4.12 - 1st March 2026

- Configs:
  - `demo_temperature_detector.yml`: switched from `qwen3:4b` to
    `qwen2.5:3b` (1.9 GB, no thinking overhead). Updated COMFORTABLE
    range from `~20 °C` to `11-29 °C`. Added 25 °C COMFORTABLE example.
- Integration tests:
  - `test_multi_model.py`: switched models to `qwen2.5:3b`/`gemma3:4b`.
    Added explicit temperature range prompt and 5 examples to the
    factory fixture for reliability with small models.
  - `test_regressor.py`: updated `test_regressor_extracts_numeric_value`
    input from clinical phrasing to natural language — `qwen2.5:3b`
    needs contextual cues to classify borderline temperatures correctly.


### v1.4.11 - 1st March 2026

- Integration tests:
  - `test_multi_model.py`: switched models from `qwen3`/`gpt-oss:20b`
    to lightweight `qwen3:4b`/`gemma3:4b` for faster execution.
- Configs:
  - `demo_temperature_detector.yml`: switched from `qwen3` to
    `qwen3:4b` — fixes batch timeout (19 min → ~1 min).
  - `demo_labeler.yml`, `demo_bio_generator.yml`,
    `demo_labeler_params.yml`: switched from `qwen3` to `qwen3:4b`
    for consistency.
- Module `.claude`:
  - `rules/testing.md`: added 5-minute time budget for the full
    integration test suite.


### v1.4.10 - 1st March 2026

- Config `demo_sentiment_classifier.yml`:
  - Switched model from `qwen3` to `gemma3:12b` — qwen3
    consistently returns invalid `DESCRIPTIVE` reason for
    NEGATIVE category, ignoring category-reason constraints.
  - Added system examples for all 9 `SentimentReason` values
    (was 4/9). Ensures full enum coverage in the prompt.


### v1.4.9 - 1st March 2026

- Module `agents`:
  - `ollama.py`: added automatic thinking fallback — when a model
    does not support `think`, the provider catches the error and
    retries without it. Applies to all chat, generate, and
    streaming methods. Added `_think_kwargs` helper to only send
    `think` when `True`. Added `_is_thinking_not_supported`
    helper.
- Module `.claude`:
  - `rules/testing.md`: added Output Coverage section — test data
    must cover every possible output category defined by the model.


### v1.4.8 - 1st March 2026

- Module `.claude`:
  - New file `rules/shell.md`: added shell command guidelines
    to enforce simple, single-command Bash calls (no compound
    commands, no `cd` prefix, no heredoc commit messages).
  - `CLAUDE.md`: linked the new shell rules file.


### v1.4.7 - 1st March 2026

- Module `.claude`:
  - `settings.json`: added hooks for automated code quality and
    testing:
    - `PostToolUse` hook: runs `ruff check --fix` and
      `ruff format` on `.py` files after every Write/Edit.
    - `PreToolUse` hook: runs unit tests before every
      `git commit` command.


### v1.4.6 - 23rd February 2026

- Module `.claude`:
  - `rules/versioning.md`: expanded changelog template to cover
    `Tests:` and `Module .claude:` entry patterns.
  - `rules/refactoring.md`: clarified that the 6-7 version counter
    resets after each refactoring pass.


### v1.4.5 - 23rd February 2026

- Module `.claude`:
  - `CLAUDE.md`: fixed stale function name (`_load_yaml_settings` →
    `load_yaml_settings`), linked all 6 rule files, added
    `ResponseParser`/`Pipeline`/`config_loader` to Code Patterns,
    added single-line property docstring convention, merged
    "ollama serve" into integration test step.
  - `rules/testing.md`: clarified test class rule (unit: no classes,
    integration: classes allowed), removed duplicated Running Tests
    section.


### v1.4.4 - 23rd February 2026

- Module `agents`:
  - `labeler.py`: standardized type hints (`dict = None` →
    `dict | None = None`) on field and 6 method parameters;
    single-line property docstrings for `prompt`, `prompt_for_batch`,
    `output_model_list`.
  - `pipeline.py`: standardized type hints on 4 method parameters;
    moved `_default_mapper` into `PipelineStep` as `@staticmethod
    default_mapper`.
  - `base.py`: single-line property docstring for
    `resolved_system_prompt`.
  - `detector.py`: single-line property docstrings for
    `possible_reasons`, `prompt`, `prompt_for_batch`.
  - `classifier.py`: single-line property docstrings for
    `possible_categories`, `possible_reasons`,
    `category_reasons_constraints`.
- Module `llm`:
  - `streaming.py`: added `Any` type hint to `chunk` parameter in
    `_extract`, `_read_final`, `_process_chunk`.
- Tests:
  - `test_pipeline.py`: updated imports and references to use
    `PipelineStep.default_mapper`.


### v1.4.3 - 22nd February 2026

- Module `ui`:
  - `ollama_utils.py`: new `render_model_selector()` function
    extracting the shared Ollama model selector widget.
  - `_pages/1_Define.py`: replaced duplicated model selector
    with `render_model_selector()`.
  - `_pages/2_Configure.py`: replaced duplicated model selector
    with `render_model_selector()`.
- Module `agents`:
  - `parsing.py`: fixed `_get_enum_mappings()` to handle
    `X | None` union type syntax (`types.UnionType`).
- Tests:
  - New file `test_batch.py`: 8 tests for batch flattening
    logic (sync/async chunking, idx, shared metadata, empty
    input, order preservation).
  - New file `test_enum_parsing.py`: 13 parametrized tests for
    enum conversion edge cases (multiple enums, optional fields,
    unknown names, nested lists, empty data).


### v1.4.2 - 22nd February 2026

- Module `agents`:
  - New file `parsing.py`: `ResponseParser` class extracted from `Labeler`,
    encapsulating response parsing, JSON fixing, enum mapping, and validators.
  - New file `config_loader.py`: `load_yaml_settings()`, `validate_yaml_config()`,
    `build_engine()`, and `parse_yaml_examples()` extracted from `Labeler` as
    plain functions. Used by all agent `from_yaml()` methods.
  - `labeler.py`: delegates to `ResponseParser` and `config_loader`; reduced
    from ~760 to ~470 lines.
  - `detector.py`, `classifier.py`, `regressor.py`: `from_yaml()` now calls
    `load_yaml_settings()` directly.
- Module `llm`:
  - `base.py`: consolidated 8 public `LLMEngine` methods via 4 private helpers
    (`_execute`, `_execute_async`, `_stream`, `_stream_async`) that encapsulate
    the cache-lookup/retry/cache-store boilerplate.


### v1.4.1 - 22nd February 2026

- Module `ui`:
  - `discovery/analyzer.py`: `discover_schema()` and `generate_prompt()` now include the `_messages` list in their return dicts.
  - `_pages/1_Define.py`: added "Last Discovery Prompt Sent" expander showing the messages sent to the LLM during schema discovery.
  - `_pages/2_Configure.py`: added "Last Generation Prompt Sent" expander showing the messages sent to the LLM during prompt generation.
- `tests/test_ui_discovery.py`: added tests for `_messages` key in analyzer results.


### v1.4.0 - 22nd February 2026

- Module `ui`:
  - New subpackage `discovery/`:
    - New file `models.py`: Pydantic suggestion models (`ClassifierSuggestion`, `PropertySuggestion`, `LabelerSuggestion`, `PromptSuggestion`, etc.) for structured LLM responses.
    - New file `prompts.py`: system prompt templates for schema discovery and prompt generation, plus `describe_schema()` helper.
    - New file `analyzer.py`: `discover_schema()` and `generate_prompt()` functions that call the LLM with structured output to auto-suggest schemas, system prompts, and examples.
  - `state.py`: added `intent` and `sample_data` fields to `AgentBuilderState`.
  - `_pages/__init__.py`: updated `_state_from_dict()` to read new discovery fields.
  - `_pages/1_Define.py`: added collapsible "Discover from Data" section with model selector, intent input, JSONL upload, preview, and "Discover Schema" button.
  - `_pages/2_Configure.py`: added "Generate Prompt & Examples" button that auto-generates system prompt and few-shot examples from intent + schema + sample data.
- `docs/agent_builder.md`: documented discovery feature in Step 1 and Step 2 sections.
- `README.md`: updated Agent Builder UI table to mention auto-discovery.
- New file `tests/test_ui_discovery.py`: unit tests for discovery models, prompts, and analyzer.


### v1.3.11 - 22nd February 2026

- New file `docs/agent_builder.md`: full workflow documentation for the Agent Builder UI (overview, step-by-step guide, agent types, data flow, session state, file formats).
- `README.md`: added link to the new documentation from the Agent Builder UI section.


### v1.3.10 - 22nd February 2026

- Module `ui`:
  - `app.py`: fixed double-click issue on sidebar navigation radio buttons — added explicit `key` parameter.


### v1.3.9 - 22nd February 2026

- Module `ui`:
  - `3_Test.py`: added a preview of the first 5 records after uploading a JSONL file in the Batch Test tab, along with a total record count.


### v1.3.8 - 22nd February 2026

- Module `ui`:
  - `1_Define.py`: fixed double-click bug on the agent type and value type selectboxes — added explicit `key` parameters so Streamlit's session state takes precedence over the `index` parameter on reruns.


### v1.3.7 - 22nd February 2026

- Module `ui`:
  - `1_Define.py`: added info box describing each agent type (classifier, detector, regressor, labeler) below the type selector.
  - `1_Define.py`: fixed "Number of categorys" typo — `_enum_editor` now uses the `key` parameter (already plural) instead of naive `label + "s"` pluralization.


### v1.3.6 - 22nd February 2026

- Infrastructure:
  - `Dockerfile`: replaced build `ARG` with BuildKit `--mount=type=secret` for `GITLAB_TOKEN` — the token is no longer persisted in image layer history.
  - `docker-compose.yml`: added `secrets` section to pass `GITLAB_TOKEN` via BuildKit secrets during build; added comment explaining `host.docker.internal` and the `extra_hosts` workaround for native Linux Docker (20.10+).


### v1.3.5 - 22nd February 2026

- Module `ui`:
  - New `safe_index()` helper in `_pages/__init__.py` — prevents `ValueError` from stale session state in selectbox widgets.
  - `1_Define.py`: replaced 3 unsafe `.index()` calls with `safe_index()`.
  - `ollama_utils.py`: narrowed bare `except Exception` to `(ImportError, httpx.HTTPError, ollama.ResponseError)` for better debugging.
- `.claude/`: updated backlog marking 3 items as completed (item 5 was already done in v1.3.2).


### v1.3.4 - 22nd February 2026

- Tests:
  - New `tests/conftest.py` with `_chdir_to_project_root` session fixture and shared UI helpers (`mock_metrics`, `make_agent_mock`, `MockModel`, `make_response`).
  - Deduplicated `_make_response()` helpers across `test_ui_comparison.py` and `test_ui_evaluation.py` into shared `make_response()`.
  - Standardized all test comment headers from `# === Section ===` to `# --- Section ---` across `test_cache.py`, `test_labeler.py`, `test_streaming.py`, `test_retry.py`.
- Module `llm`:
  - `LLMEngine`: extracted `_set_stream_cache_callback()` helper, replacing 4 duplicated blocks in streaming methods.
- Module `agents`:
  - `Labeler`: merged `_apply_enum_mappings` and `_apply_enum_mappings_to_item` into a single recursive method.
- `.claude/`: updated backlog marking 7 items as completed.


### v1.3.3 - 22nd February 2026

- Code deduplication — Agents:
  - `labeler.py`: New `_build_prompt()` helper method that
    centralises prompt assembly (system prompt + schema +
    instruction + extra sections + examples). Used by all
    4 agent types, replacing 8 near-identical prompt/
    prompt_for_batch property implementations across
    `labeler.py`, `detector.py`, `classifier.py`, and
    `regressor.py`.
  - `pipeline.py`: Extracted `_process_step_result()`,
    `_evaluate_batch_step()`, and `_assemble_batch_results()`
    helpers, deduplicating ~100 lines of shared logic between
    sync/async act and batch methods.
  - `labeler.py`: Consolidated duplicated `chat_kwargs`
    docstrings (6 occurrences) — full docs on `act()`, all
    other methods reference it.
  - `ollama.py`: Extracted `_needs_format_fallback()` and
    `_chat_call_kwargs()` helpers to deduplicate the
    fallback-retry pattern in `chat()`/`chat_async()`.
- Code deduplication — LLM:
  - `llm/base.py`: Consolidated docstrings across `LLMEngine`
    and `LLMProvider` — full docs on `LLMProvider` abstract
    methods, cross-references elsewhere (12 → 4 unique docs).
  - `llm/ollama.py`: Same cross-reference approach for all
    `OllamaProvider` methods.
- Code deduplication — UI:
  - `_pages/__init__.py`: New `resize_list()` helper that
    replaces the duplicated `while len < num / items[:num]`
    list-expansion pattern in `1_Define.py`, `2_Configure.py`,
    and `4_Evaluate.py` (4 occurrences).
  - `codegen/model_generator.py`: Removed local `type_map`
    dict in `_class_for_property()` — now reuses module-level
    `_FIELD_TYPE_MAP`.
- Cleanup:
  - Removed unused imports (`get_examples_str`,
    `model_class_to_string`) from `detector.py`,
    `classifier.py`, and `regressor.py`.
  - Fixed minor inconsistency: detector `prompt_for_batch`
    used "Output Models Schema" (plural) while all others
    used "Output Model Schema" (singular) — now consistent.
  - Updated backlog with completed items.


### v1.3.2 - 22nd February 2026

- Documentation & correctness:
  - Fixed README.md regressor example (wrong variable name and
    config file path).
  - Added `BaseAgent` to `agents/__init__.py` `__all__` exports.
  - Removed redundant manual `NOT_APPLICABLE` from demo
    `TemperaturePropertyReason` (auto-added at runtime).
  - Created `examples/regressor.py` example script.
- Type hints:
  - Added `-> None` to `BaseAgent.__post_init__`, `Any` param
    types to abstract `act()`.
  - Added return type hints to all UI page `render()` functions
    and helper functions across `1_Define.py`, `2_Configure.py`,
    `3_Test.py`, `4_Evaluate.py`, `5_Export.py`.
  - Fixed `callable` -> `Callable`, bare `list` -> `list[dict]`,
    `list` -> `list[EnumMember]`, removed extra parens in
    `comparison/runner.py` union type.
  - Added parameter type hints to `_compare_field()` in
    `evaluation/runner.py`.
- Tests:
  - Updated `test_detector.py` and `test_regressor.py` to account
    for auto-added `NOT_APPLICABLE` member.
- Cleanup:
  - Deleted leftover `.tmp` files from previous sessions.
  - Updated backlog with refactoring sweep findings.


### v1.3.1 - 22nd February 2026

- Module `llm`:
  - `ollama.py`: Fixed infinite recursion in `_build_options()` — the
    method was calling itself instead of building the options dict
    (introduced in v1.2.8 refactor).


### v1.3.0 - 22nd February 2026

- UI `_pages/__init__.py`:
  - New `parse_jsonl()` shared helper for JSONL file parsing.
- UI `_pages/4_Evaluate.py`:
  - File uploader now accepts both CSV and JSONL files.
  - JSONL files are parsed line-by-line into a DataFrame.
- UI `_pages/3_Test.py`:
  - Batch Test tab now includes a JSONL file uploader as an
    alternative to the text area.
- Tests:
  - New `tests/test_ui_jsonl.py` with 12 tests for JSONL parsing
    (valid input, empty/blank, malformed lines, non-object JSON,
    edge cases).
- `README.md` updated to reflect JSONL upload support.


### v1.2.8 - 15th February 2026

- Code deduplication:
  - `labeler.py`: Extracted `_sanitize_chat_kwargs()` static method
    (4 occurrences).
  - `ollama.py`: Extracted `_build_options()` static method
    (8 occurrences).
  - `models/base.py`: Extracted `_extend_enum_with_not_applicable()`
    shared function.
  - UI `_pages/__init__.py`: Added `get_expected_fields()` and
    `make_progress_callback()` (deduplicated from multiple pages).
  - `codegen/model_generator.py`: Extracted `_enum_code()` helper
    (3 occurrences).
- Tests:
  - Created `tests/helpers.py` with shared test helpers
    (ENGINE, DummyOutput, DUMMY_METRICS, make_labeler, chat_chunk,
    async_iter, mock_metrics, make_agent_mock).
  - Converted all 50 test classes to plain functions across 10 files.
  - Added `pythonpath = ["tests"]` to pytest config.
- UI convention fixes:
  - Moved all function-level imports to module level in
    `_pages/*.py` files.
  - Broke down long functions in `_pages/3_Test.py` into smaller
    helpers: `_save_version()`, `_render_version_list()`,
    `_render_prompt_previews()`, `_show_comparison_summary()`,
    `_show_version_output()`.
- Style: Fixed 37 line-length violations (>78 chars) in docstrings
  across `labeler.py`, `classifier.py`, `ollama.py`, `base.py`,
  and test files.


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
- `.claude/rules/`: Added `backlog.md` and `refactoring.md`
  rules files.


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
