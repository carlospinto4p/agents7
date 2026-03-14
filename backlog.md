
# Backlog - Agents7

### Pending Manual Testing

- [ ] End-to-end manual test of the full Agent Builder UI (all 5 steps: Define, Configure, Test, Evaluate, Export).
- [ ] Test prompt versioning and A/B comparison UI: save/load/delete versions, run A/B comparison, verify side-by-side results. Confirm Single Input, Batch Test, and History tabs still work.
- [ ] Test JSONL file upload in Step 3 (Batch Test) and Step 4 (Evaluate).
- [ ] Test auto-discovery flow: upload JSONL, write intent, click Discover Schema, verify form fills, then Generate Prompt & Examples in Step 2.

### Refactoring — 22nd February 2026

#### Architecture (high impact)

- [x] Extract response parsing from `Labeler` into a `ResponseParser` class — done in v1.4.2 (`agents7.agents.parsing`).
- [x] Extract YAML loading from `Labeler` into config loader — done in v1.4.2 (`agents7.agents.config_loader`).
- [x] Consolidate `LLMEngine` cache/retry boilerplate — done in v1.4.2 (`_execute`, `_execute_async`, `_stream`, `_stream_async` helpers).

#### Duplication (medium impact)

- [x] Extract shared Ollama model selector widget — done in v1.4.3 (`render_model_selector()` in `ollama_utils.py`).

#### Test Coverage (medium impact)

- [x] Add tests for batch flattening logic — done in v1.4.3 (`tests/test_batch.py`, 8 tests).
- [x] Add parametrized tests for enum conversion edge cases — done in v1.4.3 (`tests/test_enum_parsing.py`, 13 tests).

#### Style & Consistency (low impact)

- [x] Standardize type hint style — done in v1.4.4 (normalized `dict = None` → `dict | None = None`).
- [x] Standardize docstring format — done in v1.4.4 (single-line for properties).
- [x] Add missing type hints in `streaming.py` — done in v1.4.4 (`Any` for `chunk` parameter).
- [x] Move `_default_mapper` in `pipeline.py` — done in v1.4.4 (`PipelineStep.default_mapper` staticmethod).

### Refactoring — 1st March 2026

#### Duplication — Source (high impact)

- [x] Extract OllamaProvider thinking-fallback into a shared helper — done in v1.4.16 (`_with_think_fallback`, `_with_think_fallback_async`, `_chat_call_kwargs`, `_generate_call_kwargs`).
- [x] Deduplicate `retry_sync` / `retry_async` in `retry.py` — done in v1.4.16 (extracted `_retry_delay` helper).
- [x] Add missing `@staticmethod` on `ResponseParser._apply_enum_mappings` in `parsing.py` — already correct, no change needed.

#### Duplication — Tests (high impact)

- [x] Move `_OLLAMA_AVAILABLE` check to `integration_tests/conftest.py` — done in v1.4.17 (`pytest_collection_modifyitems` hook).
- [x] Move shared integration test data (temperature, sentiment) to `integration_tests/data.py` — done in v1.4.17.
- [x] Consolidate integration test agent fixtures to `integration_tests/conftest.py` — done in v1.4.17.

#### Duplication — Unit Tests (medium impact)

- [x] Consolidate `_batch_chat_response()` helper — done in v1.4.17 (`batch_chat_response` in `tests/helpers.py`).
- [x] Consolidate shared test model classes (`DummyReason`, `DummyOutputWithEnum`) — done in v1.4.17 (moved to `tests/helpers.py`).
- [x] Consolidate `_make_response()` variants — done in v1.4.17 (`make_agent_response` in `tests/helpers.py`).

#### Architecture (medium impact)

- [x] Reduce sync/async duplication in `Labeler.act_in_batch` / `act_in_batch_async` — done in v1.4.18 (`_chunk_inputs`, `_batch_chat_params`, `_parse_batch_response`, `_flatten_batch_responses`).
- [x] Reduce sync/async duplication in `Pipeline.act` / `act_async` — done in v1.4.18 (`_pipeline_result`, refactored `_process_step_result`).

#### Complexity (medium impact)

- [x] Simplify enum mapping logic in `ResponseParser._get_enum_mappings` — done in v1.4.18 (`_resolve_strenum` static method).

### Optimization — 8th March 2026

#### Library — HIGH impact

- [x] Cache `output_model_list` in `Labeler.__post_init__` instead of recreating via `create_model()` on every property access — done in v1.4.20.
- [x] Cache `output_model.model_json_schema()` result in `__post_init__` instead of regenerating on every `act()` / `act_async()` call — done in v1.4.20.
- [x] Reuse `AsyncClient` instance in `OllamaProvider` instead of instantiating a new one per async call — done in v1.4.20.

#### Library — MEDIUM impact

- [x] Cache `possible_categories` / `possible_reasons` in `Classifier.__post_init__` and `Detector.__post_init__` — done in v1.4.21.
- [x] ~~Cache `prompt` / `prompt_for_batch` after first build.~~ **Skipped**: prompt building is string concatenation — negligible cost compared to the LLM call. Not worth the complexity.
- [ ] ~~Use `asyncio.gather` or batch calls in `Pipeline.act_async` where steps are independent (`pipeline.py:239`).~~ **Not applicable**: pipeline steps are inherently sequential (each feeds into the next).

#### UI — HIGH impact

- [x] Cache Ollama connection check and model list with `@st.cache_resource` and 30s TTL — done in v1.4.21.
- [x] Use `agent.act_in_batch()` in batch test instead of calling `act()` in a loop — done in v1.4.21.

#### UI — MEDIUM impact

- [x] Cache `generate_model_code()` / `generate_yaml()` output with `@st.cache_data` — done in v1.4.22.
- [x] Cap `test_history` with `deque(maxlen=50)` — done in v1.4.22.
- [ ] ~~Cache `build_agent_from_state()` to avoid rebuilding agent on every render.~~ **Skipped**: agent construction is lightweight (no I/O); caching adds complexity for minimal gain.
- [ ] ~~Use `act_in_batch()` per version in A/B comparison runner.~~ **Skipped**: batching sacrifices per-item error isolation that `_run_single`'s try/except provides.
- [x] Strip full `AgentResponse` from `eval_results` before storing in session state — done in v1.4.22.

#### Tests — MEDIUM impact

- [x] Change integration test fixtures to `scope="module"` — done in v1.4.22.

### Design Improvements — 8th March 2026

- [x] Make `think` configurable per agent — done in v2.0.0 (`think` field on `Labeler`, parsed from YAML, wired through `_sanitize_chat_kwargs`).
- [x] Strengthen NOT_APPLICABLE guidance in prompts — done in v2.0.0 (`_not_applicable_guidance` property on `Detector`, added to all Property-based agent prompts).
- [x] Add NOT_APPLICABLE integration test helper — done in v2.0.0 (`assert_not_applicable()` in `integration_tests/helpers.py`).
- [x] Add model size recommendations to YAML configs — done in v2.0.0 (comments referencing `docs/model-selection.md`).
- [x] Retry with fallback model — done in v2.0.0 (`fallback_model` on `LLMEngine`, parsed from YAML engine config).

### Refactoring — 8th March 2026

#### Source (high impact)

- [x] Consolidate enum helpers — done in v2.1.1: extracted `resolve_strenum()` and `enum_members()` to `utils.py`; `parsing.py` now uses them instead of duplicated `_resolve_strenum`.
- [x] Move function-level imports to module level in `model_selection.py` — done in v2.1.1.

#### Tests (medium impact)

- [x] Extract `ConcurrencyTracker` helper into `tests/helpers.py` — done in v2.1.1; replaced duplicate tracking in `test_labeler.py` and `test_pipeline.py`.
- [x] Consolidate `_gen_chunk()` as `gen_chunk()` in `tests/helpers.py` — done in v2.1.1.
- [x] Move agent fixtures to `tests/conftest.py` — done in v2.1.1 (`temperature_detector`, `sentiment_classifier`, `temperature_regressor`).

#### Tests (low impact)

- [x] Generalize agent factory as `make_agent()` in `helpers.py` — done in v2.1.1; replaced `_make_detector/classifier/regressor` in `test_prompt_params.py`.

### Refactoring — 8th March 2026 (pass 2)

#### Source (medium impact)

- [x] Consolidate `_chat_call_kwargs` / `_generate_call_kwargs` in `OllamaProvider` into a single `_build_call_kwargs` method — done in v2.1.3.
- [x] Extract thinking-error string `"does not support thinking"` to a class constant `_THINKING_NOT_SUPPORTED_MSG` in `OllamaProvider` — done in v2.1.3.

#### Source (low impact)

- [x] Add `@staticmethod` to `Labeler._chunk_inputs` — already present, no change needed.

#### Tests (medium impact)

- [x] Consolidate `_make_engine()` helpers from `test_cache.py` and `test_streaming.py` into `tests/helpers.py` as `make_engine()` — done in v2.1.3.
- [x] Consolidate integration test data (`demo_pipeline_data`, `demo_company_data`, `SIMPLE_TEST_CASES`) into `integration_tests/data.py` — done in v2.1.3.

#### Tests (low impact)

- [x] Add `# --- Section ---` headers to `test_classifier.py`, `test_detector.py`, `test_regressor.py` — done in v2.1.3.

### Improvements — 8th March 2026

#### HIGH impact

- [x] Agent-level streaming: add `act_stream()` and `act_stream_async()` to `Labeler` — done in v2.2.0 (`agents7.agents.streaming`, `AgentStreamResult` / `AgentAsyncStreamResult`).
- [x] Structured exception hierarchy: add `ParseError`, `ProviderError`, `ModelUnavailableError`, `RateLimitError` to `exceptions.py` — done in v2.2.0; wired through `retry.py` and `parsing.py`.
- [x] Agent evaluation metrics in core: extract accuracy, confusion matrix logic from UI into `agents7.metrics` — done in v2.2.0; UI re-exports from core.

#### MEDIUM impact

- [x] Pipeline convenience: `Pipeline.from_agents(agents)` — done in v2.3.0.
- [x] Batch export helpers: `export_jsonl()` / `export_csv()` in `agents7.export` — done in v2.3.0.
- [x] Persistent cache backend: `SqliteCache` with `CacheConfig(backend="sqlite")` — done in v2.3.0.
- [x] Agent introspection: `output_schema`, `enum_values()`, `describe()` on `Labeler` — done in v2.3.0.

#### LOW impact

- [x] YAML config inheritance: support `base_config` key that merges a parent YAML into the current one — done in v2.4.0.
- [x] Conditional pipeline steps: add optional `condition: Callable` to `PipelineStep` — done in v2.4.0.

### Improvements — 14th March 2026

#### HIGH impact

- [ ] OpenAI-compatible LLM provider — new `openai_provider.py` implementing `LLMProvider` ABC for OpenAI, Azure, vLLM, LM Studio, etc. Optional dependency.
- [ ] Retry on parse failure — configurable `max_parse_retries` on `Labeler` that re-prompts the LLM with error feedback when `ResponseParser` fails.
- [ ] Batch evaluation — `run_evaluation()` should use `act_in_batch()` instead of looping `act()`. Add `run_evaluation_async()`.
- [ ] Agent cloning with overrides — `Labeler.with_overrides(**kwargs)` returns a new instance with selected fields replaced. Useful for A/B testing and model comparison.

#### MEDIUM impact

- [ ] Pipeline streaming — `Pipeline.act_stream()` that streams the final step's output.
- [ ] Export to pandas DataFrame — `export_dataframe(responses) -> pd.DataFrame` with pandas as optional dependency.
- [ ] Structured call logging — optional structured logging with trace IDs, agent name, model, duration, tokens, success/failure per call. JSON format for observability tools.
- [ ] Model warm-up helper — `Labeler.warm_up()` sends a trivial prompt to preload the Ollama model.

#### LOW impact

- [ ] Input preprocessor on Labeler — optional `preprocessor: Callable[[str], str]` that transforms input before building the prompt.
- [ ] Evaluation report export — `export_eval_report(results, metrics, path)` writes a formatted Markdown/HTML evaluation report.

### Infrastructure

- [x] Consider Docker BuildKit secrets for `GITLAB_TOKEN` in `Dockerfile` — currently passed as build ARG, which may persist in layer history.
- [x] Document Linux workaround for `host.docker.internal` in `docker-compose.yml` — works on Docker Desktop (Windows/Mac) but not natively on Linux.
