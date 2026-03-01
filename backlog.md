
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

### Infrastructure

- [x] Consider Docker BuildKit secrets for `GITLAB_TOKEN` in `Dockerfile` — currently passed as build ARG, which may persist in layer history.
- [x] Document Linux workaround for `host.docker.internal` in `docker-compose.yml` — works on Docker Desktop (Windows/Mac) but not natively on Linux.
