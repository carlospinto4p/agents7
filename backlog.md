
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

- [ ] Extract OllamaProvider thinking-fallback into a shared helper. The `try/except ResponseError → retry without think` block is copy-pasted across 8 methods in `ollama.py` (~80 duplicated lines).
- [ ] Deduplicate `retry_sync` / `retry_async` in `retry.py`. The retry loop, delay calculation, and error handling are near-identical (~90 lines each). Extract shared core logic.
- [ ] Add missing `@staticmethod` on `ResponseParser._apply_enum_mappings` in `parsing.py` (doesn't use `self`).

#### Duplication — Tests (high impact)

- [ ] Move `_OLLAMA_AVAILABLE` check to `integration_tests/conftest.py` using a `pytest_collection_modifyitems` hook or session-scoped fixture. Currently copy-pasted in all 7 integration test files (~14 lines each).
- [ ] Move shared integration test data (temperature, sentiment) to `integration_tests/conftest.py`. Currently duplicated across `test_detector`, `test_regressor`, and `test_async`.
- [ ] Consolidate integration test agent fixtures (`temperature_detector`, `sentiment_classifier`, `temperature_regressor`) to `integration_tests/conftest.py`. Currently duplicated between individual test files and `test_async.py`.

#### Duplication — Unit Tests (medium impact)

- [ ] Consolidate `_batch_chat_response()` helper — duplicated between `tests/test_labeler.py` and `tests/test_batch.py`. Move to `tests/helpers.py` or `tests/conftest.py`.
- [ ] Consolidate shared test model classes (`DummyReason`, etc.) — defined independently in `test_labeler.py`, `test_prompt_params.py`, and `test_enum_parsing.py`. Define once in `tests/conftest.py`.
- [ ] Consolidate `_make_response()` variants in `tests/test_pipeline.py` with `make_response()` in `tests/conftest.py`.

#### Architecture (medium impact)

- [ ] Reduce sync/async duplication in `Labeler.act_in_batch` / `act_in_batch_async` (~120 lines each with near-identical structure). Consider extracting shared chunk-and-flatten logic.
- [ ] Reduce sync/async duplication in `Pipeline.act` / `act_async` (~35 lines each with identical control flow). Consider a shared generator pattern.

#### Complexity (medium impact)

- [ ] Simplify enum mapping logic in `ResponseParser._get_enum_mappings` (`parsing.py`). The nested type extraction for `Optional[StrEnum]` fields could be a separate helper.

### Infrastructure

- [x] Consider Docker BuildKit secrets for `GITLAB_TOKEN` in `Dockerfile` — currently passed as build ARG, which may persist in layer history.
- [x] Document Linux workaround for `host.docker.internal` in `docker-compose.yml` — works on Docker Desktop (Windows/Mac) but not natively on Linux.
