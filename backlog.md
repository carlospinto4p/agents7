
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

### Infrastructure

- [x] Consider Docker BuildKit secrets for `GITLAB_TOKEN` in `Dockerfile` — currently passed as build ARG, which may persist in layer history.
- [x] Document Linux workaround for `host.docker.internal` in `docker-compose.yml` — works on Docker Desktop (Windows/Mac) but not natively on Linux.
