
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

- [ ] Extract shared Ollama model selector widget — `1_Define.py` (discovery section) and `2_Configure.py` both implement the same `check_connection → list_available_models → selectbox/fallback` pattern. Extract to a reusable helper in `_pages/__init__.py` or `ollama_utils.py`.

#### Test Coverage (medium impact)

- [ ] Add tests for batch flattening logic — `Labeler.act_in_batch` (line ~295) and `act_in_batch_async` (line ~369) have complex state management (per-input step tracking, idx increments, failed item propagation) with no dedicated tests.
- [ ] Add parametrized tests for enum conversion edge cases — nested enums in lists, optional enum fields, missing enum members in `_convert_enum_names_to_values` and `_apply_enum_mappings`.

#### Style & Consistency (low impact)

- [ ] Standardize type hint style — some methods use `dict = None` while others use `dict | None = None`. Normalize to `dict | None = None` throughout.
- [ ] Standardize docstring format — some properties use single-line docstrings, others use multi-line with `:return:`. Pick one style for properties vs full methods.
- [ ] Add missing type hints in `streaming.py` — `_extract()` method `chunk` parameter lacks type annotation.
- [ ] Move `_default_mapper` in `pipeline.py` to a `@staticmethod` on `PipelineStep` for better encapsulation.

### Infrastructure

- [x] Consider Docker BuildKit secrets for `GITLAB_TOKEN` in `Dockerfile` — currently passed as build ARG, which may persist in layer history.
- [x] Document Linux workaround for `host.docker.internal` in `docker-compose.yml` — works on Docker Desktop (Windows/Mac) but not natively on Linux.
