
# Backlog - Agents7

### Pending Manual Testing

- [ ] End-to-end manual test of the full Agent Builder UI (all 5 steps: Define, Configure, Test, Evaluate, Export).
- [ ] Test prompt versioning and A/B comparison UI: save/load/delete versions, run A/B comparison, verify side-by-side results. Confirm Single Input, Batch Test, and History tabs still work.

### 15th February 2026 — Refactoring

#### Code Deduplication

- [x] `labeler.py`: Extract duplicated `chat_kwargs` setup (4 occurrences) into a shared helper method.
- [x] `ollama.py`: Extract duplicated options dict construction (8 occurrences) into a `_build_options()` static method.
- [x] `pipeline.py`: Remove unused `as_completed` import. (False positive — `as_completed` is used in `labeler.py`.)
- [x] `models/base.py`: Extract shared enum-extension logic from `Property.__init_subclass__` and `ClassificationProperty.__init_subclass__`.
- [x] UI `_pages/4_Evaluate.py` and `_pages/5_Export.py`: Extract duplicated `_get_expected_fields()` to `_pages/__init__.py`.
- [x] UI `_pages/3_Test.py` and `_pages/4_Evaluate.py`: Extract duplicated nested `_progress()` callback to a shared module-level helper.
- [x] UI `codegen/model_generator.py`: Extract duplicated enum generation pattern (3 occurrences) into a `_enum_code()` helper.

#### Tests

- [x] Convert test classes to plain functions (50 classes across 10 files).
- [x] Create `tests/helpers.py` with shared fixtures: `make_labeler()`, `chat_chunk()`, `async_iter()`, mock metrics/responses (deduplicated from 7+ files).

#### UI Convention Violations

- [x] Move imports to module level in all `_pages/*.py` files.
- [x] Break down long functions in `_pages/3_Test.py`: `_render_ab_compare()`, `_show_comparison()`, `_render_version_management()`.

#### Style

- [x] Fix line-length violations (>78 chars) in docstrings across `labeler.py`, `classifier.py`, `ollama.py`, `base.py`, and test files (~37 lines).

### Low Priority

- [ ] Add more LLM providers (e.g., OpenAI, Anthropic) beyond Ollama.
- [ ] Add persistent disk-based cache option (SQLite) alongside the in-memory LRU.
- [ ] Publish Docker image to a container registry for easier deployment.
