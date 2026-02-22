
# Backlog - Agents7

### Pending Manual Testing

- [ ] End-to-end manual test of the full Agent Builder UI (all 5 steps: Define, Configure, Test, Evaluate, Export).
- [ ] Test prompt versioning and A/B comparison UI: save/load/delete versions, run A/B comparison, verify side-by-side results. Confirm Single Input, Batch Test, and History tabs still work.
- [ ] Test JSONL file upload in Step 3 (Batch Test) and Step 4 (Evaluate).

### 22 Feb 2026 - Refactoring Sweep

#### Documentation & Correctness

- [x] Fix README.md regressor example: rename variable `detector` to `regressor` and fix config path from `demo_detector.yml` to `demo_temperature_detector.yml` (line ~105).
- [x] Add `BaseAgent` to `agents/__init__.py` `__all__` — it is documented in README but not importable via `from agents7.agents import BaseAgent`.
- [x] Remove manual `NOT_APPLICABLE` from `TemperaturePropertyReason` in `models/demo.py` — it is auto-added by `_extend_enum_with_not_applicable()`, so defining it manually is redundant and risks description drift.
- [x] Create `examples/regressor.py` example script — all other agent types have examples, but Regressor is missing.

#### Type Hints

- [x] Add missing type hints to core code: `BaseAgent.__post_init__` (`-> None`), abstract `act()` param types in `base.py`.
- [x] Add missing type hints to UI pages: all `render()` functions, `_render_*` helpers in `3_Test.py`, `_show_confusion_matrix` in `4_Evaluate.py`, `_write_eval_results`/`_generate_usage_example` in `5_Export.py`.
- [x] Fix type hint inconsistencies: `make_progress_callback` returns `callable` (should be `Callable`), `_parse_batch` returns bare `list` (should be `list[dict]`), `_enum_code` has bare `list` param (should be `list[EnumMember]`).

#### Code Duplication — Agents

- [ ] Extract prompt-building logic into a shared helper method on `Labeler` — identical pattern repeated in `prompt`/`prompt_for_batch` properties across `labeler.py`, `detector.py`, `classifier.py`, `regressor.py` (8 occurrences).
- [ ] Consolidate duplicated `chat_kwargs` docstrings in `labeler.py` — identical param blocks repeated 6+ times across `act`, `act_async`, `act_in_batch`, etc.
- [ ] Consolidate duplicated docstrings in `llm/base.py` and `llm/ollama.py` — same param/return blocks repeated 8+ times across sync/async/stream method pairs.

#### Code Duplication — UI

- [ ] Extract duplicated list-expansion pattern into a helper in `_pages/__init__.py` — the `while len(items) < num: items.append(...)` / `items = items[:num]` pattern appears in `1_Define.py` (fields, enum members), `2_Configure.py` (examples), and `4_Evaluate.py` (eval rows).
- [ ] Deduplicate type map in `codegen/model_generator.py` — local `type_map` dict in `_class_for_property()` duplicates the module-level `_FIELD_TYPE_MAP`.

#### Code Duplication — Tests

- [ ] Move shared test helpers (`mock_metrics`, `make_agent_mock`) from `tests/helpers.py` to `tests/conftest.py` as proper pytest fixtures.
- [ ] Deduplicate `_make_response()` helpers — nearly identical functions exist in `test_ui_comparison.py` and `test_ui_evaluation.py`.

#### Test Infrastructure

- [ ] Add `tests/conftest.py` with session-scoped `chdir` to project root — unit tests use relative paths for YAML configs but lack the `_chdir_to_project_root()` fixture that `integration_tests/conftest.py` has, so tests may fail if run from a non-root directory.
- [ ] Standardize test comment headers — some files use `# --- Section ---`, others use `# === Section ===`. Adopt `# --- Section ---` consistently per testing conventions.
- [ ] Add missing comment headers to `test_ui_export.py` and `test_ui_jsonl.py` — group tests by function/category.

#### Code Quality

- [ ] Extract cache callback setup into helper in `llm/base.py` — the `if key is not None: result._on_complete = functools.partial(...)` block is duplicated 4 times across streaming methods.
- [ ] Consolidate enum-handling methods in `labeler.py` — `_apply_enum_mappings` and `_apply_enum_mappings_to_item` could be merged to reduce the 4-method chain.
- [ ] Make state dict `index()` calls safe in `1_Define.py` and `2_Configure.py` — `.index(value)` on selectbox options will raise `ValueError` if the value isn't in the list; use a safe fallback.

#### Infrastructure

- [ ] Consider Docker BuildKit secrets for `GITLAB_TOKEN` in `Dockerfile` — currently passed as build ARG, which may persist in layer history.
- [ ] Document Linux workaround for `host.docker.internal` in `docker-compose.yml` — works on Docker Desktop (Windows/Mac) but not natively on Linux.
