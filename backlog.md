
# Backlog - Agents7

### Pending Manual Testing

- [ ] End-to-end manual test of the full Agent Builder UI (all 5 steps: Define, Configure, Test, Evaluate, Export).
- [ ] Test prompt versioning and A/B comparison UI: save/load/delete versions, run A/B comparison, verify side-by-side results. Confirm Single Input, Batch Test, and History tabs still work.
- [ ] Test JSONL file upload in Step 3 (Batch Test) and Step 4 (Evaluate).
- [ ] Test auto-discovery flow: upload JSONL, write intent, click Discover Schema, verify form fills, then Generate Prompt & Examples in Step 2.

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.

### SceneChatter — 5th April 2026

- [x] ~~Integration test for SceneChatter~~ — removed: SceneChatter uses the same `engine.chat()` path as Chatter, no new integration to test. Feature-level behavior tests don't belong in integration tests.

### Refactor Pass — 25th April 2026

- [ ] Remove `from __future__ import annotations` from the remaining 35 source files (all under `src/agents7/`). Already done for `chatter.py`, `guarded.py`, `guarded_scene.py`, `context_guard.py` — the rest still carry it.
- [ ] De-duplicate YAML config loading in `agents/config_loader.py` and `agents/scene.py`: extract a shared `_load_base_chatter_config()` helper to eliminate the repeated engine/persona/optional-field pattern across `load_guarded_yaml_settings()`, `load_guarded_scene_yaml_settings()`, and `_load_scene_yaml()`.
- [ ] Fix inconsistent optional-field casting in `agents/scene.py` `_load_scene_yaml()`: replace inline `if "temperature" in raw:` / `if "think" in raw:` blocks with a call to the existing `cast_optional_fields()` utility (already used in `config_loader.py`).
- [ ] Reduce sync/async duplication in `agents/guarded.py`: `_guard_and_fix()` and `_guard_and_fix_async()` are 90% identical — extract the shared retry loop into a helper or consolidate via a protocol/callback approach.
- [ ] Reduce sync/async duplication in `agents/labeler.py`: `_retry_parse_loop()` (line ~686) and `_retry_parse_loop_async()` (~721) are 34-line near-copies — extract shared logic.
- [ ] Fix NumPy-style docstrings (`Args:` / `Returns:`) in `agents/detector.py`, `agents/classifier.py`, and `agents/regressor.py` — convert to project-standard Sphinx/reST (`:param:`, `:return:`).
- [ ] Verify `model_selection.py` is dead code (no import found outside its own docstring) and remove if so.

### Benchmarks — 6th April 2026

- [x] Save benchmark reports automatically after each benchmark run (last run's reports were not persisted).
