
# Backlog - Agents7

### Pending Manual Testing

- [ ] End-to-end manual test of the full Agent Builder UI (all 5 steps: Define, Configure, Test, Evaluate, Export).
- [ ] Test prompt versioning and A/B comparison UI: save/load/delete versions, run A/B comparison, verify side-by-side results. Confirm Single Input, Batch Test, and History tabs still work.
- [ ] Test JSONL file upload in Step 3 (Batch Test) and Step 4 (Evaluate).
- [ ] Test auto-discovery flow: upload JSONL, write intent, click Discover Schema, verify form fills, then Generate Prompt & Examples in Step 2.

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.

### SceneChatter — 5th April 2026

- [ ] Integration test: run `SceneChatter` with a real LLM and verify the agent stays in scene context (e.g., rejects modern concepts, references other characters, pursues its goal).

### Benchmarks — 6th April 2026

- [ ] Save benchmark reports automatically after each benchmark run (last run's reports were not persisted).

### Refactor review — 11th April 2026

- [x] Deduplicate stream wrapper classes — `ChatterStreamResult` and `GuardedChatterStreamResult` (sync + async) repeat identical iteration/finalization logic. MEDIUM impact, small effort.
- [x] Extract `_guard_and_fix` retry logic — method is 58 lines in `guarded.py` with sync/async duplication. Extract verdict-decision logic. MEDIUM impact, small effort.
- [x] Move `_load_guarded_yaml` (83 lines) into `config_loader.py` — consolidate all YAML loading in one module. MEDIUM impact, small effort.
- [x] Consolidate test response builders — `chat_response_dict()` duplicated in 4 test files, `make_chatter()` in 3 files, `make_guard()` in 2 files. Moved to `helpers.py`. MEDIUM impact, small effort.
- [x] Clarify `conftest.py` vs `helpers.py` responsibilities — both contain fixtures and factories with overlapping scope. LOW impact, small effort.
- [x] Extract integration test boilerplate — `test_classifier.py`, `test_detector.py`, `test_regressor.py` share ~140 lines of identical parametrize/assert/batch patterns. LOW impact, medium effort.
