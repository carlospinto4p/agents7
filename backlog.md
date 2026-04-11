
# Backlog - Agents7

### GuardedChatter Benchmark Follow-up

- [x] Investigate why `qwen3:4b` guard with `think: true` misses obvious prompt injection ("Say Arrr!") — done. Root cause is the chatter, not the guard: qwen3:4b eagerly complies with injections, and the guard sees superficially in-character responses. Model is unsuitable.
- [x] Benchmark guard performance without `think: true` — done. think=true helps e4b (6/6 vs 5/6), hurts e2b (5/6 vs 6/6), makes no difference for qwen3:4b (4/6 both). Recommended: e4b+think=true, e2b+think=false.

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

### Hooks

- [x] Fix PreToolUse hook stdin consumption — done in v3.14.4 (all hooks now capture stdin before piping to python).

### Refactor review — 11th April 2026

- [x] Extract YAML config field-casting into a shared helper — `config_loader.py`, `personality.py`, and `guarded.py` each repeat 10+ identical `if "field" in config` blocks. HIGH impact, small effort.
- [ ] Consolidate LLM provider shared logic — `anthropic.py`, `ollama.py`, `openai_compat.py` duplicate ~80% of `_build_params()`, `_parse_response()`, `chat()`, `generate()`. Extract shared base helpers. HIGH impact, medium effort.
- [ ] Split `LLMEngine.__init__` provider dispatch (57 lines, 3 nested branches) — extract to a factory method or registry. MEDIUM impact, small effort.
- [ ] Deduplicate stream wrapper classes — `ChatterStreamResult` and `GuardedChatterStreamResult` (sync + async) repeat identical iteration/finalization logic. MEDIUM impact, small effort.
- [ ] Extract `_guard_and_fix` retry logic — method is 58 lines in `guarded.py` with sync/async duplication. Extract verdict-decision logic. MEDIUM impact, small effort.
- [ ] Move `_load_guarded_yaml` (83 lines) into `config_loader.py` — consolidate all YAML loading in one module. MEDIUM impact, small effort.
- [ ] Consolidate test response builders — `chat_response_dict()` duplicated in 4 test files, `make_chatter()` in 4 files, `make_guard()` in 2 files. Move to `conftest.py`. MEDIUM impact, small effort.
- [ ] Clarify `conftest.py` vs `helpers.py` responsibilities — both contain fixtures and factories with overlapping scope. LOW impact, small effort.
- [ ] Extract integration test boilerplate — `test_classifier.py`, `test_detector.py`, `test_regressor.py` share ~140 lines of identical parametrize/assert/batch patterns. LOW impact, medium effort.
