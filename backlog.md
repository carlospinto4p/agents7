
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

### Hooks

- [x] Fix PreToolUse hook stdin consumption — done in v3.14.4 (all hooks now capture stdin before piping to python).
