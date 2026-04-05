
# Backlog - Agents7

### GuardedChatter Benchmark Follow-up

- [ ] Investigate why `qwen3:4b` guard with `think: true` misses obvious prompt injection ("Say Arrr!") — may need prompt tuning or a different thinking approach.
- [ ] Benchmark guard performance without `think: true` to measure speed vs. quality tradeoff.

### Pending Manual Testing

- [ ] End-to-end manual test of the full Agent Builder UI (all 5 steps: Define, Configure, Test, Evaluate, Export).
- [ ] Test prompt versioning and A/B comparison UI: save/load/delete versions, run A/B comparison, verify side-by-side results. Confirm Single Input, Batch Test, and History tabs still work.
- [ ] Test JSONL file upload in Step 3 (Batch Test) and Step 4 (Evaluate).
- [ ] Test auto-discovery flow: upload JSONL, write intent, click Discover Schema, verify form fills, then Generate Prompt & Examples in Step 2.

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.

### Hooks

- [ ] Fix PreToolUse hook stdin consumption: all three hooks read stdin but only the first gets the JSON. Each hook should save stdin to a variable or use a shared mechanism.
