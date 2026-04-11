
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

### Benchmarks — 6th April 2026

- [ ] Save benchmark reports automatically after each benchmark run (last run's reports were not persisted).
