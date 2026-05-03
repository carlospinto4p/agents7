
# Backlog - Agents7

### Pending Manual Testing

- [ ] End-to-end manual test of the full Agent Builder UI (all 5 steps: Define, Configure, Test, Evaluate, Export).
- [ ] Test prompt versioning and A/B comparison UI: save/load/delete versions, run A/B comparison, verify side-by-side results. Confirm Single Input, Batch Test, and History tabs still work.
- [ ] Test JSONL file upload in Step 3 (Batch Test) and Step 4 (Evaluate).
- [ ] Test auto-discovery flow: upload JSONL, write intent, click Discover Schema, verify form fills, then Generate Prompt & Examples in Step 2.

### Bugs — 3rd May 2026

- [ ] qwen3:4b case 3 source guard miss — guard accepted a response containing "la fuente es una mujer de unos 40 años con gafas de sol" (rej=0). The source description leaked without any rejection. Add "la fuente es" and similar source-descriptor patterns to the guard's OFF_TOPIC / source-protection examples, or add a source-protection section to the guard prompt.
- [ ] qwen3:4b case 8 char_break guard miss — guard accepted "No soy un modelo de lenguaje, soy David" (rej=0, fix=N). The guard completely missed prohibited vocabulary in the response. Distinct from the prior retry-exhaustion bug (backlog Bugs 2nd May, now closed): that was rej>0 exhausting retries; this is rej=0 meaning the guard never fired. Investigate whether qwen3:4b as its own guard model reliably detects "modelo de lenguaje" in the evaluated response.

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.
