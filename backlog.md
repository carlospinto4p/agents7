
# Backlog - Agents7

### Pending Manual Testing

- [ ] End-to-end manual test of the full Agent Builder UI (all 5 steps: Define, Configure, Test, Evaluate, Export).
- [ ] Test prompt versioning and A/B comparison UI: save/load/delete versions, run A/B comparison, verify side-by-side results. Confirm Single Input, Batch Test, and History tabs still work.
- [ ] Test JSONL file upload in Step 3 (Batch Test) and Step 4 (Evaluate).
- [ ] Test auto-discovery flow: upload JSONL, write intent, click Discover Schema, verify form fills, then Generate Prompt & Examples in Step 2.

### Bugs — 2nd May 2026 (benchmark findings)

- [x] Guard parse reliability: guard occasionally returns Python class source instead of JSON, triggering fail-open. Happened 3/10 times in benchmark. Investigate whether a stricter output format instruction or a retry on parse failure would help.
- [x] Scene-break enforcement: case 9 (Tokyo restaurants) failed — character engaged with off-topic content when guard parse failed. Strengthen scene rules in `demo_guarded_scene_chatter.yml` and/or add OFF_TOPIC examples to guard prompt that cover location/geography deflection.

### Bugs — 2nd May 2026 (benchmark v2 findings)

- [ ] Guard fix echoes forbidden vocabulary: when generating a `fix` response, the guard quotes the user's challenge back (e.g. "¿Modelo de lenguaje? No sé de qué hablas.") — valid human behaviour but trips the bad_signal check. Add a constraint to the guard prompt: fix responses must not use the prohibited vocabulary list.
- [ ] Case 9 OFF_TOPIC echo-redirect: guard accepts responses where the character mentions the off-topic word while redirecting ("¿Tokio? Eso no viene al caso"). Either add a guard example showing echo-redirects are still OFF_TOPIC, or narrow the bad_signal to require an engagement signal alongside "tokio".

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.
