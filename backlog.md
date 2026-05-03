
# Backlog - Agents7

### Pending Manual Testing

- [ ] End-to-end manual test of the full Agent Builder UI (all 5 steps: Define, Configure, Test, Evaluate, Export).
- [ ] Test prompt versioning and A/B comparison UI: save/load/delete versions, run A/B comparison, verify side-by-side results. Confirm Single Input, Batch Test, and History tabs still work.
- [ ] Test JSONL file upload in Step 3 (Batch Test) and Step 4 (Evaluate).
- [ ] Test auto-discovery flow: upload JSONL, write intent, click Discover Schema, verify form fills, then Generate Prompt & Examples in Step 2.

### Scan — 2nd May 2026

#### Test coverage

- [x] `agents/memory.py:124` — `JsonFileMemory.load()` with corrupted JSON untested. **MED/M**
- [x] `model_selection.py:204` — `find_best_model()` with `runs=0` or empty model list untested. **MED/M**
- [x] `llm/streaming.py:56` — `_extract()` with missing chunk attributes untested. **MED/M**
- [x] `agents/jitter.py:49` — `_inject_typo()` word-length boundary (exactly 4 chars) untested. **MED/S**
- [x] `tests/test_labeler.py:261` — `assert result is not None` is a weak assertion; verify actual output model contents. **MED/S**

### Bugs — 2nd May 2026

- [ ] Benchmark bad_signal 'la capital' too broad — fires on in-character redirect phrases (e.g. "con la capital de Australia ya nos vamos a perder el hilo"). Narrow the signal or add an exclusion for redirect context.
- [ ] qwen3:4b case 5 injection guard miss — guard accepted a response containing 'pirata' in a humorous aside (rej=0). The persona-swap injection was not caught. Investigate guard prompt or threshold for this model.
- [ ] qwen3:4b case 8 char_break retry pattern — in earlier runs the model said 'soy un modelo' after guard rejection, exhausting retries. Tighten the fix instruction or increase max_guard_retries for weaker models.

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.
