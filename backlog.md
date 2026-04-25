
# Backlog - Agents7

### Pending Manual Testing

- [ ] End-to-end manual test of the full Agent Builder UI (all 5 steps: Define, Configure, Test, Evaluate, Export).
- [ ] Test prompt versioning and A/B comparison UI: save/load/delete versions, run A/B comparison, verify side-by-side results. Confirm Single Input, Batch Test, and History tabs still work.
- [ ] Test JSONL file upload in Step 3 (Batch Test) and Step 4 (Evaluate).
- [ ] Test auto-discovery flow: upload JSONL, write intent, click Discover Schema, verify form fills, then Generate Prompt & Examples in Step 2.

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.

### Refactor Pass — 25th April 2026

- [x] Reduce sync/async duplication in `agents/guarded.py`: `_guard_and_fix()` and `_guard_and_fix_async()` are 90% identical — extract the shared retry loop into a helper or consolidate via a protocol/callback approach.
- [x] Reduce sync/async duplication in `agents/labeler.py`: `_retry_parse_loop()` (line ~686) and `_retry_parse_loop_async()` (~721) are 34-line near-copies — extract shared logic.
- [x] Fix NumPy-style docstrings (`Args:` / `Returns:`) in `agents/detector.py`, `agents/classifier.py`, and `agents/regressor.py` — convert to project-standard Sphinx/reST (`:param:`, `:return:`).
- [x] Verify `model_selection.py` is dead code — confirmed NOT dead: has tests in `tests/test_model_selection.py` and is documented in `README.md`. No action needed.
