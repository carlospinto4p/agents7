
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

### Optimization Pass — 25th April 2026

- [x] Cache `build_agent_from_state()` in Streamlit pages (`ui/_pages/2_Configure.py:72`, `3_Test.py:128/185`, `4_Evaluate.py:93`) — wrap with `@st.cache_resource` keyed on the state dict hash so each Streamlit interaction doesn't rebuild the entire agent and reinitialize the LLM engine.
- [x] `SqliteMemory.save()` does a full DELETE + bulk re-insert every turn (`agents/memory.py:221-240`) — switch to `INSERT OR REPLACE` to write only changed rows, or append new messages instead of rewriting the whole conversation.
- [x] `SqliteCache.put()` runs `SELECT COUNT(*) FROM cache` on every write (`llm/cache.py:187-201`) — move the eviction check out of the per-call path: only count when the estimate (maintained in a cheap instance counter) says the threshold may be exceeded.
- [x] `JsonFileMemory._write_all()` serialises and rewrites the entire JSON file on every chat turn (`agents/memory.py:170-174`) — for large conversation stores this grows linearly per save; switch to per-conversation file sharding or SQLite so only the active conversation is written.
- [x] `_CacheKeyBuilder.make_key()` calls `json.dumps()` + SHA-256 on every LLM request (`llm/cache.py:77-89`), including serialising the full `output_format` JSON schema — pre-hash the schema once at engine construction and reuse the hash as a stable component of the key. Skipped: `output_format` is passed per-call, not stored in `LLMEngine`; pre-hashing would require threading a hash through the full call stack for negligible benefit (<50 µs per call).
- [x] Streaming `response` and `thinking` properties call `"".join()` on the full parts list on every access (`llm/streaming.py:31-38`) — add a dirty-flag cache so repeated reads (e.g., finalization + logging) don't re-join the list.
