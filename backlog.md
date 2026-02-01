
# Backlog - Agents7

- [x] Check `pydantic_aio` library, it seems interesting.
- [x] Add tests for Labeler agent directly (currently untested).
- [ ] Add explicit async tests for `act_async` and
  `act_in_batch_async`.
- [x] Add unit tests for JSON parsing fixes
  (`_fix_malformed_json`, `_apply_enum_mappings`,
  markdown block removal) without needing an LLM.
- [ ] Add error path tests (invalid YAML configs, wrong model
  types, network failures).
- [ ] Add retry/resilience mechanism with configurable backoff
  at the engine or agent level.
- [ ] Consider agent chaining / pipelines (composing agents
  sequentially).
- [ ] Add streaming support (Ollama already supports it).
- [ ] Add response caching for deterministic calls (same input +
  seed + temperature=0).
- [ ] Add validation hooks (post-processing step for custom
  business logic on parsed models).
- [ ] Evaluate extracting `cpinto` utilities (YAML loading,
  logging) to reduce private dependency.
- [ ] Clean up sketches folder (remove or move to docs/examples).
