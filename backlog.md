
# Backlog - Agents7

- [x] Check `pydantic_aio` library, it seems interesting.
- [x] Add tests for Labeler agent directly (currently untested).
- [x] Add explicit async tests for `act_async` and
  `act_in_batch_async`.
- [x] Add unit tests for JSON parsing fixes
  (`_fix_malformed_json`, `_apply_enum_mappings`,
  markdown block removal) without needing an LLM.
- [x] Add error path tests (invalid YAML configs, wrong model
  types, network failures).
- [x] Add retry/resilience mechanism with configurable backoff
  at the engine or agent level.
- [x] Add streaming support (Ollama already supports it).
- [x] Add response caching for deterministic calls (same input +
  seed + temperature=0).
- [x] Add validation hooks (post-processing step for custom
  business logic on parsed models).
- [x] ~~Evaluate extracting `cpinto` utilities~~ (discarded).
- [x] ~~Clean up sketches folder~~ (discarded).
- [x] Consider agent chaining / pipelines (composing agents
  sequentially).
