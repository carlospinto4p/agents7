
# Backlog - Agents7

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.


### Scan — 17th May 2026

#### Test Coverage

- [x] Add edge-case tests for `BatchProcessor._chunk_inputs()` in `agents/batch.py:212-222`: `requests_per_prompt=1`, single item, empty list (MED, M)
- [x] Add tests for `LLMEngine` initialization and provider fallback in `llm/base.py` (MED, M)
