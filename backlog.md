
# Backlog - Agents7

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.


### 9th May 2026 — /scan

#### Test coverage

- [x] `agents7_framework/src/agents7/agents/classifier.py`, `detector.py`, `regressor.py`, `context_guard.py` — Add unit tests for `from_yaml()` in all four agents — MED — S
- [x] `agents7_framework/tests/test_model_selection.py:78-108` — Add tests for `_compare_output()` with `None` fields and partial field matching — MED — S
- [x] `agents7_framework/src/agents7/llm/dispatcher.py:259` — Add dedicated test for `asyncio.TimeoutError` path in `execute_async()` — MED — S
- [x] `agents7_framework/src/agents7/agents/batch.py:194` — Add tests for malformed chunk responses in `_flatten()` — MED — M
- [ ] `agents7_framework/src/agents7/agents/guarded_scene.py:35` — Test `from_yaml()` persona_summary derivation with scene context — MED — M
