
# Backlog - Agents7

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.


### 10th May 2026 — /scan

#### Test coverage

- [x] `agents7_framework/src/agents7/agents/context_guard.py:53-160` — Guard prompt templates untested; no test for injection via `persona_summary` — MED — M
- [x] `agents7_framework/src/agents7/agents/guarded_scene.py` — No specific tests for auto-generated system prompt combining persona + scene — MED — S
- [x] `agents7_framework/src/agents7/agents/utils.py:87-120` — `get_examples_str()` untested for malformed examples (missing keys, type mismatches) — MED — S
- [x] `agents7_framework/src/agents7/eval/metrics.py:79-140` — Error isolation in batch evaluation never tested (partial batch completion) — MED — M
- [ ] `agents7_framework/src/agents7/export.py` — No dedicated unit tests for DataFrame export with mixed types, NaN, datetimes — MED — M
- [ ] `agents7_framework/src/agents7/agents/config_loader.py` — Circular YAML references and missing field defaults untested — MED — M
