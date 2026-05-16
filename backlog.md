
# Backlog - Agents7

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.


### 10th May 2026 — /scan

#### Test coverage

- [ ] `agents7_framework/src/agents7/export.py` — No dedicated unit tests for DataFrame export with mixed types, NaN, datetimes — MED — M
- [ ] `agents7_framework/src/agents7/agents/config_loader.py` — Circular YAML references and missing field defaults untested — MED — M
