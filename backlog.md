
# Backlog - Agents7

### Eval — 18th July 2026

- [ ] **Adopt the composed eval route across the catalog** (roadmap phase
  2, unblocked by `run_eval_suite` in v4.11.0). Only
  `scripts/eval_agent.py` and `tagger_agent`'s tests consume
  `agents7.eval` today; every catalog
  agent should be evaluable the same way, so that the standing gate — no
  accuracy or latency claim without an eval run behind it — is
  enforceable in practice. (MED, M)
