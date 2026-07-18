
# Backlog - Agents7

### Eval — 18th July 2026

- [ ] **Compose the eval pieces into one documented workflow** — this is
  roadmap phase 1 and currently has no item behind it in any subproject.
  `agents7.eval` (metrics, A/B, export, `load_eval_rows`,
  `normalise_rows`, `model_output_fields`, `build_agent`), the
  `agents7 eval` CLI subcommand, and `summarize_call_log` all exist but
  are separate entry points rather than one path a user follows. Tie
  them into a single route — eval data → run → metrics → report —
  documented end to end in the README. Note the gate this clears: no new
  eval primitive should land before the existing ones compose. (HIGH, M)
- [ ] **Adopt the composed eval route across the catalog** (roadmap phase
  2, blocked on the item above). Only `scripts/eval_agent.py` and
  `tagger_agent`'s tests consume `agents7.eval` today; every catalog
  agent should be evaluable the same way, so that the standing gate — no
  accuracy or latency claim without an eval run behind it — is
  enforceable in practice. (MED, M)
