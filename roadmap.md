# agents7-framework Roadmap

The standing direction for development sessions. `backlog.md` holds the
*items*; this file holds the *ordering, gates, and invariants* that decide
which items to pick and which to refuse. Consult it **before selecting work**.

**Maintenance:** update when a direction-level decision lands, when a gate
clears, or when a phase completes. Keep it short — ordering and gates only;
rationale lives in the docs or ADRs it cites.

## Current phase: measurement first

1. ~~**Compose the eval pieces into one workflow.**~~ Shipped in
   v4.11.0 as `agents7.eval.run_eval_suite` — eval data → run →
   metrics → call log → report in one documented call, backing both
   the Python API and `agents7 eval`. Gate cleared.
2. **Adopt it across the catalog.** Only `scripts/eval_agent.py` and
   `tagger_agent`'s tests consume `agents7.eval` today; every catalog
   agent should be evaluable the same way.
   **Gate: no accuracy or latency claim in a changelog, README, or
   refinement log without an eval run behind it.**
3. **Then resume feature work.** The catalog `ToolAgent` showcase
   (agents_catalog backlog) is the next consumer-facing item once
   measurement is in place — tool-calling v1 is shipped and validated
   end-to-end across all three providers (v4.4.0–v4.9.0).

## Standing invariants (apply to every session)

- **An agent is a frozen config around a validated call, not an
  autonomous loop.** `ToolAgent`'s bounded loop is the one deliberate
  exception. Adding autonomy elsewhere is a design decision, not a
  patch — see `docs/design.md`.
- **Not every agent needs an LLM.** When the signal space is
  enumerable, a rule-based pipeline is legitimate and often far faster
  at equal accuracy. Consider it before reaching for a model.
- **Provider-agnostic by default.** A feature landing for one of
  Ollama / Anthropic / OpenAI-compat either works across all three or
  states explicitly why it is scoped to one.
- **Measure before claiming** — see the phase gate above.

## What NOT to do (recent refuted directions)

- **Don't use an LLM classifier when the signal space is enumerable.**
  `tagger_agent` began as an LLM-based classifier and became a pure
  rule-based pipeline: ~1000x faster at the same accuracy
  (`docs/design.md`).
- **Don't leave thinking enabled for structured output.** Thinking
  models (qwen3, gemma3n) overran structured-output calls and dropped
  fields; `think=False` is now always sent explicitly (v4.9.1). A
  dropped-field bug is not to be "fixed" by loosening the schema.
