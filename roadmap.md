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
2. ~~**Adopt it across the catalog.**~~ Both consumers now route
   through `run_eval_suite` (the catalog and a classifier): the
   script owns only discovery, and all five evaluable agents score
   identically. Gate cleared — but the standing invariant below
   keeps it enforced.
   **Standing gate: no accuracy or latency claim in a changelog,
   README, or refinement log without an eval run behind it.**
3. ~~**Then resume feature work.**~~ The catalog `ToolAgent` showcase
   shipped (store_assistant v0.1.0), and v4.14.0 closed the measurement
   gap it exposed: tool agents are scored on their **trajectory**
   (which tools, in what order, with what arguments) rather than on a
   label, so the standing gate above now covers every agent type the
   framework builds. Gate cleared.
4. ~~**Run the tool eval for real.**~~ Run 3rd September 2026
   against four Ollama models; `eval_agent.py` gained `--model`
   and `--trajectory-mode` to make it repeatable, and the
   sample's README now carries the measured table instead of
   anecdotes (one refuted: `qwen3:8b` scores 12/12). Gate
   cleared — the standing invariant below keeps it enforced.
5. ~~**Make a passing trajectory row mean what it says.**~~ Both
   rows that passed while the agent misbehaved are closed. An
   empty `tools` list is scored under `exact` in either mode, so
   a row asserting "no tool" can fail (v4.38.2); and
   `ExpectedCall.defaults` pins an argument as absent-or-equal,
   so an invented optional argument fails without failing a
   model that correctly omits it (v4.39.0). `store_assistant`
   was rescored under both. Gate cleared.
6. ~~**A row still passes on a call the tool rejected.**~~
   Closed in v5.0.0. `ToolRegistry.execute` returns a
   `ToolOutcome` carrying the failure as its own field,
   `collect_trajectory` pairs it onto each `ToolCall`, and a
   rejected call satisfies no expectation — with
   `ExpectedCall.allow_failure` for the one row that tests the
   error path. `store_assistant` was rescored on four models
   under the corrected rule. Gate cleared.
7. **Next: resume feature work.** No item of its own — the three
   instances of "a row passes while the agent did the wrong
   thing" are all closed, so the trajectory scorer is no longer
   what limits a claim. The standing gate in step 2 still
   applies to every claim made.

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
  One classifier agent began LLM-based and became a pure
  rule-based pipeline: ~1000x faster at the same accuracy
  (`docs/design.md`).
- **Don't leave thinking enabled for structured output.** Thinking
  models (qwen3, gemma3n) overran structured-output calls and dropped
  fields; `think=False` is now always sent explicitly (v4.9.1). A
  dropped-field bug is not to be "fixed" by loosening the schema.
