---
# Baselines for produced numbers: only when the eval code, a benchmark, or a figure is open.
paths:
  - "src/agents7/eval/**"
  - "benchmarks/**"
  - "tests/test_style_history.py"
---

# A Number With No History Is Not A Measurement

A figure produced by running a model (an accuracy, a violation rate, a
latency, a cost) is one sample from a process that varies. It is a
measurement only when there is a record of what the same configuration
scored before. One run is not a baseline, a before or an after.

Applies to the `agents7` framework and to every consumer of a figure it produces:
agents built on it, the UI, changelogs, backlog items, replies.

Set 30th August 2026 at the operator's instruction. That day a config
rule was reverted and two regressions were reported, one in a
changelog, all from single runs; none held up over more runs.

## The rules

1. **Persist the whole invocation, with each run's script label.** The
   comparable unit is one invocation: every repeat and every script.
   `agents7.eval.style_history` does this (`record_style_run`,
   `append_style_run`, `load_style_history`). A new measurement gets
   the same before its first published number.
2. **Identify the configuration by hash, not by version label.**
   `fingerprint_run` hashes the resolved system prompt with the
   contract's rules and scripts. Bands compare on `configuration_key`
   (the same hash without rule ids), and each rule's band holds only
   runs scored under that rule's own definition. Adding a rule keeps
   other rules' baselines; changing a rule's check under the same id
   starts a new one.
3. **No verdict below two prior invocations.** Return
   `Verdict.NO_BASELINE`, and print the prior count beside every
   verdict. Two (`MIN_BASELINE_RUNS`) is a floor, not a target.
4. **Call a change only when the rate leaves its band by more than its
   sampling margin** (`sampling_margin`).
   - From four prior invocations (`MIN_SPREAD_PRIORS`), the margin is
     the spread of their scores, read at its upper 90% bound
     (`SPREAD_CONFIDENCE`). Below four, it is estimated from inside the
     current invocation, and `margin_source` says so.
   - A rule broken less than once per invocation is shown, not judged
     (`Verdict.RARE`).
5. **Never quote a single invocation as a baseline** in a changelog,
   README, backlog item or reply. Write "one run said X", not "X".
6. **A record keeps the threshold it was scored under**, so tightening
   one later does not turn old invocations into failures.
7. **The history is append-only.** The one exception: when a
   fingerprint function is corrected, records under the old key may be
   refiled under the new one, only if the commit shows all of these:
   - no measured number changes (the file diffed with the identity
     fields removed is byte-identical);
   - version control shows the config, the contract and the hashed
     code were unchanged between the run and now;
   - the contract header and the changelog say the old key, the new
     key and why;
   - it corrects a key; it never moves records into a band they did
     not belong to.
8. **Put other possible causes of a move on the record.** `answered_by`
   is the model the provider says answered, and a band uses one model
   only. `lost_runs` counts conversations that raised and were dropped.
   Before crediting a move to an edit, run the unchanged config the
   same day. A newly found cause goes on the record too.
9. **The sample size is independent runs, not replies.** A
   conversation's turns depend on each other: 80 replies from 5
   conversations carry about a third of the information of 80 draws.
   Build every test and margin on the run of one script: measure the
   spread within each script and pool across scripts, since the gap
   between scripts is fixed, not noise.

## Limits

A band says when a difference is larger than the variation already
seen. It does not explain that variation. A wide band cannot detect
small changes; only reducing the variation fixes that.

## Related

- `metrics-completeness.md`: a declared measurement must be produced.
  This rule adds that a produced one must be comparable before it is
  believed.
