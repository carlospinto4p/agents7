---
# Baselines for produced numbers: only when the eval code, a benchmark, or a figure is open.
paths:
  - "src/agents7/eval/**"
  - "benchmarks/**"
  - "tests/test_style_history.py"
---

# A Number With No History Is Not A Measurement

Project law for the `agents7` framework and everything that reads its
output. **Any figure produced by running a model — an accuracy, a
violation rate, a latency, a cost — is one sample from a process
that varies. It becomes a measurement only when there is a record
of what the same thing scored before.** A single run is evidence.
It is not a baseline, it is not a before, and it is not an after.

Set 30th August 2026 at the operator's instruction, after a
session in which four separate claims were made from single runs
and the instrument built that same day overturned every one.

Sibling to `metrics-completeness.md`. That rule is about a
measurement the framework declares and never produces. This one is
about a measurement the framework produces and cannot interpret.

## The failure class, precisely

Not "we should run things more than once". The failure is
**comparing two numbers that were never comparable, and acting on
the difference.**

The shape to recognise:

- A figure comes out of a run.
- A change is made.
- A figure comes out of another run.
- The two differ, and **the difference is attributed to the
  change**, because nothing on hand can attribute it to anything
  else.

Every step is reasonable. The conclusion is unfounded, and nothing
in the output says so.

## What happened

Three invocations of a chat agent's style contract on 30th August
2026 — identical config, identical scripted counterpart, identical
code. One rule, `one_message_when_dry`:

| Invocation | Runs | Rate |
|---|---|---|
| 3 repeats | 6 | 23% (11/48) |
| 5 repeats | 10 | 70% (56/80) |
| 5 repeats | 10 | 41% (33/80) |

A two-proportion test on the last two gives z ≈ 3.9, and this file
originally read that as proof that something outside the config
varies. **That inference was wrong, and it is corrected here on
30th August 2026 rather than deleted, because the mistake is the
more useful half of the lesson.**

The test assumes 80 independent replies. There are not 80. The
contract plays one script five times, and only the cold script's
16 short lines exercise this rule — so the invocation is **5
conversations of 16 turns**, and a turn is not independent of the
turn before it in the same conversation. The agent settles into a
one-message or a two-message rhythm early and keeps it.

Scored over the committed transcripts, that is exactly what the
data shows. `ends_on_a_question` ran 47%-64% across nine
conversations, a spread *narrower* than binomial noise alone would
predict — its turns really are close to independent.
`one_message_when_dry` ran 42%, 82% and 81% across the three
conversations long enough to mean anything: about three times the
variance the pooled figure assumes. Correcting z ≈ 3.9 for that
puts it near 2, which distinguishes nothing.

So the honest statement is the weaker one: **three invocations of
an unchanged config disagreed, and the disagreement is consistent
with the variation between conversations that the same rule
already shows within a single invocation.** Nothing established a
cause outside the config; the evidence that seemed to demand one
was an artefact of counting replies as if they were draws. Ten
runs looked stable within an invocation and told you nothing about
the next one — that part still holds, and rule 9 below is what it
became.

Now the damage that did before it was noticed, all in one session:

- **A config rule was reverted** on the strength of 46% → 81%
  across two transcripts. The reverted config then measured 81%
  too. The rule was never the cause.
- **A regression was reported** in a capitalisation rule at 19% →
  52%, measured on one conversation. Across all conversations the
  same two runs give 45% → 38% — no regression, opposite sign.
- **A second regression was reported** in a line-length rule at 0%
  → 43%. Over ten runs it is 1%.
- **A committed changelog carried two of these as fact**, and a
  later version had to correct them.

Nobody was careless. Every number was real, correctly computed,
and honestly reported. They were all single samples of a process
whose spread nobody had measured, and there was nothing in the
tooling to say so.

## The rules

1. **Persist the invocation, not just the run — and persist the
   runs with their labels.** The comparable unit is one whole
   invocation of the measurement — every repeat, every script,
   together. Inside it, each run keeps the script it played,
   because a rate stripped of that is a number nothing can
   stratify (see rule 9). `agents7.eval.style_history` is the
   implementation: `record_style_run` builds the record,
   `append_style_run` appends it, `load_style_history` reads it
   back. A new measurement surface gets the same treatment before
   it gets its first published number.

2. **Identify the configuration by hashing it, not by labelling
   it.** A version string is asserted by whoever types it, and a
   config can change without one moving. `fingerprint_run` hashes
   the agent's resolved system prompt together with the contract's
   rules and scripts, and a baseline is built only from records
   sharing it. Editing a persona starts a fresh baseline instead
   of being judged against a band that described a different
   agent.

   **The rule set is identity for the record, not for the band**
   (since agents7 v9.1.0). Comparison keys on `configuration_key`,
   the same hash without the rule ids, and each rule's band holds
   only runs scored under that rule's own `definition` — its check
   and arguments. Adding a rule cannot move another rule's number,
   so it must not spend that rule's baseline; changing what a rule
   checks under the same id is a new measurement, so it must not
   inherit one. Before v9.1.0 both went the wrong way: a chat
   agent's 73-cent baseline was one added rule from unusable, and a check
   could be edited under an unchanged id and judged against the
   old check's band without anything saying so.

3. **Refuse a verdict below two prior invocations.** One earlier
   run is a point, and a point has no spread. `Verdict.NO_BASELINE`
   is the honest answer and it is what the comparison returns. Two
   is a floor, not a recommendation — `MIN_BASELINE_RUNS` says so
   in a comment, and the count is printed beside every verdict so a
   reader can discount it.

4. **Call a change only when the rate leaves the band its own
   history established** — and by more than its own sampling
   margin. Both halves are load-bearing. The band alone flagged
   four stray greetings out of 175 against a band built from one
   and one, which is not a distinguishable difference; the margin
   (`sampling_margin`, twice the spread, floored at half an event)
   is what stops a rule with a handful of events firing on one
   more.

   **And the margin comes from the prior invocations once there are
   four of them** (`MIN_SPREAD_PRIORS`). What is being judged is
   whether the invocation's score moved, so the bar is how much
   that score has moved before; anything measured from inside the
   current invocation is a proxy for it. Measured over a chat
   agent's six gate invocations the proxy ran two to three times too wide
   on four rules — `two_questions_in_a_row` 20% against a real 6% —
   and too narrow on a fifth, so it was wrong in both directions at
   once and no constant could correct it. Below four priors the
   proxy is still what is printed, and `margin_source` says so on
   every line rather than letting it read as the measured figure.
   Above four the proxy is **dropped, not kept as a floor**: taking
   whichever figure is larger would leave every too-wide rule
   exactly as wide as it was.

   **And that spread is read at its upper 90% bound** (since
   agents7 v9.11.0, `SPREAD_CONFIDENCE`). A standard deviation from
   four values carries three degrees of freedom, so four priors
   that happen to agree give a bar of almost nothing: on 15th
   September 2026 a chat agent's gate failed an unchanged config on
   `copies_an_example` at 18.4% against 14.9%-16.5%, with a margin
   of 1.4 points. Replayed over that gate's whole history — 328
   rule checks on four or more priors — the point estimate failed
   an unchanged config six times and the 90% bound never, and the
   median margin went from 9.2 to 16.7 points. That sensitivity is
   given up on purpose: it stays under the 33- and 40-point
   regressions the contract was built for, and the widening fades
   as priors accumulate.

   **A rule broken less than once per invocation is shown, not
   judged** (since agents7 v9.6.0, `Verdict.RARE`). A band built
   mostly from zeros cannot tell a regression from bad luck:
   a chat agent's gate failed `no_trailing_laugh` twice at 3/111 on an
   unchanged config, the laughs in different conversations each
   time. The price was accepted on 13th September 2026: v0.2.9's
   `one_message_when_dry` went from 1, 1, 0 and 0 of 48 to 10 of
   48, and that now passes, visible only in the summary.

5. **Never quote a single invocation as a baseline, in a
   changelog, a README, a backlog item, or a reply.** If only one
   invocation exists, say what it was and say it is one
   invocation. The phrasing that keeps this honest is *"one run
   said X"*, never *"X"*.

6. **A record keeps the budget it was scored under.** A threshold
   tightened later would otherwise make every earlier invocation
   look like a failure it never was.

7. **The history is append-only.** It is the only thing that can
   say how much a figure moves when nothing changed. Rewriting it
   to tidy an inconvenient invocation destroys exactly the
   evidence it exists to hold.

   **The one exception: correcting a key the instrument now
   computes differently.** A fingerprint is not a measurement — it
   is the identity the code assigns to one, and fixing the code
   can change it. When a fingerprint function is corrected, records
   made by the old one are filed under a key nothing will match
   again, and that is a defect in the filing rather than in the
   evidence. Refiling them is allowed, under four conditions, all
   of which must hold and be stated in the commit:

   - **Not one measured number changes.** Only the identity fields.
     Prove it — diff the file with those fields deleted and show it
     is byte-identical.
   - **The old key was demonstrably this configuration.** Show it
     from version control, not from recall: that the config, the
     contract and the code the hash covers were unchanged between
     the run and now. If any of that cannot be shown, the records
     are not this configuration and must not be refiled.
   - **The correction is written where a reader of the data will
     hit it** — the contract header, the changelog entry — saying
     what the old key was, what it is now, and why.
   - **It is a correction, never a merge.** Refiling records so
     they join a band they did not belong to is the thing rule 7
     forbids, wearing this exception as a disguise.

   Taken once, on 3rd September 2026: agents7 v4.37.0 folded the
   guard into `fingerprint_run`, so a chat agent's five guarded
   invocations sat at `cfcfa8d5d117` — which had become the key an
   *unguarded* run of the same config carries. Leaving them was
   not neutral: eleven unguarded records shared that key, and only
   which file they sat in told them apart. They were refiled under
   `f52c25ec29b8`.

8. **Record what else could explain a move.** The model that
   *answered* is on every record (`answered_by`, read from the
   provider's reply), because a provider serving a different model
   explains a shifted number as readily as a config edit does — and
   a band is built only from invocations answered by the same one.
   The configured name is not enough. On 10th September 2026
   DeepSeek moved `deepseek-chat` to a new model under an unchanged
   config, and a chat agent's casing rule went from 75% to 10%. A
   single full stop was nearly credited with the change, until a
   same-day control of the unchanged config moved by the same
   amount. **Before crediting a move to an edit, run the unchanged
   config on the same day.** So
   is `lost_runs`: a conversation that raised is dropped rather
   than allowed to end the invocation, and an invocation that
   measured fewer runs than it planned carries more uncertainty
   than its repeat count implies. When a new confound is
   identified, it goes on the record rather than into someone's
   memory.

9. **Say what the sample is before quoting a sample size, and
   never let the row count stand in for it.** A denominator is
   only a sample size when its rows were drawn independently, and
   a conversation's turns are not: each one answers an agent that
   has read its own previous replies. `one_message_when_dry`'s
   "80 replies" is 5 conversations, and carries about a third of
   the information 80 draws would. Every test, margin and
   confidence claim has to be built on the independent unit — the
   run — or it will report a certainty the data cannot support.
   Where four prior invocations exist `sampling_margin` sidesteps
   the question entirely, taking the spread of those whole scores;
   below that it takes the runs and estimates the error from how
   much they disagreed, falling back to the pooled figure only when
   a record kept none, and marking each fallback as the weaker
   figure. **This rule was written because the file you are
   reading asserted a z-score that made this mistake**, and a
   session then went looking for a cause outside the config on the
   strength of it.

   **And the independent unit is a run of one script, not a run of
   any script.** Two scripts are two stimuli with two different
   true rates, and every invocation replays both in the same
   proportion — so the gap between them is fixed structure, not
   something that varies between invocations. Counting it as noise
   is the same error pointing the other way: it reports a margin
   far wider than the instrument's real precision and makes a gate
   blind to regressions it could see. So a record persists each
   run's label, and the spread is measured within each script and
   pooled across them.

## What this does not claim

**It does not make the underlying variation go away.** Comparing
against a band tells you when a difference is too large to be the
variation you have already seen. It cannot tell you what causes
that variation. On the chat agent above the largest identified source is the
conversation itself — the agent picks a rhythm in the first turns
and holds it — which is enough on its own to produce the
23%/70%/41% spread. That does not rule out a further cause, but it
does mean there is currently **no evidence demanding one**, and an
experiment hunting for a provider effect should be sized against
the between-conversation variation rather than against a pooled
reply count.

**A wide band means the instrument cannot see small changes.** That
is a true and useful thing to know, and it is better stated than
hidden behind a confident single number. A rule whose band spans
23% to 70% cannot detect a ten-point regression, and no amount of
reporting will change that; only reducing the variation will.

## Scope

The `agents7` framework, and every consumer of a figure it produces —
an agent built on it, the UI, a changelog entry, a backlog item, a
reply to the operator. The principle generalises to any
measurement of a stochastic system; the implementation here is this
framework's.

## Related

- `metrics-completeness.md` — the sibling. A declared measurement
  must be produced; this rule adds that a produced measurement must
  be comparable before it is believed.
- The style contract is the machine-scored instrument this rule
  governs; a human reading of a chat agent's conversations is not a
  rate and is not compared this way.
