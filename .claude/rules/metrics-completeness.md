---
# Metrics guard: only when framework code or a surface that renders a measurement is open.
paths:
  - "src/agents7/**"
  - "tests/test_metrics_completeness.py"
  - "tests/test_metrics.py"
---

# A Declared Measurement Must Be Produced

Project law for the `agents7` framework. **Every field the framework
declares for observability — tokens, durations, costs, models,
trace ids — must be filled by every code path that could fill it.
A field that exists and is never populated is worse than one that
does not exist**, because the value it falls back to looks like a
measurement.

Set 29th August 2026 at the operator's instruction, after
`cost_cents` was found to have been declared on every response and
populated by nothing, for as long as it had existed.

## The failure class, precisely

Not "we forgot to measure something". The framework *was* designed
for this: `LLMMetrics` has carried `cost_cents` from early on,
`LLMEngine` accepted a rate, and the dispatcher had a working cost
calculation. Every piece existed. **The one line that connects
them — `build_engine` passing a rate through — was never
written**, so a config-driven agent could not produce a cost, and
the field returned `None` on every call the framework ever made.

The shape to recognise:

- A **declaration** in one file (a model field, a dataclass
  attribute, a documented dict key).
- **Producers** in other files — one per provider, one per
  response path, one per wrapper.
- **Nothing that checks the second against the first.**

Add a provider, add a streaming path, add a guard wrapper, and the
new code omits a field. Nothing fails.

## Why nothing caught it, and why more care would not have

Three reasons, and none is fixed by being more careful:

1. **Every default is a plausible value.** `cost_cents: float |
   None = None` reads as "this model is free". `total_duration:
   float = 0.0` reads as "the call was instant". `cached_prompt_
   tokens: int = 0` reads as "nothing was cached". A missing
   measurement is indistinguishable from a real one, on screen and
   in a test.
2. **The type checker is satisfied.** The field has a default, so
   omitting it is legal everywhere. There is no unfilled-field
   error to see.
3. **The tests asserted presence, not completeness.** Every test
   checked the fields it happened to care about. Not one asked
   *"does every producer fill every field?"* — so the suite was
   green while half the surface was empty.

The result is a bug with no symptom. It surfaces only when a
person asks a question the data should have answered, which here
took months.

## The nine instances found

All in one framework, all the same class. Listed because the
number is the argument: this is not a one-off.

| # | What was declared | What produced nothing |
|---|---|---|
| 1 | `cost_cents` | `build_engine` never passed a rate — no config-driven agent could cost anything |
| 2 | `total_duration` | `openai_compat`, plain path, returned a hard `0.0` |
| 3 | `total_duration` | `openai_compat`, streamed path, returned a hard `0` |
| 4 | `cost_cents` | the streaming path never called cost injection at all |
| 5 | `cached_prompt_tokens` | the streaming path omitted it |
| 6 | the guard's own call | `_guard_check` had the metrics and returned only the duration |
| 7 | a retried reply's calls | the guard retry loop discarded every regenerated call's metrics |
| 8 | `cost_cents`, `cached_prompt_tokens` | `CallRecord` and `CallStats` had no such fields, so no trace could report cost |
| 9 | `cached_prompt_tokens` | `ollama` left it to the model default rather than stating it |

Numbers 2 and 3 are the same bug in two paths, fixed nine days
apart, because fixing the first never prompted anyone to look at
the second. That is the class in miniature.

## The mechanism — `tests/test_metrics_completeness.py`

**This rule is enforced by a test, not by intention.** The guard
reads the declaration and checks the producers against it, so it
fails when something new is added and not wired up.

What it enforces:

- **Every field on `LLMMetrics` is classified** as produced by a
  provider or added by the dispatcher. A new field belongs to
  nobody until it is classified, and that fails.
- **Every provider fills every field it owns** — `ollama.chat`,
  `ollama.generate`, `openai_compat`, `anthropic`.
- **No provider reports a zero duration**, because a zero here has
  only ever meant "not measured".
- **The streamed path reports the same field set as the plain
  one**, which is what instances 3, 4 and 5 all violated.
- **`CallRecord` carries every field `LLMMetrics` carries**, so
  the trace surface cannot fall behind the measurement surface.
- **A cached token count never exceeds the prompt total**, an
  invariant the cost calculation depends on and that differs by
  provider.

**When it fails, fix the producer — never the guard.** Widening
the exemption list to make it pass reintroduces exactly the bug it
exists to catch. An exemption needs a comment saying which other
layer fills that field, and a test proving that layer does.

## The rules

1. **A new measurement is added to the guard first.** Write the
   field, watch the completeness test fail for every producer,
   then fix them. Adding the field and the producers together and
   then the test never proves the test works.
2. **State the measurement; never let a default stand in for it.**
   Ollama bills nothing and caches no prompts, so its cached count
   is genuinely `0` — and it says `0` explicitly, because
   "measured as zero" and "never measured" must not look the same
   in the source.
3. **Distinguish "not measured" from "measured as zero" in the
   type.** `None` means nobody measured it; a number means
   somebody did. This is why `cost_cents` is `float | None` and
   why an unpriced run reports `None` rather than `0.0` all the
   way out to the screen.
4. **A new response path reports what the plain path reports.**
   Streaming, batching, retries, tool loops, guarded wrappers — a
   caller must not have to know which path produced a number to
   know whether it is real.
5. **A wrapper that makes extra LLM calls reports them.** A
   guarded reply is two calls and a rejected one is four. Both
   the retries and the guard's own calls are measured, because
   the expensive turns are precisely the ones that were
   under-reported.
6. **Downstream surfaces mirror the measurement surface — its
   shape as well as its fields.** The call log, an API payload, a
   UI line: each carries every field, or says plainly why not.
   `CallRecord` silently carrying three of five fields is how an
   eval run lost the ability to report its own cost. And **one
   row means one call**: a row that carries every field while
   standing for three calls satisfies every field check and still
   reports a false count, which is how `CallStats.calls` said one
   where a `Labeler` had made three. Where a row aggregates, it
   carries the id the rows of one answer share, so a consumer can
   group them itself rather than being handed a total it cannot
   take apart.
7. **Never render an absent measurement as a number.** `$0.00`
   for an unpriced call and `0.0s` for an untimed one are the
   on-screen form of this same bug. Show "unpriced", or show
   nothing.

## What must be measured

The standing surface, so that "we should have metrics for
everything" is a checklist rather than an aspiration. Every LLM
call carries:

- **Tokens** — prompt, completion, and how many prompt tokens the
  provider served from its own cache.
- **Duration** — wall clock when the provider reports none, the
  provider's own figure when it does.
- **Cost** — estimated from the engine's declared rates, `None`
  when it declares none.
- **Whether it was served from the response cache** — a hit is a
  real call with its own numbers, not a replay of the stored
  ones: the lookup's duration, a cost of zero where the engine
  can price at all, and `cached` set, which is what tells a
  consumer the token counts it kept were not billed again. On a
  response that adds several calls up, `cached` holds only when
  every one of them was a hit.
- **What it took to get an answer at all** — `retries`, the
  attempts that failed and were repeated, and `retry_duration`,
  the seconds they and their backoff took. A failed attempt
  raises before the provider reports any usage, so it has no
  tokens and no price; time is the whole of what it cost, and
  saying nothing left a run that fought rate limits for minutes
  looking identical to one that sailed through. It counts
  retries rather than attempts so that it still adds up on a
  response made of several calls, where a total attempt count
  could not be told apart from several clean ones.
- **Model** — the resolved `provider::model` the call actually
  used, which is not always the one configured. `deepseek-chat`
  is an alias the API resolves to `deepseek-v4-flash`, and a
  `fallback_model` answers under its own name after the primary
  failed. Read it from the provider's reply, never from the
  engine's configuration: the configured name is what was asked
  for, and asking is not what is being measured. Both are kept —
  `CallRecord.model_id` is the configured one, `CallRecord.model`
  what answered — because the interesting fact is the two
  differing.
- **Trace id, agent name, agent version, success, error** — on
  every call log record.
- **The prompt, on a call that failed** — the messages that were
  sent, as a JSON string on `CallRecord.prompt`. See below.

### The prompt is recorded on failure, and only on failure

Decided 29th August 2026. This is the one field on `CallRecord`
that is deliberately not on every record, so it is stated here
rather than left to look like the omission it would otherwise
resemble.

- **A failed call records the messages it sent.** That is the
  evidence needed to reproduce the row by hand, and it cannot be
  recovered afterwards: a re-run at temperature produces a
  different call, which is the same lesson
  `chat-agent-validation.md` learned when a read's token counts
  were gone the moment the run ended.
- **A successful call records none.** Its prompt is derivable
  from the agent's config, and the cost of carrying it is not
  small — a chat agent resends its whole persona every turn, so
  on a chat agent the text is around 6,200 tokens against roughly
  ten generated. Recording it per row would multiply the log by
  about a hundred to one and put the other person's messages in
  it.
- **Call logging is opt-in already**, so no run writes prompt
  text unless a caller asked for a call log in the first place.
- **A producer that cannot tell a failure from a success records
  nothing**, rather than guessing. `ToolAgent` hardcodes
  `success=True` per turn, so it carries no prompt today. A
  chatter's own reply is the same case — there is nothing to
  parse, so a call that returned is a success. Its **guard** is
  not: an unparseable verdict is a real failure, and since
  v4.25.0 that record carries the guard input that produced it.

### Known gaps, tracked in `backlog.md`

Named here so they are not rediscovered as surprises. **None are
open as of v9.0.0** — the list below is what has been closed.
When one is found, it goes here with a backlog item behind it,
and moves down once it ships.

Closed since this rule was written, kept here because each was an
instance of the class rather than a one-off bug:

- **A streamed call never reached the call log** — instance 10,
  closed in v4.24.0. `act_stream` returned the stream and emitted
  no `CallRecord`, so an all-streamed run wrote an empty log. The
  record now comes from the stream result, once iteration has
  parsed the text and the outcome is known.
- **A cache hit replayed the original call's numbers** — closed
  in v4.24.0. It returned the stored duration and cost, so a
  cached call reported time it did not take and money it did not
  spend. A hit now reports the lookup's own duration, a cost of
  zero on a priced engine (`None` on an unpriced one), and
  `LLMMetrics.cached`, which is what says its token counts were
  not billed again.
- **A chat turn never reached the call log** — instances 11 and
  12, closed in v4.25.0. `Chatter` carried no `call_logger` at
  all, so every conversational agent was absent from the log; and
  the guard's metrics reached the `GuardEvent` and stopped there,
  so even a log that existed would have excluded them. On
  a chat agent that is 46% of what a turn costs. A turn now emits
  one record per LLM call it makes — the reply, the guard's
  check, and both again for each regeneration a rejection forced
  — and they share the turn's `trace_id` so the calls of one
  reply can be grouped. The guard's record carries the guard's
  own name, version and model rather than being folded into the
  reply's, because a guard usually runs a different model and one
  record cannot honestly report two.
- **The UI rendered a cache hit as a fast free call** — closed in
  `agents7_ui` v0.35.0. Its payloads carried tokens, duration and
  cost but not `cached`, so a cached turn looked like a real call
  that took no time and cost nothing, with its token counts
  reading as billed a second time. The Chatter and Runner payloads
  now carry `cached` (and the Runner's carry
  `cached_prompt_tokens`, which three of its six hand-built dicts
  had dropped), and the metrics line ends in `cached` — or `reply
  cached` when the guard really ran.
- **Retries were invisible everywhere** — instances 13 and 14,
  closed in v4.26.0 and `agents7_ui` v0.36.0. Two mechanisms,
  both silent, and they needed different answers because they
  cost differently. A **transport retry** raises before the
  provider reports any usage, so it has no tokens and no price:
  `LLMMetrics.retries` and `retry_duration` now say that it
  happened and what it cost in time, which is all there is to
  measure. A **parse retry** is a whole extra LLM call that
  succeeded and merely answered off-schema, and its metrics were
  discarded outright — so a `Labeler` that needed two retries
  reported one call's tokens and cost while having paid for
  three. Those are now added into the response. The rule's own
  wording had assumed one mechanism; finding the second while
  fixing the first is why *a wrapper that makes extra LLM calls
  reports them* is worth re-reading against each new wrapper.
- **A parse retry was not its own row in the call log** —
  instance 15, closed in v4.27.0, and the second half of the one
  above. Adding the retries' cost into the response made the
  money right and left the count wrong: all three calls were
  written as a single `CallRecord`, so `CallStats.calls` said one
  where three were made, and a per-model average duration divided
  three calls' seconds by one. Each attempt is now its own
  record, sharing the answer's `trace_id`, carrying only its own
  figures, and — when it failed — the messages *it* sent, which
  are not the first call's: a retry appends the parse error the
  model has to correct. Only the last attempt can be a success,
  since the retry loop stops at the first call that parses. Worth
  noting how it was missed: rule 6 says a downstream surface
  mirrors the measurement surface, and it was read as being about
  *fields*. A row that carries every field and stands for three
  calls satisfies the letter of it and reports a false count, so
  a surface mirrors the **shape** as well — one row per call.
- **Nothing ever said which model answered** — instance 16,
  closed in v5.4.0. The one item on the standing surface above
  that had no field at all: `LLMMetrics` declared no `model`, no
  provider read the name its own API echoed back, and every
  `CallRecord` was written with `model_id=engine.model_id`. So
  both cases where the answering model differs from the
  configured one were invisible — a provider resolving an alias,
  and a `fallback_model` answering after the primary failed, the
  second of which logged a run half-served by the fallback
  entirely against the model that had failed. All four parsers
  and both streamed paths now report `metrics.model` from the
  reply, `CallRecord` carries it beside the configured
  `model_id`, and `CallStats.models` collects the distinct names
  a bucket was answered by.
- **A call that raised left no record at all** — instance 18,
  closed in v9.0.2. Every chat producer built its `CallRecord`
  from the metrics a *returned* call carried, so a call that
  raised had nothing to build one from and wrote nothing. A run
  could lose conversations to `Connection error.` while its call
  log showed no failed call — the style record's `lost_runs` was
  the only sign.
  A second loss hid inside the first: the retry count lived in a
  local `AttemptStats` and died with the raise, so even a logged
  failure could only have said zero retries. The dispatcher now
  attaches the count to the exception, and every chat, guard and
  stream path writes a failed record, with the cost left `None`
  because nothing was measured. The rule this adds to 6: **a
  surface that records calls records the ones that failed** —
  the log must not read cleaner than the run was.
- **The style history recorded the configured model, and pooled
  two models into one band** — instance 17, closed in v9.0.0.
  `record_style_run` wrote `agent.engine.model_id` into a field
  documented as the resolved model, one layer further out than
  instance 16 and missed by the same reading: the call log had the
  answering model and the record did not ask it. It surfaced as a
  measurement error, not a missing field. DeepSeek moved
  `deepseek-chat` from `deepseek-v4-flash` to `deepseek-flash` on
  10th September 2026. Twelve gate invocations of a chat agent then shared
  one key across both models, with a casing rule at 75% on one and
  10% on the other, so the gate's band would pass nearly anything.
  `StyleRunRecord.answered_by` now reads the models from the calls,
  and `compare_to_history` bands only same-model invocations and
  names the ones it left out.

  Two lessons beyond the field itself. **The guard checked field
  *presence*, and a key set to `None` is present** — every
  provider passed the existing completeness test while naming
  nothing, so a value check was needed as well. And **a stored
  surface needs a migration, not just a column**: `CREATE TABLE
  IF NOT EXISTS` left every existing log at its old schema, so
  the first write to one failed outright and the first read of
  one failed too. Adding a measurement is now a schema change,
  and `_add_missing_columns` handles it generally rather than
  once, because there will be a next field.
- **The guard's half of instance 17** — instance 19, closed in
  v9.3.0. v9.0.0 read the answering model for the chatter's calls
  only, so `guard_model_id` kept the configured `deepseek-chat` on
  every guarded record of a chat agent while `calls.db` showed the guard
  answered by `deepseek-flash`. `StyleRunRecord.guard_answered_by`
  now reads it from the guard's calls, and `compare_to_history`
  bands on it as it does on `answered_by`. Found by the same fix
  that closed 17, and left open by it: a wrapper that makes extra
  LLM calls reports them (rule 5), and that includes which model
  answered them.
- **Remembering a conversation never reached the call log** —
  instance 20, closed in v9.7.2. `_build_character_facts` built
  the `FactExtractor` with no `call_logger`, so every `remember()`
  call was absent from the log: 22 extractor calls on a chat agent on
  14th September 2026 left no row. The part that mattered was the
  failure: an unparseable extraction returns no facts, which looks
  exactly like a conversation that revealed nothing. `Chatter` now
  hands its own logger to an extractor that has none, **at call
  time**, not in the loader — the style contract swaps the logger
  onto an agent that already exists, so wiring it at load would
  have fixed the config path and left the eval path silent. Rule
  5 again, for a call made outside a turn: an agent a chatter owns
  reports to the chatter's log.

## Scope

The `agents7` framework, and any code that consumes its metrics —
the UI's API payloads and rendered lines included. The principle
generalises, but the guard is this framework's.

## Related

- `testing.md` — the suite this guard lives in.
- `committing.md` — a bug fix ships with a regression test; here
  the regression test is structural and catches the *next*
  instance rather than only this one.
- `measurement-needs-history.md` — a produced measurement must also
  be comparable with earlier runs before it is believed.
