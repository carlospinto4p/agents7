# Cost, caching and logging

## Cost Estimation

Every response carries an `LLMMetrics`, and its `cost_cents` is `None` until the engine is told what the model costs. Declare `pricing` and every call estimates its own cost.

Rates are **US dollars per million tokens**, which is how providers publish them. Copy the published number straight in:

```yaml
engine:
    provider: openai_compat
    model: deepseek-chat
    pricing:
        input: 0.44          # prompt tokens read fresh
        output: 1.32         # generated tokens
        cached_input: 0.014  # prompt tokens served from the provider's cache
```

Or in Python:

```python
from agents7.llm import LLMEngine
from agents7.llm.pricing import Pricing

engine = LLMEngine(
    "openai_compat",
    "deepseek-chat",
    pricing=Pricing(input=0.44, output=1.32, cached_input=0.014),
)
```

**`cached_input` is the one that matters for a chat agent.** A persona chatter resends its whole system prompt on every turn: persona, scene, style examples, remembered facts. After the first turn nearly all of it is a prompt-cache hit. Measured on a chat agent: 6,144 of 6,239 prompt tokens, 98%. At the rates above a cache hit is **31× cheaper** than a fresh read, so pricing all input at the fresh rate overstates a turn by roughly 20×. `cached_input` defaults to `input`, which is correct for a provider that does not discount repeated prompts.

`cached_prompt_tokens` on the metrics reports how many prompt tokens were cache hits. It is a *subset* of `prompt_tokens`, on every provider. `openai_compat` reads OpenAI's `prompt_tokens_details.cached_tokens` or DeepSeek's `prompt_cache_hit_tokens`, and Anthropic's cache figures are added into `prompt_tokens` so the subset relationship holds there too. A provider that reports nothing gives 0, and the estimate falls back to charging everything fresh.

A rate the provider varies by time of day is not modelled. Declare the rate you want the estimate against. `cost_cents` is an estimate, which is what it is named.

**An unpriced call reports `None`, never `0.0`.** That distinction is load-bearing: a zero cost and a zero duration are what a *missing* measurement looks like, and telling the two apart is the whole point. `tests/test_metrics_completeness.py` enforces it, along with the rule that every provider and every response path fills every declared metric. See `.claude/rules/metrics-completeness.md`.

### Guarded agents: two calls, two costs

A `GuardedChatter` reply is at least two LLM calls, because the chatter writes and the guard reads it and votes. A rejected reply is four. Both halves are reported, and adding them up is the caller's job:

```python
events = []
agent.guard_callback = events.append

resp = agent.chat("hola")

chatter = resp.metrics.cost_cents
guard = sum(e.metrics.cost_cents for e in events if e.metrics)
print(f"turn cost: {chatter + guard:.5f} cents")
```

Two things worth knowing:

- **`resp.metrics` covers every chatter call the turn made**, not just the first. When the guard rejects a reply the chatter is called again, and those retries are added in. Otherwise a rejected turn, which is the expensive one, would under-report.
- **The guard's engine is priced separately.** A guard that declares no `engine` of its own inherits the agent's, pricing included; one that declares its own needs its own `pricing` block, since it may well run a different model.

On one guarded chat agent, where both run the same model, the guard costs roughly as much as the chatter. A guarded agent is therefore about twice the price of an unguarded one, not a rounding error on top.

### Retries

Two different things can make a response take more than one LLM call, and the metrics report them differently because they cost differently.

A **transport retry** is a call that never reached the model: a connection failure, a 500, a 429. `retry.py` waits and tries again. The attempt raises before the provider reports any usage, so it has no tokens and no price. What it cost is time, and that is what is measured:

```python
resp = agent.act({"text": "..."})

resp.metrics.retries          # attempts that failed and were retried
resp.metrics.retry_duration   # seconds they took, backoff included
```

`retry_duration` is **not** part of `total_duration`, which is the successful attempt's own figure. The wall clock a caller waited is the two added together. A cache hit reports `0` for both: no request left the machine, so nothing could fail.

A **parse retry** is different: the call succeeded and the model answered, but the answer did not match the schema, so `max_parse_retries` asks again. That is a whole extra LLM call with its own tokens and its own price, and since v4.26.0 its metrics are added into the response. An agent that needed two retries used to report one call's cost while having paid for three. It is *not* counted in `retries`, which is only for attempts that produced nothing.

A retry always goes back to the **same model on the same provider**. Only a `fallback_model` changes which model answers, and it is never automatic. The next section covers it.

### Which model answered

`metrics.model` names the model that actually produced the response, as the provider named it in its own reply:

```python
resp = agent.act({"text": "..."})
resp.metrics.model      # 'openai_compat::deepseek-v4-flash'
agent.engine.model_id   # 'openai_compat::deepseek-chat'
```

It is a separate field from the configured `engine.model_id` because the two differ in two situations that were both invisible before v5.4.0:

- **A provider resolving an alias.** Ask DeepSeek for `deepseek-chat` and `deepseek-v4-flash` answers. Only the reply knows.
- **A fallback answering.** When the primary fails after its retries and the fallback succeeds, `metrics.model` is the fallback's. That holds under `fallback_model` and under `fallback_engine` alike, so a fallback on another provider names itself too. Every call used to be recorded against the configured name, so a run half-answered by a fallback logged all of it against the model that had failed.

The same split is in the call log: `CallRecord.model_id` is what the agent was configured with, `CallRecord.model` is what answered. A `CallStats` bucket carries `models`, the distinct names its calls were answered by. More than one entry means the bucket was not all served by the same model.

On a response that adds several calls together, `model` holds only when every one of them named the same model, the way `cached` does. A guarded turn's reply and its guard usually run different models, and one field cannot report two. The per-call truth is in the call log, one row per call.

### Adding calls up

Anything that makes several LLM calls for one response adds them with `LLMMetrics.sum`:

```python
from agents7.models import LLMMetrics

total = LLMMetrics.sum([first, second])
```

Counters and durations add. `cost_cents` adds only when at least one part carries it and stays `None` otherwise, so an unpriced response reads as unpriced rather than as free. `cached` holds only when every part was a hit. Use it rather than summing field by field: it is the one place a new metric has to be handled, which is why a guarded turn, a tool loop and a pipeline all agree.

## Caching

Deterministic LLM calls (`temperature=0`) can be cached in memory with an LRU eviction policy. Cached responses are returned instantly without hitting the LLM. When `temperature != 0`, caching is bypassed automatically.

Streaming responses are also cached: on a miss the result is stored after the stream is fully consumed; on a hit the full response is returned as a single chunk.

The default in-memory cache is lost on process restart. For persistent caching, use the SQLite backend:

```python
from agents7.llm import LLMEngine, CacheConfig

engine = LLMEngine(
    provider="ollama",
    model_name="qwen3",
    cache=CacheConfig(backend="sqlite", path="cache.db"),
)
```

```yaml
engine:
    provider: ollama
    model: qwen3
    cache:
        backend: sqlite
        path: cache.db
```

**YAML configuration:**

```yaml
name: MyAgent
version: 1
engine:
    provider: ollama
    model: qwen3
    cache:
        max_size: 128
system_prompt: Analyze this text.
output_model: mymodule.MyOutput
```

All `cache` fields are optional and fall back to defaults. An empty `cache: {}` enables caching with defaults. Omitting the `cache` key entirely disables caching (backward compatible).

**Python configuration:**

```python
from agents7.llm import LLMEngine, CacheConfig

engine = LLMEngine(
    provider="ollama",
    model_name="qwen3",
    cache=CacheConfig(max_size=256),
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_size` | `128` | Maximum cached entries (0 = unlimited) |
| `backend` | `"memory"` | `"memory"` for in-memory LRU, `"sqlite"` for persistent |
| `path` | `"cache.db"` | SQLite file path (ignored for memory backend) |

**What a cache hit reports about itself.** A hit is a real call
with numbers of its own, so it does not replay the stored ones:

- `metrics.cached` is `True`. On a response that adds several
  calls up, such as a `ToolAgent` loop or a `GuardedChatter` turn,
  it is `True` only when every one of them was a hit.
- `metrics.total_duration` is how long the lookup took, typically
  well under a millisecond, not how long the original call took.
- `metrics.cost_cents` is `0.0` on a priced engine, because no
  request left the machine. On an engine with no declared pricing
  it stays `None`, so an unpriced run keeps reading as unpriced
  rather than as free.
- `metrics.tokens` and `metrics.prompt_tokens` are the original
  call's, because they describe the response being returned. `cached` is
  what tells you they were not billed a second time, and
  `CallStats.cached_calls` counts them across a run.

Until v4.24.0 a hit returned the stored duration and cost
verbatim, so any total that included one reported time it did not
take and money it did not spend.

## Structured Call Logging

Optional per-call logging that captures trace ID, agent name, model, tokens, duration, cost, success/failure, and the prompt behind any call that failed. Useful for debugging, observability, and cost tracking.

```python
from agents7.call_log import JsonCallLogger

# Log to file (JSON lines)
logger = JsonCallLogger("calls.jsonl")
agent = Labeler(..., call_logger=logger)

# Log to stdout
logger = JsonCallLogger()
agent = Labeler(..., call_logger=logger)

# Custom logger — any callable accepting CallRecord
agent = Labeler(..., call_logger=lambda r: print(r.to_dict()))
```

When logging to a file, `JsonCallLogger` holds a single append
handle (opened lazily on first write, flushed per record) instead
of reopening the file on every call. Call `close()` or use it as a
context manager to release the handle:

```python
with JsonCallLogger("calls.jsonl") as logger:
    agent = Labeler(..., call_logger=logger)
    ...
```

**One `CallRecord` is emitted per LLM call**, so the log counts the
calls actually made. An `act()`, `act_async()`, `act_stream()` or
`act_stream_async()` that parsed first time made one call and emits
one record. An `act()` or `act_async()` that needed **parse retries**
(`max_parse_retries`) emits one record per attempt: a retry is a
whole extra call to the provider, with its own tokens, its own price
and its own messages. They share the answer's `trace_id`, and only the
last one can be a success, because the retry loop stops at the
first call that parses. Each failed attempt carries the messages
*it* sent, which include the parse error fed back to the model. Batch runs
(`act_in_batch()` / `act_in_batch_async()`) emit one record per **LLM
request**. That is one per chunk, not one per input item.

Each record reports its own call, never a total over several, so
summing a log never counts the same tokens twice. The
`AgentResponse.metrics` returned to the caller *is* the total across
every call the answer took.

A **streamed** call emits its record once the caller has finished
iterating the stream, not when `act_stream()` returns: whether the
call succeeded depends on parsing the streamed text, which is not
known before then. A caller that abandons a stream without reading
it records nothing.

**A chat turn emits one record per LLM call it makes.** `Chatter`
takes a `call_logger` like any other agent. A plain turn is one
call. A `GuardedChatter` turn is at least two: the reply and the
guard's check. Each regeneration a rejection forces adds another
pair, so a turn rejected once is four records. The guard's
calls get their own records, carrying the guard's name, version
and model: a guard usually runs a different model from the
chatter, and folding its tokens into the reply's record would
report one call made on two models.

Every call in one turn shares that turn's `trace_id`, so the
records of a single reply can be grouped and added up. A
structured-output agent that parsed first time made one call, so
that id is unique to its record; one that needed parse retries
shares it across the attempts.

**Above the turn sits `run_id`**, shared by every call of one
conversation or one eval run. It is what a log needs to answer
*what did this conversation cost*, which grouping by turn cannot. A
`Chatter` fills it from its `conversation_id` automatically.
Every other agent takes it from the caller:

```python
agent = Labeler(..., call_logger=logger, run_id="sweep-2026-08-30")
```

`run_evaluation_suite()` sets one for you, so each eval run's
calls group even when several runs share a log file. Left unset it
is `null`, which means nobody named a run. It does not mean that
every call belongs to one.

```python
from agents7.call_log import JsonCallLogger

with JsonCallLogger("chat.jsonl") as logger:
    agent = GuardedChatter(..., call_logger=logger)
    agent.chat("hola")
    # -> 2 lines: the reply, then the guard, same trace_id
```

```json
{
  "trace_id": "a1b2c3d4e5f6",
  "agent_name": "Extractor",
  "agent_version": 1,
  "model_id": "ollama::qwen3",
  "success": true,
  "tokens": 42,
  "prompt_tokens": 128,
  "cached_prompt_tokens": 0,
  "total_duration": 1.23,
  "cost_cents": 0.0124,
  "cached": false,
  "retries": 0,
  "retry_duration": 0.0,
  "error": null,
  "prompt": null,
  "run_id": "sweep-2026-08-30",
  "recorded_at": "2026-08-30T09:14:02.4813+00:00"
}
```

`recorded_at` is stamped by the producer when the record is
built, immediately after the call returned. A log without it can
be read in order but not against a date, so it cannot say what a
week cost or whether a model got slower. A row read back from a
log written before this field existed reads as `null`: unstamped,
rather than having happened at the moment it was read.

**The prompt is recorded on a call that failed, and only then.**
`prompt` holds the messages that were sent, as a JSON string, so a
row that failed to parse can be reproduced by hand. A re-run at
temperature would produce a different call, so a prompt not
captured during the run is gone. A call that succeeded records
`null`: its prompt is derivable from the agent's config, and
carrying it on every row would multiply the log by the size of the
system prompt.

```json
{
  "trace_id": "9f8e7d6c5b4a",
  "agent_name": "Extractor",
  "success": false,
  "error": "ParseError: expecting value at line 1 column 1",
  "prompt": "[{\"role\": \"system\", \"content\": \"Extract ...\"}, {\"role\": \"user\", \"content\": \"Process the following:\\n{'text': 'hello'}\"}]"
}
```

Batch runs record the chunk's messages, not one item's.
`ToolAgent` records no prompt: it reports no per-call failure the
field could key off yet. Nor does a chatter's own reply, which has
nothing to parse and so cannot fail this way. Its **guard** can,
though, and an unparseable verdict records the guard input that
produced it.

**Two sinks: a file or a database.**

`JsonCallLogger` appends JSON lines. It suits one eval run and
not months of chat turns. Reading it means parsing the whole
file, and two processes appending at once interleave.
`SqliteCallLogger` writes one row per call to a `calls` table
whose columns are `CallRecord`'s fields, indexed on `run_id`,
`trace_id` and `recorded_at`, in WAL mode so the log can be
queried while a run is still writing to it:

```python
from agents7.call_log import SqliteCallLogger

with SqliteCallLogger("data/calls.db") as logger:
    agent = Chatter(..., call_logger=logger)
```

Neither creates its file until the first record is written, so a
run that logged nothing leaves nothing behind.

**YAML configuration:**

```yaml
call_log:
  backend: sqlite        # json (default) | sqlite
  path: data/calls.db    # relative paths sit beside this config
```

A relative `path` is resolved against the directory holding the
config, not the working directory. That way the same agent logs to the
same file whether it is loaded by a script or by a web backend.
Written bare (`call_log:`) it defaults to `calls.jsonl` beside the
config.

The older form still works and is for a sink you built yourself:

```yaml
call_logger: mymodule.my_logger
```

The `call_logger` value is a dotted import path to any callable
accepting a `CallRecord`. It is an *alternative* to `call_log`,
not an addition. Declaring both raises
`AgentConfigurationError`, since only one sink can be attached
and picking one silently would leave a config that reads as
logging to a database writing somewhere else.

**Reading a log back:**

```python
from agents7.call_log import read_call_log

for record in read_call_log("data/calls.db"):
    print(record.run_id, record.cost_cents)
```

`read_call_log()` and `summarize_call_log()` both accept either
sink's output, told apart by the file's own header rather than by
its name.

**Analyzing a call log:**

`summarize_call_log()` reads a JSONL file written by
`JsonCallLogger` (or any iterable of `CallRecord` / dict / JSON
line) and returns a `CallLogSummary` with overall stats plus
per-agent, per-model and per-run breakdowns and error counts:

```python
from agents7.call_log import summarize_call_log

summary = summarize_call_log("calls.jsonl")
print(summary.format())

# Programmatic access:
summary.overall.success_rate   # fraction of calls that parsed
summary.overall.tokens         # total completion tokens
summary.by_agent["Extractor"]  # CallStats for one agent
summary.by_model["ollama::qwen3"].avg_duration
summary.by_run["cold-read"].cost_cents  # what one run cost
summary.errors                 # {error message: count}
summary.overall.cost_cents     # None when nothing was priced
summary.overall.cached_calls   # how many came from the cache
summary.overall.retries        # attempts that failed and were retried
summary.overall.retried_calls  # how many calls needed one
summary.overall.retry_duration # seconds those attempts took
```

`cost_cents` sums only the records that carried one and stays
`None` when none did, so a run on an engine with no declared
pricing reads as *unpriced* rather than as free. `cached_calls`
counts the calls served from the response cache: their tokens are
in the totals above but were never billed a second time, so a run
with many hits has spent far less than its token totals suggest.
`by_run` holds only the records that named a run, so a log written
before anyone did comes back empty rather than as one run called
`None`.

`retries` and `retry_duration` are what a slow run needs to
explain itself. A run that hit rate limits and retried its way
through used to be indistinguishable in the log from one that
succeeded first time: same records, same tokens, same cost, and
nothing to account for the wall clock. `retried_calls` separates
one call retried five times from five calls retried once.

During an evaluation you do not need to wire this up by hand. Pass
`call_log=` to [`run_eval_suite`](evaluation.md#evaluation-workflow) and the route
attaches the logger, summarises the result, and folds it into the
report's *Call Log* section.

## Batch Concurrency Control

When using `act_in_batch_async`, you can limit the number of concurrent LLM requests via a `BatchSettings` instance. This prevents overwhelming the LLM server with large input lists.

```python
from agents7.settings import BatchSettings

# Labeler: limit to 3 concurrent chunk requests
responses = await agent.act_in_batch_async(
    inputs,
    batch_settings=BatchSettings(max_concurrent=3),
)

# Pipeline: limit to 2 concurrent requests per step
results = await pipeline.act_in_batch_async(
    inputs,
    batch_settings=BatchSettings(max_concurrent=2),
)

# Unlimited concurrency
results = await pipeline.act_in_batch_async(
    inputs,
    batch_settings=BatchSettings(max_concurrent=None),
)
```

| `batch_settings` | `max_concurrent` | Behavior |
|-------------------|------------------|----------|
| `None` (default) | `5` | Sensible throttling out of the box |
| `BatchSettings(max_concurrent=None)` | `None` | Unlimited concurrency |
| `BatchSettings(max_concurrent=N)` | Positive `int` | Caps concurrent LLM requests via `asyncio.Semaphore` |

The default `max_concurrent` value can be overridden via the `BATCH_MAX_CONCURRENT` environment variable.

The semaphore is created per call (not stored on the instance), so different calls can use different limits.

## Batch Export

Export agent responses to JSONL or CSV files after `act_in_batch`:

```python
from agents7.eval.export import export_jsonl, export_csv

responses = agent.act_in_batch(inputs)

# Export to JSONL (one JSON object per line)
export_jsonl(responses, "results.jsonl")

# Export to CSV (auto-detects model fields)
export_csv(responses, "results.csv")

# Export specific fields only
export_csv(responses, "results.csv", fields=["category", "reason"])

# Export to Polars DataFrame (requires polars)
from agents7.eval.export import export_dataframe

df = export_dataframe(responses)
df = export_dataframe(responses, fields=["category", "reason"])
```
