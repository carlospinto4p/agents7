# Evaluation

## Evaluation Workflow

`run_eval_suite()` is the composed route from a golden set to a report. It is one call that loads and normalises the eval data, builds the agent from its YAML config, scores every row, aggregates metrics, summarises the calls it made, and writes a Markdown report. Reach for this first; the primitives in the next section are for when you need finer control.

```python
from agents7.eval import run_eval_suite

run = run_eval_suite(
    "configs/sentiment.yml",       # the agent's YAML config
    "data/golden_set_40.jsonl",    # .jsonl, .json, or .csv
    report="reports/sentiment.md",         # optional Markdown report
    call_log="reports/sentiment_calls.jsonl",  # optional call log
)

print(run.format())                # human-readable summary block
print(run.metrics["accuracy"])     # 0.925
print(run.call_log.overall.tokens) # completion tokens for the run
```

The returned `EvalRun` carries everything the run produced: `agent_name`, `model_id`, `data_path`, the scored `fields`, the per-row `results`, the aggregate `metrics`, the `call_log` summary (when requested), and `report_path`.

**What the route resolves for you:**

- **Expected fields** are derived from the golden set intersected with the config's `output_model` fields, so input columns are never mistaken for expected outputs. Pass `fields=[...]` to override.
- **Row shape**: both nested rows (`{"input": {...}, "category": "..."}`) and flat rows (`{"text": "...", "category": "..."}`) are normalised automatically.
- **The confusion matrix** defaults to `category` when that field is scored, otherwise the first scored field. Override with `confusion_field=`.
- **The call log** is written by a `JsonCallLogger` attached to a clone of the agent (the agent you passed is never mutated), then summarised into the report's *Call Log* section. It holds one record per real LLM call, so a batched run reports the number of requests, not the number of rows.
- **Report notes** always cover the eval set and the accuracy. Pass `notes=[...]` to append domain caveats a reader needs to interpret the numbers (e.g. a category the output model deliberately disables).

The same route backs the CLI, so `agents7 eval` and a Python caller produce identical results:

```bash
agents7 eval -c configs/sentiment.yml -d data/golden_set_40.jsonl \
    --report reports/sentiment.md --call-log reports/sentiment_calls.jsonl
```

## Evaluating Tool Agents

A `ToolAgent` answers in free text, so there is no label to compare
against. Two correct answers to the same question rarely share a
wording. What *is* checkable is the **trajectory** it took: which
tools it called, in what order, and with what arguments. Set
`agent_type: tool` and `run_eval_suite` routes the run through the
trajectory scorer automatically. The call log, the metrics and the
report are unchanged.

A golden-set row pairs an input with the calls the agent should make:

```json
{"input": "Can I order 3 blue mugs, and what is the total?",
 "tools": [
   {"name": "find_product"},
   {"name": "check_stock", "arguments": {"sku": "MUG-01"}},
   {"name": "order_total", "arguments": {"unit_price_cents": 1250,
                                         "quantity": 3}}
 ]}
```

```python
run = run_eval_suite("config.yml", "eval_data.jsonl")

print(run.metrics["accuracy"])                    # rows fully matched
print(run.metrics["trajectory"]["extra_calls"])   # calls not expected
print(run.metrics["trajectory"]["per_tool"])      # recall per tool
```

**Matching rules:**

- **Order matters.** Expected calls must appear in the expected
  relative order.
- **`in_order` (the default)** treats the expected calls as an ordered
  *subsequence*: extra calls are tolerated, but counted and reported.
  **`exact`** additionally requires that no extra call was made. Set it
  per run with `trajectory_mode=` / `--trajectory-mode`, or per agent
  with a `trajectory_mode:` key in the config.
- **Arguments match on the stated keys only.** Pin the ones that
  matter (`sku`, `quantity`) and leave the rest out rather than
  spelling out every default. Omitting `arguments` scores the call on
  its name alone, which is the right choice for an argument the model
  legitimately phrases differently (a search `query`).
- **`defaults` pins an argument as absent, or equal to a value.**
  For example, `{"name": "order_total", "arguments": {"quantity": 4},
  "defaults": {"express": false}}` passes a model that omits
  `express` and fails one that invents `express=true`. Neither of the
  other two options can express that: left out entirely, an invented
  argument is ignored and the row passes; pinned under `arguments`,
  a model that correctly omits it fails.
- **A row may expect no calls at all** (`"tools": []`). That is how
  you assert an agent *doesn't* reach for a tool it doesn't need. Such
  a row is scored under `exact` whatever mode the run is in: with no
  expectation to miss, `in_order` would tolerate every call as merely
  extra and the row could never fail.
- **A call the tool rejected satisfies nothing.** A row naming
  `find_product` asserts the agent looked the product up, so a call
  whose arguments the tool threw out leaves that expectation
  unmatched; it is reported among the extras, marked `[failed]`, and
  counted in `run.metrics["trajectory"]["failed_calls"]`. This is the
  case pinning arguments cannot cover. A row leaves the search
  `query` unpinned precisely because its wording is the model's to
  choose, which is where a malformed value would otherwise slip
  through on the tool name alone.
- **`allow_failure` is for a row that tests the error path.**
  `{"name": "check_stock", "arguments": {"sku": "NOPE-99"},
  "allow_failure": true}` is a row where the tool is *meant* to
  raise and the agent is meant to relay it, so the call counts as
  made either way. Leave the key off everywhere else; it is the one
  opt-out of the rule above.

Alongside the usual accuracy, the report gains a **Trajectory** section
with per-tool recall, and each failing case names exactly which calls
were missing and which were extra. The confusion matrix is omitted, because a
sequence of calls is not a label to cross-tabulate.

The scoring primitives are public for finer control:

```python
from agents7.eval import (
    ExpectedCall, ToolCall, collect_trajectory, match_trajectory,
)

calls, response = collect_trajectory(agent, "Any blue mugs?")
result = match_trajectory([ExpectedCall("check_stock", {"sku": "MUG-01"})],
                          calls)
print(result.passed, result.missing, result.extra)
```

## Evaluation Metrics

The `agents7.metrics` module provides headless evaluation primitives. Run an agent against labeled data and compute accuracy, per-field metrics, and confusion matrices.

`run_evaluation` uses `act_in_batch()` internally for efficient processing. If the batch fails, it falls back to sequential `act()` calls so individual errors don't break the entire evaluation. An async variant `run_evaluation_async` is also available.

```python
from agents7.eval.metrics import (
    EvalResult,
    run_evaluation,
    run_evaluation_async,
    compute_metrics,
    build_confusion_matrix,
)

# Labeled evaluation data
eval_data = [
    {"input": {"text": "Great product!"}, "category": "POSITIVE"},
    {"input": {"text": "Terrible service."}, "category": "NEGATIVE"},
]

# Run agent against all rows (uses batch processing)
results = run_evaluation(agent, eval_data, ["category"])

# Or use async for concurrent processing
results = await run_evaluation_async(agent, eval_data, ["category"])

# Compute aggregate metrics
metrics = compute_metrics(results, ["category"])
print(f"Accuracy: {metrics['accuracy']:.0%}")
print(f"Per-field: {metrics['per_field']}")

# Build confusion matrix
cm = build_confusion_matrix(results, "category")
print(f"Labels: {cm['labels']}")
print(f"Matrix: {cm['matrix']}")
```

**A/B testing:** Compare two agents on the same test cases and get side-by-side metrics. Deltas are reported as *A minus B*, so a positive `accuracy_delta` means agent A scored higher.

```python
from agents7.eval import run_ab_test  # run_ab_test_async also available

result = run_ab_test(
    agent_a,
    agent_b,
    eval_data,          # same rows as run_evaluation
    ["category"],       # optional — inferred from the rows when omitted
    label_a="qwen",
    label_b="llama",
)

print(f"Winner: {result.winner}")               # "qwen", "llama", or "tie"
print(f"Accuracy delta: {result.accuracy_delta:+.0%}")
print(f"Per-field delta: {result.per_field_delta}")
print(f"Avg time delta: {result.avg_time_delta:+.2f}s")
print(f"Agreement: {result.agreement}")         # both_passed, a_only, ...
```

`ABTestResult` also carries the full `metrics_a` / `metrics_b` dicts and the per-case `results_a` / `results_b` lists for deeper inspection.

**Evaluation reports:** Generate a formatted Markdown report from evaluation results:

```python
from agents7.eval.export import export_eval_report

export_eval_report(
    results,
    metrics,
    "report.md",
    agent_name="SentimentClassifier",
    model_id="ollama::qwen3",
    confusion_field="category",  # optional
)
```

The report includes: agent/model info, date, overall accuracy, per-field breakdown, optional confusion matrix, and per-case detail with pass/fail status.

## Testing Utilities

The `agents7.testing` module provides helpers for unit-testing agents without LLM calls:

```python
from agents7.testing import create_mock_engine, assert_accuracy_above

# Mock engine with predefined responses
engine = create_mock_engine({
    "good": '{"category": "POSITIVE", "reason": "IS_POSITIVE"}',
    "bad": '{"category": "NEGATIVE", "reason": "IS_NEGATIVE"}',
})
agent = Classifier(..., engine=engine)

# Single string response for all inputs
engine = create_mock_engine('{"value": "ok"}')

# Assert evaluation accuracy
eval_data = [
    {"input": {"text": "good"}, "category": "POSITIVE"},
    {"input": {"text": "bad"}, "category": "NEGATIVE"},
]
metrics = assert_accuracy_above(agent, eval_data, ["category"], 0.9)
```

`create_mock_engine` supports dict (input→response mapping), single string, or default fallback. Works with sync and async methods.
