# Store Assistant

A tool-calling agent built on agents7's `ToolAgent`. It answers
questions about an online homeware store — what is sold, what is in
stock, what an order costs — by **calling Python functions** rather
than recalling facts from the model's weights.

This is the catalog's reference example of the tool-calling
round-trip:

```
"Can I order 3 blue mugs? What's the total?"

  -> turn 1  call   find_product({"query": "blue mug"})
             result {"query": "blue mug", "matches": [
                      {"sku": "MUG-01", "price_cents": 1250}]}
  -> turn 2  call   check_stock({"sku": "MUG-01"})
             result {"sku": "MUG-01", "units": 42, ...}
  -> turn 3  call   order_total({"unit_price_cents": 1250,
                                 "quantity": 3})
             result {"total_cents": 4961, ...}
  -> turn 4  answer "Yes, we have the Blue ceramic mug in stock.
                     The total for 3 of these mugs is EUR 49.61."
```

That transcript is real output, not an illustration.

## The tools

| Tool | Returns | Purpose |
|------|---------|---------|
| `find_product` | `ProductSearch` | Turn a description into SKUs and prices |
| `check_stock` | `StockLevel` | Units on hand, by SKU |
| `order_total` | `OrderTotal` | Subtotal, tax, shipping, total |

Resolved from `agents7.tools.demo` by dotted path — the same
convention a structured-output sample uses for `output_model`, so
this sample needs no package of its own.

Three framework behaviours are worth noting in the code:

- **Structured returns become JSON — if they are a single model.**
  Tools return Pydantic models; `Tool.run` serializes those with
  `model_dump_json`. This is why `find_product` returns a
  `ProductSearch` wrapper rather than a bare `list[Product]`: a list
  falls back to `json.dumps(..., default=str)`, which stringifies
  each element and would hand the model
  `["sku='MUG-01' name='Blue ceramic mug'..."]` — a Python `repr`
  inside a JSON array. The wrapper keeps the payload real JSON.
- **A raising tool is not fatal.** `ToolRegistry.execute` catches the
  exception and returns a `ToolOutcome` whose `output` —
  `"Error: <message>"` — is fed back to the model, which can then
  correct itself or explain the problem. `check_stock` raises
  `LookupError` on an unknown SKU to demonstrate this. The outcome
  also carries `error`, so the eval scorer can tell a rejected call
  from one that returned text beginning with `Error:`.
- **Optional arguments stay optional.** `order_total`'s `express`
  parameter has a default, so it is absent from the schema's
  `required` list.

## Usage

This is a **sample**, not a package — there is nothing to install.
Point the agent at the config:

```python
from agents7.agents.tool_agent import ToolAgent

agent = ToolAgent.from_yaml("samples/store_assistant/config.yml")

response = agent.act("Can I order 3 blue mugs, and what's the total?")
print(response.response)
print(response.metrics.tokens)  # summed across every model turn
```

Observe the intermediate steps instead of waiting for the answer:

```python
stream = agent.act_stream("Any blue mugs left?")
for event in stream:
    if event.kind == "tool_call":
        print(event.name, event.arguments)
    elif event.kind == "tool_result":
        print(event.result)
    elif event.kind == "answer":
        print(event.text)
```

The tools are importable on their own, with no agent involved:

```python
from agents7.tools.demo import check_stock, find_product

sku = find_product("cast iron teapots").matches[0].sku
print(check_stock(sku).units)  # 3
```

Matching is token-based and strips a trailing plural `s`, so a model
asking for "blue mugs" finds the "Blue ceramic mug". Naive substring
matching missed exactly that case against a real model.

## Data

`agents7.tools.demo` holds a five-product literal dataset with its
stock levels. The demo is the round-trip, not a database, so the
data lives in process: every tool is pure and deterministic, and the
agent needs no service beyond the LLM.

## Configuration

`config.yml` is a `ToolAgent` config: the usual `name` / `version` /
`engine` / `system_prompt`, plus a `tools` list of dotted import
paths and the loop's `max_iterations`. The system prompt is what
keeps the agent honest — it forbids inventing a SKU, price or stock
level, and forbids doing the arithmetic itself.

It runs on Ollama `gemma4:12b` by default. Tool calling works across
all three providers, so pointing `engine` at Anthropic or an
OpenAI-compatible API needs no code change.

### On model choice

A multi-hop chain asks more of a model than a single call, and the
smaller local models are not all up to it. Scored against
`eval_data.jsonl` on 3rd September 2026 — agents7 v5.0.0, Ollama,
`temperature: 0.2`, default `in_order` scoring. The duration column
is the call log's total LLM time, not wall clock:

| Model | Rows passed | Expected calls made | Extra calls | Rejected calls | LLM time |
|-------|-------------|---------------------|-------------|----------------|----------|
| `gemma4:12b` | 12/12 | 21/21 | 3 | 1 | 72.4s |
| `qwen3:8b` | 11/12 | 20/21 | 3 | 1 | 27.9s |
| `qwen2.5:3b` | 4/12 | 6/21 | 14 | 13 | 23.2s |
| `llama3.1:8b` | 4/12 | 11/21 | 7 | 7 | 32.9s |

**This is one run per model, not a baseline.** Each figure is a
single sample of a process that varies, so treat a gap of a row or
two as noise and only the two clear tiers as real. Reproduce any
row with the `--model` command below.

A **rejected call** is one the tool threw out — bad arguments, or an
unknown SKU. The single rejection the two larger models each carry
is row 9, which asks about SKU `NOPE-99` and exists precisely to
make `check_stock` raise; it is marked `"allow_failure": true` in
the golden set, so it counts as a call made. Every other rejection
in the table is a real failure, and the two smaller models' totals
are dominated by them.

The two that fail do so by tool, not uniformly. Nothing survives the
argument problems for `qwen2.5:3b` — `find_product` 3/10,
`check_stock` 2/6, `order_total` 1/5 — while `llama3.1:8b` gets the
first hop right and loses everything after it: `find_product` 9/10,
`check_stock` 1/6, `order_total` 1/5. A model that cannot carry a
result from one call into the next fails the whole second half of
the chain.

Each has one dominant failure, and both are visible in the report's
per-case list:

- `llama3.1:8b` **plans the chain in one turn and fills the
  arguments with placeholders**, calling `check_stock(sku='result
  of find_product')` and `order_total(unit_price_cents='result of
  check_stock')` — references to results it has not received. Those
  are the rejections in its row: an invented SKU makes `check_stock`
  raise, and a non-numeric price makes `order_total` reject the
  arguments outright.
- `qwen2.5:3b` **wraps arguments in a `{"type": "string", "value":
  ...}` envelope** echoing the schema, so the tool receives a dict
  where it expects a string. Not intermittent: 8 of its 12 rows.

The framework **catches** that second shape rather than absorbing
it. `ToolRegistry.execute` validates arguments against the schema
it derives and hands the model back ``Error: Invalid arguments for
`find_product`: query: Input should be a valid string`` (agents7
v4.13.0). That is a clean, correctable signal, and two rows show
`qwen2.5:3b` retrying with a plain string and recovering — but on
the other six it apologises to the customer instead, so the call is
still lost. A model has to act on the error for the validation to
save the turn.

Prefer a larger model for tool work.

## Evaluation

This agent is scored on its **trajectory** — which tools it called, in
what order, with what arguments — rather than on a predicted label,
since `agent_type: tool` produces free text with nothing to compare
against. `eval_data.jsonl` holds 12 rows, each pairing a customer
question with the calls the config's rules require:

```bash
# from this sample's directory
uv run --project ../.. agents7 eval \
    -c config.yml -d eval_data.jsonl -r reports/store_assistant.md
```

The run writes `reports/store_assistant.md` with per-tool recall and,
for every failing row, exactly which calls were missing or extra.

To score the same rows on another model, change `model:` in
`config.yml` (it takes `model` or `provider::model`) and give the
report another name, so per-model results sit side by side instead of
overwriting each other.

Rows pin the arguments that are deterministic (a `sku` from the
catalog, a `unit_price_cents`/`quantity` for a total) and leave
`find_product`'s `query` unpinned — the wording a model searches with
is a legitimate free choice, and pinning it would score phrasing
rather than behaviour.

`order_total`'s optional `express` is neither. The four rows for an
order nobody asked to expedite carry `"defaults": {"express":
false}`, which passes a model that omits the argument and fails one
that invents `express=true`. Row 6 — *"shipped express"* — pins it
under `arguments` instead, because there it is a requirement rather
than a default to leave alone.

Row 9 — *"Is the SKU NOPE-99 in stock?"* — carries `"allow_failure":
true`. `check_stock` raises `LookupError` on an unknown SKU, and the
whole point of the row is that the agent relays that rather than
guessing, so the call counts as made even though it never returned.
Every other row leaves the key off and a rejected call fails it.

Scoring defaults to `in_order`, so a model that calls an extra tool on
the way to the right answer still passes while the extra call is
counted. Pass `--trajectory-mode exact` (or set `trajectory_mode:
exact` in `config.yml`) to require the exact sequence.

The gap between the two modes is the price of that tolerance:
`gemma4:12b` scores 12/12 in order and **9/12 exact**, losing
precisely the three rows where it checked stock before quoting a
total — sensible behaviour a strict sequence has no way to allow.
Read `in_order` as "did it get there", `exact` as "did it take the
declared route", and expect the second to be lower for a model
doing nothing wrong.

Three rows used to pass while the model did the wrong thing, and
all three are fixed as of agents7 v5.0.0 — the current table is the
first scored under the corrected rules:

- **A row expecting no calls could never fail**, since with nothing
  to miss every call the agent made was merely extra, which
  `in_order` tolerates. `llama3.1:8b` answered "what time do you
  open?" by searching the catalog for "store hours" and passed. An
  empty `tools` list is now scored under `exact` in either mode, and
  that row is a fail (v4.38.2).
- **An invented optional argument passed**, because a row pins only
  the keys it names. `qwen3:8b` added `express=True` to an order
  nobody asked to expedite and quoted the customer EUR 115.33 for a
  EUR 102.34 order, still passing. The four affected rows now carry
  `"defaults": {"express": false}`, and that row is a fail (v4.39.0)
  — the single row separating `qwen3:8b` from `gemma4:12b` above.
- **A call the tool rejected passed**, because the scorer looked at
  which tools were called and not at whether the call worked.
  `qwen2.5:3b` sent `find_product(query={'type': 'string', 'value':
  'one bamboo serving tray'})`, the tool threw the argument out,
  the agent apologised to the customer without a product — and the
  row passed on the attempt. Pinning `query` is no answer: the row
  leaves it unpinned because its wording is the model's to choose,
  which is exactly the case where a malformed value slips through.
  A rejected call now satisfies nothing (v5.0.0), which is most of
  why `qwen2.5:3b` drops from 5/12 to 4/12 while its expected calls
  made drop from 10/18 to 6/21.

The tool functions' own business logic (catalog matching, stock
lookup, order costing) is covered by the framework's own hermetic
test suite — no Ollama or agent involved:

```bash
uv run pytest tests/test_tools_demo.py -v
```

## Requirements

- Python >= 3.11
- agents7 >= 5.0.0 (the golden set uses `allow_failure`)
- Ollama with a tool-capable model (`gemma4:12b` by default), only to
  run the agent for real
