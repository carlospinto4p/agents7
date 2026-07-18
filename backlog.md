
# Backlog - Agents7

### Tool calling — 18th July 2026

Found while building the catalog's `store_assistant` ToolAgent
showcase, driving it against four local Ollama models.

- [ ] Validate and coerce tool arguments against the schema
  `schema_from_callable` already derives, in `ToolRegistry.execute`.
  Models routinely send the right arguments in the wrong shape and
  the raw dict is currently splatted straight into the callable:
  `llama3.1:8b` sent `{"quantity": "3", "unit_price_cents": "1000"}`
  (strings for ints), and `qwen2.5:3b` intermittently sent
  `{"query": {"type": "string", "value": "blue mug"}}` — an envelope
  echoing the schema. Both surface to the model as opaque Python
  errors (`'<' not supported between instances of 'str' and 'int'`,
  `'dict' object has no attribute 'lower'`) instead of being coerced
  or reported as a usable validation message. Re-validating through
  the generated Pydantic args-model would fix both classes and make
  small models far more usable for tool work. (HIGH, M)
- [ ] Make `serialize_tool_result` recurse into containers of
  Pydantic models. A tool returning `list[Product]` currently hits
  the `json.dumps(..., default=str)` branch, so each element is
  stringified to a Python `repr` inside an otherwise-JSON array:
  `["sku='MUG-01' name='Blue ceramic mug' price_cents=1250"]`. The
  model then has to parse a repr. `store_assistant.find_product`
  works around it by returning a `ProductSearch` wrapper, but a bare
  list is the obvious thing to return and should just work. (MED, S)
- [ ] Support evaluating tool agents: add `tool` to `AGENT_TYPE_MAP`
  and a trajectory scorer that checks which tools were called, in
  what order, with what arguments — rather than comparing a single
  predicted label. Free-text answers make the current label-vs-label
  harness inapplicable, so `store_assistant` ships untracked by
  `eval_agent.py` and asserts its trajectory in unit tests instead.
  Blocks any accuracy claim about a tool agent under the roadmap's
  measurement invariant. (MED, L)
