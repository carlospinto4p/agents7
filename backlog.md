
# Backlog - Agents7

### Tool calling — 18th July 2026

Found while building the catalog's `store_assistant` ToolAgent
showcase, driving it against four local Ollama models.

- [ ] Support evaluating tool agents: add `tool` to `AGENT_TYPE_MAP`
  and a trajectory scorer that checks which tools were called, in
  what order, with what arguments — rather than comparing a single
  predicted label. Free-text answers make the current label-vs-label
  harness inapplicable, so `store_assistant` ships untracked by
  `eval_agent.py` and asserts its trajectory in unit tests instead.
  Blocks any accuracy claim about a tool agent under the roadmap's
  measurement invariant. (MED, L)
