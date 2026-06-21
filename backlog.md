
# Backlog - Agents7

### Provider configuration — 21st June 2026

- [ ] External-provider env vars + docs — define the required env vars (API key, base URL, model) for using an external OpenAI-compatible provider like DeepSeek, wire them into provider config, and document setup end-to-end in the README. (MED, M)

### Tool calling follow-ups — 21st June 2026 (v4.4.0 shipped Ollama-only v1)

- [ ] Validate the Ollama tool-calling path end-to-end against a real model — `integration_tests/test_tool_agent.py` has only ever been SKIPPED (no Ollama on the dev box), so the full request → tool_calls → tool result → final answer round-trip is unvalidated against a live model; only the `FakeToolProvider` unit path has run. Run it against a tool-capable Ollama model and fix any wire-format drift. (MED, S)
  - 2026-06-21: still blocked — `ollama` CLI is not installed and no server answers on `:11434` (conn refused), so the live run cannot be performed on this box. Static wire-format audit against the installed `ollama-python` types found **no drift**: the request `tools` schema (`{"type":"function","function":{name,description,parameters}}`), response parsing (`tool_calls[].function.name`/`.arguments`, null-safe `content`), and tool-result messages (`role:"tool"` + `tool_name`) all match `ollama._types.Tool` / `Message`. Only the live model round-trip remains to verify; run on a machine with a tool-capable Ollama model.
- [ ] Streaming + tool calls — let `ToolAgent` stream intermediate steps/tokens during the loop. Out of v1 scope; each provider's streaming tool API is the hardest part. (LOW, L)
- [x] Structured tool returns — `Tool`/`ToolRegistry` currently coerce every tool return to `str` before feeding it back to the model. Support structured/JSON returns (serialize dict/Pydantic returns) so tools can hand back rich data. (LOW, S)


### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.
