
# Backlog - Agents7

### Tool calling follow-ups — 21st June 2026 (v4.4.0 shipped Ollama-only v1)

- [ ] Wire Anthropic provider for tool calling — implement `supports_tools()` → True and `build_tool_turn_messages()` on `AnthropicProvider`, and emit/parse `tool_use` / `tool_result` blocks in `chat`/`chat_async` so `ToolAgent` works on Claude models. Currently `tools=` is accepted and ignored. (MED, M)
- [ ] Wire OpenAI-compatible provider for tool calling — same as Anthropic but for `OpenAICompatProvider` (OpenAI `tools` / `tool_calls` / `role:tool` format). Broadest reach since most backends emulate this format. Currently `tools=` is accepted and ignored. (MED, M)
- [ ] Validate the Ollama tool-calling path end-to-end against a real model — `integration_tests/test_tool_agent.py` has only ever been SKIPPED (no Ollama on the dev box), so the full request → tool_calls → tool result → final answer round-trip is unvalidated against a live model; only the `FakeToolProvider` unit path has run. Run it against a tool-capable Ollama model and fix any wire-format drift. (MED, S)
- [ ] Streaming + tool calls — let `ToolAgent` stream intermediate steps/tokens during the loop. Out of v1 scope; each provider's streaming tool API is the hardest part. (LOW, L)
- [ ] Structured tool returns — `Tool`/`ToolRegistry` currently coerce every tool return to `str` before feeding it back to the model. Support structured/JSON returns (serialize dict/Pydantic returns) so tools can hand back rich data. (LOW, S)


### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.
