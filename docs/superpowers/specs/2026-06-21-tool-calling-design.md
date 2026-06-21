# Tool / Function Calling — Design (v1)

**Date:** 2026-06-21
**Status:** Approved, pending spec review
**Backlog item:** agents7_framework — "Tool use / function calling"
(HIGH, L)

## Goal

Let agents invoke registered Python functions (call APIs, read
files, compute, etc.) as part of answering. The model decides
which tool to call; the framework executes it and feeds the
result back, looping until the model returns a final answer.

This is the single largest capability gap versus typical agent
frameworks. v1 delivers the full abstraction with the **Ollama**
provider wired; Anthropic and OpenAI-compatible providers slot
into the same interface as a clear follow-up.

## Scope decisions (locked during brainstorming)

| Decision | Choice |
|----------|--------|
| Provider scope (v1) | Ollama wired; abstraction designed for all three |
| Agent surface | New dedicated `ToolAgent` (existing agents untouched) |
| Tool definition | Plain callable + auto-derived JSON schema |
| Loop & runtime | Auto-loop, bounded by `max_iterations`, sync + async, no streaming |

Out of scope for v1: streaming tool calls; Anthropic/OpenAI tool
wiring (designed-for, not implemented); parallel tool execution.

## Architecture

Four small, independently-testable units.

### 1. Tool abstraction — `agents7/tools/`

New subpackage. Provider-neutral; no LLM dependency.

- **`Tool`** — wraps one callable.
  - Fields: `name` (default `fn.__name__`), `description` (first
    line of the docstring), `json_schema` (auto-derived), `fn`.
  - `run(**kwargs) -> str` — invokes `fn`, coerces the return to
    a string for feeding back to the model.
- **`schema_from_callable(fn) -> dict`** — derives a JSON schema:
  builds a Pydantic model from the callable's type hints via
  `pydantic.create_model`, returns `model_json_schema()`. Raises
  `ToolSchemaError` for unsupported signatures (missing type
  hints, `*args`/`**kwargs`).
- **`ToolRegistry`** — maps `name -> Tool`.
  - `schemas() -> list[dict]` — provider-neutral JSON schemas to
    hand the model.
  - `execute(name, arguments: dict) -> str` — look up, run,
    catch exceptions and return `"Error: <message>"` so the model
    can recover rather than the loop crashing.
  - Raises `ToolSchemaError` on unknown tool name.

A function becomes a tool by being passed to `ToolAgent`; the
agent wraps bare callables in `Tool` automatically. Passing a
pre-built `Tool` is also allowed (explicit schema override).

### 2. Provider / result-dict changes (normalization seam)

The normalized result dict gains a tool-call slot, and the
provider gains two hooks for the loop's message reconstruction.

- **Result dict** gains `tool_calls: list[ToolCall]`, defaulting
  to `[]`. `ToolCall` is a small frozen dataclass
  `{name: str, arguments: dict}`. Every existing caller ignores
  the new key — fully backward-compatible.
- **`chat` / `chat_async`** (on `LLMProvider` and `LLMEngine`)
  gain `tools: list[dict] | None = None`, threaded through
  `EngineDispatcher.execute` / `execute_async`.
  - `tools` is folded into the cache key so different tool sets
    don't collide. Caching is only active at `temperature == 0`,
    so tool agents at higher temperatures bypass it naturally.
- **Provider hooks:**
  - `supports_tools() -> bool` — Ollama returns `True`; base
    default `False`. `ToolAgent` asserts this at construction.
  - `build_tool_turn_messages(assistant_text, tool_calls,
    results) -> list[dict]` — returns the assistant tool-call
    message plus tool-result messages **in that provider's wire
    shape**. Ollama implements it; base raises
    `NotImplementedError`. This is the per-provider seam that
    keeps the agent loop provider-agnostic.
- **Ollama provider:** `chat`/`chat_async` pass `tools=` to
  `ollama.chat`; `parse_ollama_chat_response` populates
  `tool_calls` from `message.tool_calls`.

**Rejected alternative:** keep raw provider tool-call dicts in the
result. Rejected — leaks Ollama's shape into the agent loop and
makes the other providers a rewrite rather than a drop-in.

### 3. `ToolAgent` — `agents7/agents/tool_agent.py`

Frozen dataclass extending `BaseAgent`.

- Fields: `tools: list[Callable | Tool]`, `max_iterations:
  int = 5`.
- `__post_init__`: build a `ToolRegistry` from `tools`; assert the
  engine's provider `supports_tools()`.
- `act(user_input) -> AgentResponse` / `act_async(...)`:
  1. `messages = [system, user]`.
  2. Loop up to `max_iterations`:
     - `result = engine.chat(messages, tools=registry.schemas())`
     - if `result["tool_calls"]`: execute each via the registry,
       append `provider.build_tool_turn_messages(...)`, continue.
     - else: return `AgentResponse(response=result["response"],
       model=None, thinking=result["thinking"], metrics=<summed>)`.
  3. On exceeding `max_iterations`, raise `ToolLoopError`.
- **Metrics** are accumulated across every LLM call in the loop
  (tokens, prompt_tokens, total_duration, cost_cents) so reporting
  stays honest for multi-step runs.
- `from_yaml` mirrors the existing config-loader pattern; tools are
  referenced by import path.

### 4. Exceptions

New exceptions in `agents7/exceptions.py`:
- `ToolSchemaError` — bad signature or unknown tool name.
- `ToolLoopError` — `max_iterations` exceeded.

## Data flow

```
ToolAgent.act("What's the weather in Paris?")
  → engine.chat([system, user], tools=[weather_schema])
  → result.tool_calls = [ToolCall("get_weather", {"city": "Paris"})]
  → registry.execute("get_weather", {"city": "Paris"}) → "18C, clear"
  → messages += provider.build_tool_turn_messages(...)
  → engine.chat(messages, tools=[...])
  → result.tool_calls = []  → final answer
  → AgentResponse(response="It's 18C and clear in Paris", ...)
```

## Error handling

- Tool raises → caught in `registry.execute`, returned as
  `"Error: <message>"` so the model can react.
- Unknown tool name requested by model → `registry.execute`
  raises `ToolSchemaError` (a model hallucinating a tool name is a
  hard error, not a recoverable tool failure).
- Loop never converges → `ToolLoopError` after `max_iterations`.
- Provider without tool support → `ToolAgent.__post_init__`
  raises immediately (fail fast at construction, not mid-run).

## Testing

Per `testing-python.md` (split unit / integration).

- **Unit** (`tests/`, no network):
  - `schema_from_callable`: happy paths across signature shapes;
    error cases (missing hints, `*args`).
  - `ToolRegistry.execute`: happy path, tool-raises path, unknown
    name.
  - `ToolAgent`: driven by a **fake `LLMProvider`** returning
    scripted `tool_calls` then a final answer. Verifies multi-step
    loop, `max_iterations` overflow → `ToolLoopError`, and metric
    accumulation — all with zero real LLM calls.
- **Integration** (`integration_tests/`, real Ollama): one
  end-to-end `ToolAgent` run against a tool-capable local model;
  skipped if Ollama is unreachable.

## Docs / release

- README: new "Tool calling" section with a runnable example.
- `changelog.md`: minor version bump (new public API: `tools`
  subpackage, `ToolAgent`, `tools=` param, `tool_calls` result
  field).
- `CLAUDE.md`/rules: no change expected.

## Backward compatibility

All changes are additive. The `tool_calls` result key and the
`tools=` chat param both default to empty/None; existing agents,
providers, and the dispatcher behave identically when no tools are
passed.
