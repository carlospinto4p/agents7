# Tool calling

## Tool Calling

A `ToolAgent` can call plain Python functions. Pass callables as
`tools`; their JSON schema is derived from the type hints, and the
agent runs a bounded loop, executing tools and feeding results back
until the model produces a final answer.

```python
from agents7.agents import ToolAgent
from agents7.llm import LLMEngine


def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"It is 18 degrees and sunny in {city}."


agent = ToolAgent(
    name="weather-bot",
    version=1,
    engine=LLMEngine(provider="ollama", model_name="qwen3"),
    system_prompt="Answer weather questions using the tool.",
    tools=[get_weather],
    max_iterations=5,
)

print(agent.act("What's the weather in Paris?").response)
```

Tool calling is supported on the Ollama, Anthropic, and
OpenAI-compatible providers. The agent raises
`AgentConfigurationError` if the engine's provider does not support
tools, and `ToolLoopError` if the loop exceeds `max_iterations`.

Tools may return structured data, not just strings. A `dict`,
`list`, or Pydantic model return is JSON-encoded before being fed
back to the model (plain strings pass through unchanged), so a tool
can hand back rich data:

```python
def get_weather(city: str) -> dict:
    """Return the current weather for a city."""
    return {"city": city, "temp_c": 18, "conditions": "sunny"}
```

Containers are encoded recursively, so returning a bare
`list[Product]` of Pydantic models yields a JSON array of objects —
no wrapper model is needed to avoid Python `repr` output.

Arguments coming back from the model are validated against the same
schema the tool advertised, using the Pydantic model the schema was
derived from. Values in the wrong shape are coerced where possible
(a `"3"` for an `int` parameter becomes `3`), and anything that
cannot be coerced is reported to the model as a readable message
rather than an opaque Python error:

```
Error: Invalid arguments for `get_weather`: city: Input should be a valid string
```

Small local models make both mistakes routinely, so this is what
makes them usable for tool work. `Tool.validate_arguments()` exposes
the same check directly. A `Tool` constructed with a hand-written
`json_schema` for an untyped callable skips validation.

### Streaming the tool loop

`act_stream()` / `act_stream_async()` run the same bounded loop but
yield a `ToolStepEvent` for each step — every tool call, every tool
result, and the final answer — so you can observe intermediate steps
as they happen instead of waiting for the whole loop. The parsed
`AgentResponse` is available via `stream.agent_response` after
iteration.

```python
stream = agent.act_stream("What's the weather in Paris?")
for event in stream:
    if event.kind == "tool_call":
        print(f"→ calling {event.name}({event.arguments})")
    elif event.kind == "tool_result":
        print(f"← {event.name} returned {event.result}")
    elif event.kind == "answer":
        print(f"answer: {event.text}")

print(stream.agent_response.metrics)  # summed across all turns
```

A `tool_result` event carries `event.error` as well: the failure
message when the tool rejected the arguments or raised, and `None`
otherwise. It is a field of its own because `event.result` holds the
text fed back to the model, and a tool may legitimately return text
beginning with `Error:` — so the string cannot be read to tell a
rejected call from a successful one. `ToolRegistry.execute` returns
the same pair as a `ToolOutcome` (`output`, `error`, `failed`).

`event.kind` is a `ToolStepKind` (a `StrEnum`), so comparing it to
the plain strings above works as well as
`event.kind == ToolStepKind.ANSWER`. Async iteration is identical
with `async for` over `await agent.act_stream_async(...)`.

The final answer is delivered as a single `ANSWER` event (whole
text), not token-by-token — streaming raw tokens through a tool loop
needs provider-specific streaming tool APIs, which this method
deliberately avoids. `ToolLoopError` is still raised during
iteration if the loop exceeds `max_iterations`.
