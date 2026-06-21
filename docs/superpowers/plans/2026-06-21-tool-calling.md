# Tool / Function Calling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an agent invoke registered Python functions in a bounded call→execute-tool→call loop until the model returns a final answer, wired for the Ollama provider and designed so Anthropic/OpenAI slot in later.

**Architecture:** A provider-neutral `tools/` subpackage (auto-schema from callables, `Tool`, `ToolRegistry`); an additive `tool_calls` slot on the normalized result dict plus a `tools=` chat param threaded through `LLMEngine`/`EngineDispatcher`; two provider hooks (`supports_tools`, `build_tool_turn_messages`) that keep the agent loop provider-agnostic; and a new `ToolAgent`.

**Tech Stack:** Python 3.12+, Pydantic (`create_model`/`model_json_schema`), `ollama` client, pytest.

## Global Constraints

- Line length: 78 characters (ruff-enforced).
- Type hints required on public methods. `@staticmethod` for methods not using `self`.
- Docstrings: Sphinx/reST (`:param:`, `:return:`, `:raises:`), single backticks for inline code.
- Tests: plain `test_` functions, no classes. Flat `tests/` dir; shared helpers in `tests/helpers.py` imported as `from helpers import ...`. Integration tests in `integration_tests/`, skipped if Ollama unreachable.
- All changes additive/backward-compatible: `tool_calls` defaults to `[]`, `tools=` defaults to `None`; existing callers and cache keys unchanged.
- `tool_calls` in the result dict are plain dicts `{"name": str, "arguments": dict}` (JSON-serializable for `SqliteCache`), NOT a dataclass.
- Run unit tests from `agents7_framework/`: `uv run pytest tests/ -v`.
- After code: bump `agents7_framework/pyproject.toml` (minor) + `changelog.md`, `git add agents7_framework/uv.lock`, update `README.md`.

---

### Task 1: Auto-schema from callables + tool exceptions

**Files:**
- Modify: `src/agents7/exceptions.py`
- Create: `src/agents7/tools/__init__.py`
- Create: `src/agents7/tools/schema.py`
- Test: `tests/test_tool_schema.py`

**Interfaces:**
- Produces: `schema_from_callable(fn: Callable) -> dict`; exceptions `ToolSchemaError`, `ToolLoopError` (both subclass `Agents7Exception`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tool_schema.py
"""Unit tests for agents7.tools.schema.schema_from_callable."""

import pytest

from agents7.exceptions import ToolSchemaError
from agents7.tools.schema import schema_from_callable


def test_schema_basic_types():
    def get_weather(city: str, days: int = 1) -> str:
        return ""

    schema = schema_from_callable(get_weather)
    assert schema["type"] == "object"
    assert schema["properties"]["city"]["type"] == "string"
    assert schema["properties"]["days"]["type"] == "integer"
    # `city` is required (no default), `days` is not.
    assert schema["required"] == ["city"]


def test_schema_missing_hint_raises():
    def bad(city) -> str:
        return ""

    with pytest.raises(ToolSchemaError, match="missing a type hint"):
        schema_from_callable(bad)


def test_schema_var_args_raises():
    def bad(*args: int) -> str:
        return ""

    with pytest.raises(ToolSchemaError, match="unsupported"):
        schema_from_callable(bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tool_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: agents7.tools`.

- [ ] **Step 3: Add the exceptions**

Append to `src/agents7/exceptions.py`:

```python
class ToolSchemaError(Agents7Exception):
    """Raised when a tool signature cannot be turned into a
    JSON schema, or an unknown tool is requested."""


class ToolLoopError(Agents7Exception):
    """Raised when a ToolAgent exceeds its max_iterations."""
```

- [ ] **Step 4: Write the schema package**

`src/agents7/tools/__init__.py`:

```python
"""Tool / function-calling support for agents."""

from agents7.tools.schema import schema_from_callable


__all__ = ["schema_from_callable"]
```

`src/agents7/tools/schema.py`:

```python
"""Derive a JSON schema from a Python callable's signature."""

import inspect
from collections.abc import Callable

from pydantic import create_model

from agents7.exceptions import ToolSchemaError


def schema_from_callable(fn: Callable) -> dict:
    """Build a JSON schema for *fn*'s parameters.

    Constructs a Pydantic model from the callable's type
    hints and returns its ``model_json_schema()``. Parameters
    without a default are marked required.

    :param fn: The function to derive a schema from. Every
        parameter must have a type hint; ``*args`` / ``**kwargs``
        are not supported.
    :return: A JSON-schema dict describing the parameters.
    :raises ToolSchemaError: If a parameter lacks a type hint or
        the signature uses ``*args`` / ``**kwargs``.
    """
    sig = inspect.signature(fn)
    fields: dict[str, tuple] = {}
    for name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ToolSchemaError(
                f"Tool `{fn.__name__}` has an unsupported "
                f"*args/**kwargs parameter `{name}`."
            )
        if param.annotation is inspect.Parameter.empty:
            raise ToolSchemaError(
                f"Parameter `{name}` of tool `{fn.__name__}` is "
                f"missing a type hint."
            )
        default = (
            ...
            if param.default is inspect.Parameter.empty
            else param.default
        )
        fields[name] = (param.annotation, default)
    model = create_model(f"{fn.__name__}_Args", **fields)
    return model.model_json_schema()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_tool_schema.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/agents7/exceptions.py src/agents7/tools/ tests/test_tool_schema.py
git commit -m "feat: add tool schema derivation and tool exceptions"
```

---

### Task 2: Tool wrapper + ToolRegistry

**Files:**
- Create: `src/agents7/tools/tool.py`
- Create: `src/agents7/tools/registry.py`
- Modify: `src/agents7/tools/__init__.py`
- Test: `tests/test_tool_registry.py`

**Interfaces:**
- Consumes: `schema_from_callable` (Task 1), `ToolSchemaError` (Task 1).
- Produces:
  - `Tool(fn, name="", description="", json_schema={})` with `.run(**kwargs) -> str`, resolved `.name`, `.description`, `.json_schema`.
  - `ToolRegistry(tools: list[Tool | Callable])` with `.schemas() -> list[dict]`, `.execute(name: str, arguments: dict) -> str`, `.names -> list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tool_registry.py
"""Unit tests for agents7.tools Tool and ToolRegistry."""

import pytest

from agents7.exceptions import ToolSchemaError
from agents7.tools.registry import ToolRegistry
from agents7.tools.tool import Tool


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def boom(x: int) -> int:
    """Always fails."""
    raise RuntimeError("kaboom")


def test_tool_defaults_from_callable():
    tool = Tool(add)
    assert tool.name == "add"
    assert tool.description == "Add two numbers."
    assert tool.json_schema["properties"]["a"]["type"] == "integer"


def test_tool_run_coerces_to_str():
    assert Tool(add).run(a=2, b=3) == "5"


def test_registry_schemas_shape():
    reg = ToolRegistry([add])
    schema = reg.schemas()[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "add"
    assert "parameters" in schema["function"]


def test_registry_execute_happy():
    assert ToolRegistry([add]).execute("add", {"a": 4, "b": 1}) == "5"


def test_registry_execute_tool_error_returned_as_string():
    out = ToolRegistry([boom]).execute("boom", {"x": 1})
    assert out.startswith("Error:")
    assert "kaboom" in out


def test_registry_execute_unknown_tool_raises():
    with pytest.raises(ToolSchemaError, match="Unknown tool"):
        ToolRegistry([add]).execute("nope", {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tool_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: agents7.tools.tool`.

- [ ] **Step 3: Write `tool.py`**

```python
"""A callable wrapped as an LLM-invocable tool."""

from collections.abc import Callable
from dataclasses import dataclass, field

from agents7.tools.schema import schema_from_callable


@dataclass(frozen=True)
class Tool:
    """Wrap a Python callable as a tool the model can call.

    :param fn: The function to expose.
    :param name: Tool name shown to the model. Defaults to
        ``fn.__name__``.
    :param description: Human description. Defaults to the first
        line of ``fn``'s docstring.
    :param json_schema: Parameter schema. Auto-derived from
        ``fn`` when omitted.
    """

    fn: Callable
    name: str = ""
    description: str = ""
    json_schema: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.fn.__name__)
        if not self.description:
            doc = (self.fn.__doc__ or "").strip().split("\n")[0]
            object.__setattr__(self, "description", doc)
        if not self.json_schema:
            object.__setattr__(
                self, "json_schema", schema_from_callable(self.fn)
            )

    def run(self, **kwargs) -> str:
        """Invoke the wrapped function and stringify its return.

        :param kwargs: Arguments parsed from the model's tool call.
        :return: The function's return value coerced to ``str``.
        """
        return str(self.fn(**kwargs))
```

- [ ] **Step 4: Write `registry.py`**

```python
"""A name-indexed collection of tools an agent can call."""

from collections.abc import Callable

from agents7.exceptions import ToolSchemaError
from agents7.tools.tool import Tool


class ToolRegistry:
    """Maps tool names to `Tool` instances and runs them.

    :param tools: A list of `Tool` instances or bare callables.
        Bare callables are wrapped in `Tool` automatically.
    """

    def __init__(self, tools: list[Tool | Callable]):
        self._tools: dict[str, Tool] = {}
        for entry in tools:
            tool = entry if isinstance(entry, Tool) else Tool(entry)
            self._tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        """Return provider-neutral function schemas for every tool.

        :return: A list of ``{"type": "function", "function":
            {...}}`` dicts suitable for an OpenAI/Ollama-style
            ``tools`` parameter.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.json_schema,
                },
            }
            for tool in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict) -> str:
        """Run the named tool with *arguments*.

        :param name: Tool name requested by the model.
        :param arguments: Keyword arguments for the tool.
        :return: The tool's stringified result, or
            ``"Error: <message>"`` if the tool raised.
        :raises ToolSchemaError: If *name* is not registered.
        """
        if name not in self._tools:
            raise ToolSchemaError(f"Unknown tool `{name}` requested.")
        try:
            return self._tools[name].run(**arguments)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model
            return f"Error: {exc}"

    @property
    def names(self) -> list[str]:
        """Registered tool names."""
        return list(self._tools)
```

- [ ] **Step 5: Update `tools/__init__.py`**

```python
"""Tool / function-calling support for agents."""

from agents7.tools.registry import ToolRegistry
from agents7.tools.schema import schema_from_callable
from agents7.tools.tool import Tool


__all__ = ["Tool", "ToolRegistry", "schema_from_callable"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_tool_registry.py -v`
Expected: PASS (6 tests).

- [ ] **Step 7: Commit**

```bash
git add src/agents7/tools/ tests/test_tool_registry.py
git commit -m "feat: add Tool wrapper and ToolRegistry"
```

---

### Task 3: Provider tool hooks + Ollama wiring + result-dict slot

**Files:**
- Modify: `src/agents7/llm/base.py` (the `LLMProvider` class)
- Modify: `src/agents7/llm/ollama.py`
- Modify: `src/agents7/llm/anthropic.py`
- Modify: `src/agents7/llm/openai_compat.py`
- Test: `tests/test_tool_provider.py`

**Interfaces:**
- Produces (on `LLMProvider` and subclasses):
  - `chat(..., tools: list[dict] | None = None)` and `chat_async(..., tools=None)` accept a `tools` arg.
  - Result dict gains `tool_calls: list[dict]` (each `{"name": str, "arguments": dict}`), defaulting `[]`.
  - `supports_tools() -> bool` (default `False`; Ollama `True`).
  - `build_tool_turn_messages(assistant_text: str, tool_calls: list[dict], results: list[str]) -> list[dict]` (base raises `NotImplementedError`; Ollama implements).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tool_provider.py
"""Unit tests for provider-level tool support (Ollama)."""

from types import SimpleNamespace

from agents7.llm.ollama import (
    OllamaProvider,
    parse_ollama_chat_response,
)


def _msg(content="", tool_calls=None, thinking=None, role="assistant"):
    return SimpleNamespace(
        content=content,
        role=role,
        thinking=thinking,
        tool_calls=tool_calls,
    )


def _chat_response(message):
    return SimpleNamespace(
        message=message,
        eval_count=1,
        prompt_eval_count=2,
        total_duration=3_000_000_000,
    )


def test_parse_no_tool_calls_defaults_empty():
    parsed = parse_ollama_chat_response(_chat_response(_msg("hi")))
    assert parsed["tool_calls"] == []
    assert parsed["response"] == "hi"


def test_parse_extracts_tool_calls():
    tc = SimpleNamespace(
        function=SimpleNamespace(
            name="get_weather", arguments={"city": "Paris"}
        )
    )
    parsed = parse_ollama_chat_response(
        _chat_response(_msg(content=None, tool_calls=[tc]))
    )
    assert parsed["tool_calls"] == [
        {"name": "get_weather", "arguments": {"city": "Paris"}}
    ]
    assert parsed["response"] == ""  # null content is null-safe


def test_ollama_supports_tools():
    assert OllamaProvider("qwen3").supports_tools() is True


def test_build_tool_turn_messages_shape():
    provider = OllamaProvider("qwen3")
    calls = [{"name": "add", "arguments": {"a": 1, "b": 2}}]
    msgs = provider.build_tool_turn_messages("thinking...", calls, ["3"])
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["tool_calls"][0]["function"]["name"] == "add"
    assert msgs[1]["role"] == "tool"
    assert msgs[1]["content"] == "3"
    assert msgs[1]["tool_name"] == "add"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tool_provider.py -v`
Expected: FAIL — `parse_ollama_chat_response` has no `tool_calls` key / `supports_tools` undefined.

- [ ] **Step 3: Add base hooks and `tools` param to `LLMProvider`**

In `src/agents7/llm/base.py`, add `tools: list[dict] | None = None` to the `chat` and `chat_async` abstract method signatures (last param). Then add these two methods to `LLMProvider` (after `get_model_info`):

```python
    def supports_tools(self) -> bool:
        """Whether this provider supports tool/function calling.

        :return: ``False`` by default; tool-capable providers
            override to return ``True``.
        """
        return False

    def build_tool_turn_messages(
        self,
        assistant_text: str,
        tool_calls: list[dict],
        results: list[str],
    ) -> list[dict]:
        """Build the messages appended after a tool round.

        Returns the assistant tool-call message plus one
        tool-result message per call, in this provider's wire
        format. Overridden by tool-capable providers.

        :param assistant_text: The assistant's text alongside the
            tool calls (often empty).
        :param tool_calls: The ``{"name", "arguments"}`` dicts the
            model requested.
        :param results: Stringified tool results, aligned by index
            with *tool_calls*.
        :return: A list of message dicts to extend the history.
        :raises NotImplementedError: If the provider does not
            support tools.
        """
        raise NotImplementedError(
            f"Provider `{self.provider}` does not support tools."
        )
```

- [ ] **Step 4: Wire Ollama**

In `src/agents7/llm/ollama.py`:

(a) Add `tools` to `_build_call_kwargs` — add a `tools: list[dict] | None = None` keyword param and, after building `kwargs`, insert:

```python
        if tools is not None:
            kwargs["tools"] = tools
```

(b) Add `tools: list[dict] | None = None` to `chat` and `chat_async` signatures (last param) and pass `tools=tools` into the `self._build_call_kwargs(...)` call in each.

(c) Add a tool-calls parser and make content null-safe. Replace the `parse_ollama_chat_response` return with:

```python
def _parse_tool_calls(message) -> list[dict]:
    """Extract tool calls from an Ollama message as plain dicts.

    :param message: The ``message`` object from an Ollama chat
        response.
    :return: A list of ``{"name", "arguments"}`` dicts (empty if
        the model requested no tools).
    """
    raw = getattr(message, "tool_calls", None) or []
    return [
        {
            "name": call.function.name,
            "arguments": dict(call.function.arguments),
        }
        for call in raw
    ]
```

and in `parse_ollama_chat_response`, change the `response` line to be null-safe and add the `tool_calls` key:

```python
    return {
        "response": (chat_response.message.content or "").strip(),
        "role": chat_response.message.role,
        "thinking": chat_response.message.thinking.strip()
        if chat_response.message.thinking
        else None,
        "tool_calls": _parse_tool_calls(chat_response.message),
        "metrics": {
            "tokens": chat_response.eval_count,
            "prompt_tokens": chat_response.prompt_eval_count,
            "total_duration": chat_response.total_duration / 10**9,
        },
    }
```

(d) Add the two hook overrides to `OllamaProvider`:

```python
    def supports_tools(self) -> bool:
        """Ollama supports tool calling."""
        return True

    def build_tool_turn_messages(
        self,
        assistant_text: str,
        tool_calls: list[dict],
        results: list[str],
    ) -> list[dict]:
        """Build Ollama assistant + tool-result messages.

        See `LLMProvider.build_tool_turn_messages`.
        """
        assistant = {
            "role": "assistant",
            "content": assistant_text,
            "tool_calls": [
                {
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    }
                }
                for call in tool_calls
            ],
        }
        tool_msgs = [
            {
                "role": "tool",
                "content": result,
                "tool_name": call["name"],
            }
            for call, result in zip(tool_calls, results)
        ]
        return [assistant, *tool_msgs]
```

- [ ] **Step 5: Accept-and-ignore `tools` on the other two providers**

In `src/agents7/llm/anthropic.py` and `src/agents7/llm/openai_compat.py`, add `tools: list[dict] | None = None` as the last param to each provider's `chat` and `chat_async` signatures. Do not otherwise use it (v1 wires only Ollama). This prevents a `TypeError` when `LLMEngine.chat` forwards `tools=None` to them. Their inherited `supports_tools()` stays `False`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_tool_provider.py -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Run the provider suites for regressions**

Run: `uv run pytest tests/test_openai_compat.py tests/test_anthropic_provider.py tests/test_streaming.py -q`
Expected: PASS (no regressions from the new `tools` param).

- [ ] **Step 8: Commit**

```bash
git add src/agents7/llm/base.py src/agents7/llm/ollama.py src/agents7/llm/anthropic.py src/agents7/llm/openai_compat.py tests/test_tool_provider.py
git commit -m "feat: add provider tool hooks and Ollama tool wiring"
```

---

### Task 4: Thread `tools` through LLMEngine, dispatcher and cache key

**Files:**
- Modify: `src/agents7/llm/base.py` (the `LLMEngine` class)
- Modify: `src/agents7/llm/dispatcher.py`
- Modify: `src/agents7/llm/cache.py`
- Test: `tests/test_tool_engine.py`

**Interfaces:**
- Consumes: provider hooks from Task 3.
- Produces:
  - `LLMEngine.chat(..., tools=None)` / `chat_async(..., tools=None)` forward `tools` to the dispatcher.
  - `LLMEngine.supports_tools() -> bool` and `LLMEngine.build_tool_turn_messages(assistant_text, tool_calls, results) -> list[dict]` delegate to the provider.
  - `_CacheKeyBuilder.make_key(..., tools=None)` includes `tools` in the key only when non-None.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tool_engine.py
"""Unit tests for tool threading through LLMEngine."""

from helpers import DUMMY_METRICS, make_engine
from agents7.llm.cache import _CacheKeyBuilder


def test_engine_forwards_tools_to_provider():
    engine = make_engine()
    engine.llm_provider.chat.return_value = {
        "role": "assistant",
        "response": "ok",
        "thinking": None,
        "tool_calls": [],
        "metrics": DUMMY_METRICS,
    }
    tools = [{"type": "function", "function": {"name": "x"}}]
    engine.chat([{"role": "user", "content": "hi"}], tools=tools)
    _, kwargs = engine.llm_provider.chat.call_args
    assert kwargs["tools"] == tools


def test_engine_supports_tools_delegates():
    engine = make_engine()
    engine.llm_provider.supports_tools.return_value = True
    assert engine.supports_tools() is True


def test_make_key_stable_without_tools():
    # Adding the tools param must not change keys when tools is None.
    a = _CacheKeyBuilder.make_key("m", "chat", messages=[], temperature=0.0)
    b = _CacheKeyBuilder.make_key(
        "m", "chat", messages=[], temperature=0.0, tools=None
    )
    assert a == b


def test_make_key_changes_with_tools():
    a = _CacheKeyBuilder.make_key("m", "chat", messages=[], temperature=0.0)
    c = _CacheKeyBuilder.make_key(
        "m", "chat", messages=[], temperature=0.0, tools=[{"x": 1}]
    )
    assert a != c
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tool_engine.py -v`
Expected: FAIL — `make_key()` got an unexpected keyword `tools` / `chat()` has no `tools`.

- [ ] **Step 3: Add `tools` to the cache key**

In `src/agents7/llm/cache.py`, add `tools: list[dict] | None = None` as the last param of `_CacheKeyBuilder.make_key`, and include it in the payload only when set (preserving existing keys):

```python
        payload = {
            "model_id": model_id,
            "method": method,
            "messages": messages,
            "prompt": prompt,
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
            "output_format": output_format,
            "think": think,
        }
        if tools is not None:
            payload["tools"] = tools
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()
```

- [ ] **Step 4: Pass `tools` through the dispatcher cache check**

In `src/agents7/llm/dispatcher.py`, in `_cache_check`, add `tools` to the `_cache_lookup` call:

```python
    def _cache_check(
        self,
        method_name: str,
        input_kwarg: str,
        input_val,
        **common,
    ) -> tuple[str | None, object | None]:
        return self._cache_lookup(
            method_name,
            common["temperature"],
            **{input_kwarg: input_val},
            seed=common["seed"],
            max_tokens=common["max_tokens"],
            output_format=common["output_format"],
            think=common["think"],
            tools=common.get("tools"),
        )
```

And in `_cache_lookup`, add `tools=None` to the signature and pass it to `make_key`:

```python
    def _cache_lookup(
        self,
        method: str,
        temperature: float,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> tuple[str | None, object | None]:
        if self._cache is None or temperature != 0:
            return None, None
        key = ResponseCache.make_key(
            model_id=self._model_id,
            method=method,
            temperature=temperature,
            tools=tools,
            **kwargs,
        )
        return key, self._cache.get(key)
```

- [ ] **Step 5: Add `tools` + delegation to `LLMEngine`**

In `src/agents7/llm/base.py`, add `tools: list[dict] | None = None` to `LLMEngine.chat` and `LLMEngine.chat_async` (last param), and add `tools=tools` to their `self._dispatcher.execute(...)` / `execute_async(...)` calls. Then add two delegating methods after `supports_warmup`:

```python
    def supports_tools(self) -> bool:
        """Whether the underlying provider supports tool calling."""
        return self.llm_provider.supports_tools()

    def build_tool_turn_messages(
        self,
        assistant_text: str,
        tool_calls: list[dict],
        results: list[str],
    ) -> list[dict]:
        """Delegate to the provider's tool-turn message builder.

        See `LLMProvider.build_tool_turn_messages`.
        """
        return self.llm_provider.build_tool_turn_messages(
            assistant_text, tool_calls, results
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_tool_engine.py tests/test_cache.py tests/test_dispatcher.py tests/test_llm_engine.py -q`
Expected: PASS (new tests + no regressions in cache/dispatcher/engine suites).

- [ ] **Step 7: Commit**

```bash
git add src/agents7/llm/base.py src/agents7/llm/dispatcher.py src/agents7/llm/cache.py tests/test_tool_engine.py
git commit -m "feat: thread tools through LLMEngine, dispatcher and cache key"
```

---

### Task 5: ToolAgent + test helpers + YAML loader + exports

**Files:**
- Create: `src/agents7/agents/tool_agent.py`
- Modify: `src/agents7/agents/config_loader.py`
- Modify: `src/agents7/agents/__init__.py`
- Modify: `tests/helpers.py`
- Test: `tests/test_tool_agent.py`

**Interfaces:**
- Consumes: `ToolRegistry` (Task 2), `LLMEngine.supports_tools` / `build_tool_turn_messages` / `chat` with `tools=` (Task 4), `ToolLoopError` / `AgentConfigurationError`.
- Produces:
  - `ToolAgent(name, version, engine, system_prompt, *, tools=[], max_iterations=5, temperature=0.7)` with `act(user_input) -> AgentResponse`, `act_async(user_input) -> AgentResponse`, and `from_yaml(file_path) -> ToolAgent`.
  - Test helpers `FakeToolProvider`, `make_tool_engine(responses)`, `tool_call_response(name, arguments, text="")`.
  - Loader `load_tool_agent_yaml_settings(file_path) -> dict`.

- [ ] **Step 1: Add test helpers**

Append to `tests/helpers.py`:

```python
# --- Tool-agent test helpers ---


from agents7.llm.base import LLMProvider  # noqa: E402


class FakeToolProvider(LLMProvider):
    """A scripted tool-capable provider for ToolAgent tests.

    ``chat`` / ``chat_async`` pop pre-built response dicts in
    order, so a test can script a tool-call turn followed by a
    final answer with zero real LLM calls.
    """

    def __init__(self, responses: list[dict]):
        super().__init__("ollama", "fake")
        self._responses = list(responses)

    def supports_tools(self) -> bool:
        return True

    def build_tool_turn_messages(self, assistant_text, tool_calls, results):
        return [
            {
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": tool_calls,
            },
            *[
                {"role": "tool", "content": r, "tool_name": c["name"]}
                for c, r in zip(tool_calls, results)
            ],
        ]

    def chat(self, messages, temperature=0.7, max_tokens=-1,
             output_format=None, think=False, seed=None, tools=None):
        return self._responses.pop(0)

    async def chat_async(self, messages, temperature=0.7, max_tokens=-1,
                         output_format=None, think=False, seed=None,
                         tools=None):
        return self._responses.pop(0)

    def chat_stream(self, *args, **kwargs):
        raise NotImplementedError

    async def chat_stream_async(self, *args, **kwargs):
        raise NotImplementedError


def make_tool_engine(responses: list[dict]):
    """Build an LLMEngine backed by a `FakeToolProvider`."""
    engine = object.__new__(LLMEngine)
    provider = FakeToolProvider(responses)
    engine.provider = "ollama"
    engine.model_name = "fake"
    engine.model_id = "ollama::fake"
    engine.retry_config = None
    engine.cost_per_1m_tokens = None
    engine.llm_provider = provider
    engine._dispatcher = EngineDispatcher(
        primary=provider,
        fallback=None,
        retry_config=None,
        cache=None,
        cost_per_1m_tokens=None,
        model_id="ollama::fake",
    )
    return engine


def tool_call_response(name: str, arguments: dict, text: str = "") -> dict:
    """Build a chat result dict that requests one tool call."""
    return {
        "role": "assistant",
        "response": text,
        "thinking": None,
        "tool_calls": [{"name": name, "arguments": arguments}],
        "metrics": DUMMY_METRICS,
    }
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_tool_agent.py
"""Unit tests for agents7.agents.tool_agent.ToolAgent."""

import pytest

from helpers import (
    ENGINE,
    chat_response_dict,
    make_tool_engine,
    tool_call_response,
)

from agents7.agents.tool_agent import ToolAgent
from agents7.exceptions import AgentConfigurationError, ToolLoopError


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _agent(responses, **kwargs):
    return ToolAgent(
        name="calc",
        version=1,
        engine=make_tool_engine(responses),
        system_prompt="You can add.",
        tools=[add],
        **kwargs,
    )


def test_returns_final_answer_without_tools():
    agent = _agent([chat_response_dict("4")])
    out = agent.act("hi")
    assert out.response == "4"


def test_executes_tool_then_returns_answer():
    agent = _agent([
        tool_call_response("add", {"a": 1, "b": 2}),
        chat_response_dict("The sum is 3"),
    ])
    out = agent.act("add 1 and 2")
    assert out.response == "The sum is 3"


def test_metrics_accumulate_across_loop():
    agent = _agent([
        tool_call_response("add", {"a": 1, "b": 2}),
        chat_response_dict("3"),
    ])
    out = agent.act("add")
    # DUMMY_METRICS.tokens == 10 per call, two calls.
    assert out.metrics.tokens == 20
    assert out.metrics.prompt_tokens == 10


def test_max_iterations_raises():
    # Every turn asks for a tool again → never converges.
    responses = [tool_call_response("add", {"a": 1, "b": 1}) for _ in range(5)]
    agent = _agent(responses, max_iterations=3)
    with pytest.raises(ToolLoopError, match="exceeded 3"):
        agent.act("loop forever")


def test_non_tool_provider_rejected():
    with pytest.raises(AgentConfigurationError, match="does not support"):
        ToolAgent(
            name="x", version=1, engine=ENGINE,
            system_prompt="p", tools=[add],
        )


async def test_act_async_executes_tool():
    agent = _agent([
        tool_call_response("add", {"a": 2, "b": 2}),
        chat_response_dict("4"),
    ])
    out = await agent.act_async("add")
    assert out.response == "4"
```

Note: `ENGINE` in `helpers.py` is a real Ollama-provider engine whose `supports_tools()` returns `True`. To make `test_non_tool_provider_rejected` meaningful, use a provider that does not support tools. Replace its engine with one whose provider reports `False`:

In the test, build a non-tool engine inline instead of `ENGINE`:

```python
from helpers import make_engine


def test_non_tool_provider_rejected():
    engine = make_engine()
    engine.llm_provider.supports_tools.return_value = False
    with pytest.raises(AgentConfigurationError, match="does not support"):
        ToolAgent(
            name="x", version=1, engine=engine,
            system_prompt="p", tools=[add],
        )
```

(Use this version; drop the `ENGINE` import.)

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_tool_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: agents7.agents.tool_agent`.

- [ ] **Step 4: Write `tool_agent.py`**

```python
"""An agent that calls registered tools in a bounded loop."""

from dataclasses import dataclass, field
from typing import Any

from agents7.agents.base import Agent, BaseAgent
from agents7.exceptions import AgentConfigurationError, ToolLoopError
from agents7.models import AgentResponse
from agents7.models.base import LLMMetrics
from agents7.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ToolAgent(BaseAgent):
    """An agent that can call Python functions to answer.

    Runs a bounded call -> execute-tool -> call loop: the model
    may request tool calls, which are executed and fed back, until
    it returns a final answer or ``max_iterations`` is hit.

    :param tools: Callables or `Tool` instances the agent may call.
    :param max_iterations: Maximum model turns before raising
        `ToolLoopError`.
    :param temperature: Sampling temperature for each model turn.
    """

    tools: list[Any] = field(default_factory=list, kw_only=True)
    max_iterations: int = field(default=5, kw_only=True)
    temperature: float = field(default=0.7, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.engine.supports_tools():
            raise AgentConfigurationError(
                f"Provider `{self.engine.provider}` does not "
                f"support tools."
            )
        object.__setattr__(self, "_registry", ToolRegistry(self.tools))

    @classmethod
    def from_yaml(cls, file_path: str) -> Agent:
        """Build a `ToolAgent` from a YAML config file.

        :param file_path: Path to the YAML configuration.
        :return: A configured `ToolAgent`.
        """
        from agents7.agents.config_loader import (
            load_tool_agent_yaml_settings,
        )

        return cls(**load_tool_agent_yaml_settings(file_path))

    def act(self, user_input: str) -> AgentResponse:
        """Answer *user_input*, calling tools as needed.

        :param user_input: The user's message.
        :return: An `AgentResponse` with the final text and metrics
            summed across every model turn.
        :raises ToolLoopError: If the loop exceeds
            ``max_iterations``.
        """
        messages = self._seed_messages(user_input)
        totals = self._new_totals()
        for _ in range(self.max_iterations):
            result = self.engine.chat(
                messages,
                temperature=self.temperature,
                tools=self._registry.schemas(),
            )
            self._accumulate(totals, result["metrics"])
            calls = result.get("tool_calls") or []
            if not calls:
                return self._final(result, totals)
            messages += self._run_tools(result, calls)
        raise self._loop_error()

    async def act_async(self, user_input: str) -> AgentResponse:
        """Async version of `act`."""
        messages = self._seed_messages(user_input)
        totals = self._new_totals()
        for _ in range(self.max_iterations):
            result = await self.engine.chat_async(
                messages,
                temperature=self.temperature,
                tools=self._registry.schemas(),
            )
            self._accumulate(totals, result["metrics"])
            calls = result.get("tool_calls") or []
            if not calls:
                return self._final(result, totals)
            messages += self._run_tools(result, calls)
        raise self._loop_error()

    def _seed_messages(self, user_input: str) -> list[dict]:
        return [
            {"role": "system", "content": self.resolved_system_prompt},
            {"role": "user", "content": user_input},
        ]

    def _run_tools(self, result: dict, calls: list[dict]) -> list[dict]:
        results = [
            self._registry.execute(call["name"], call["arguments"])
            for call in calls
        ]
        return self.engine.build_tool_turn_messages(
            result["response"], calls, results
        )

    def _final(self, result: dict, totals: dict) -> AgentResponse:
        return AgentResponse(
            response=result["response"],
            model=None,
            thinking=result.get("thinking"),
            metrics=LLMMetrics(**totals),
        )

    def _loop_error(self) -> ToolLoopError:
        return ToolLoopError(
            f"ToolAgent `{self.name}` exceeded "
            f"{self.max_iterations} iterations."
        )

    @staticmethod
    def _new_totals() -> dict:
        return {"tokens": 0, "prompt_tokens": 0, "total_duration": 0.0}

    @staticmethod
    def _accumulate(totals: dict, metrics: dict) -> None:
        totals["tokens"] += metrics.get("tokens") or 0
        totals["prompt_tokens"] += metrics.get("prompt_tokens") or 0
        totals["total_duration"] += metrics.get("total_duration") or 0.0
```

- [ ] **Step 5: Add the YAML loader**

In `src/agents7/agents/config_loader.py`, append:

```python
def load_tool_agent_yaml_settings(file_path: str) -> dict:
    """Load `ToolAgent` settings from a YAML file.

    Schema: the Chatter base keys (``name``, ``version``,
    ``engine``, ``system_prompt``) plus an optional ``tools`` list
    of import paths and optional ``max_iterations`` / ``temperature``.

    :param file_path: Path to the YAML configuration file.
    :return: A kwargs dict for `ToolAgent`.
    """
    config = _resolve_inheritance(file_path)
    validate_chatter_yaml_config(config, file_path)
    tools = [import_from_path(p) for p in config.get("tools", [])]
    result = {
        "name": config["name"],
        "version": config["version"],
        "engine": build_engine(config["engine"]),
        "system_prompt": config["system_prompt"].strip(),
        "tools": tools,
    }
    cast_optional_fields(
        config,
        result,
        {"max_iterations": int, "temperature": float},
    )
    return result
```

- [ ] **Step 6: Export `ToolAgent`**

In `src/agents7/agents/__init__.py`, add the import (alphabetical, after `streaming`):

```python
from agents7.agents.tool_agent import ToolAgent
```

and add `"ToolAgent"` to `__all__` (alphabetical position).

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_tool_agent.py -v`
Expected: PASS (6 tests, including the async one).

- [ ] **Step 8: Commit**

```bash
git add src/agents7/agents/tool_agent.py src/agents7/agents/config_loader.py src/agents7/agents/__init__.py tests/helpers.py tests/test_tool_agent.py
git commit -m "feat: add ToolAgent with bounded tool-calling loop"
```

---

### Task 6: Integration test, full suite, docs, release

**Files:**
- Create: `integration_tests/test_tool_agent.py`
- Modify: `README.md`
- Modify: `pyproject.toml` (version)
- Modify: `changelog.md`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the integration test**

```python
# integration_tests/test_tool_agent.py
"""End-to-end ToolAgent test against a real Ollama model."""

from agents7.agents.tool_agent import ToolAgent
from agents7.llm import LLMEngine


def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"It is 18 degrees and sunny in {city}."


def test_tool_agent_calls_weather_tool():
    agent = ToolAgent(
        name="weather-bot",
        version=1,
        engine=LLMEngine(provider="ollama", model_name="qwen3"),
        system_prompt=(
            "You answer weather questions. Use the get_weather "
            "tool when asked about a city's weather, then report "
            "the result to the user."
        ),
        tools=[get_weather],
    )
    out = agent.act("What's the weather in Paris?")
    assert "18" in out.response or "sunny" in out.response.lower()
    assert out.metrics.tokens > 0
```

Note: `integration_tests/conftest.py` already skips when Ollama is unreachable — confirm the same skip fixture applies here (mirror the marker/fixture used by `integration_tests/test_classifier.py`).

- [ ] **Step 2: Verify the integration test (if Ollama is available)**

Run: `uv run pytest integration_tests/test_tool_agent.py -v`
Expected: PASS if Ollama + `qwen3` (a tool-capable model) are available; SKIPPED otherwise. If the model ignores the tool, switch `model_name` to a known tool-capable local model.

- [ ] **Step 3: Run the full unit suite**

Run: `uv run pytest tests/ -q`
Expected: PASS (entire suite green).

- [ ] **Step 4: Lint**

Run: `uv run ruff check src/ tests/`
Expected: no errors. Fix any 78-char or import issues inline.

- [ ] **Step 5: Update README**

Add a "Tool calling" section to `README.md` with a runnable example:

````markdown
## Tool calling

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

Tool calling currently requires the Ollama provider. The agent
raises `AgentConfigurationError` if the engine's provider does not
support tools, and `ToolLoopError` if the loop exceeds
`max_iterations`.
````

- [ ] **Step 6: Reinstall, bump version, changelog**

Run from `agents7_framework/`: `uv pip install -e ".[dev]"` then `uv sync --all-extras` (restore editable install with `uv pip install -e ".[dev]"` if `uv run` breaks afterward).

Bump the minor version in `pyproject.toml` (new public API). Add the changelog entry via the script:

```bash
scripts/changelog-add.sh v<NEW_VERSION> <<'EOF'
- Added tool / function calling (Ollama provider):
  - `agents7.tools`: `Tool`, `ToolRegistry`, `schema_from_callable`.
  - `ToolAgent`: bounded call -> tool -> call loop with `act()` /
    `act_async()` / `from_yaml()`.
  - Added `tools=` to `LLMEngine.chat` / `chat_async` and the
    `tool_calls` field to the normalized result dict.
  - Added `LLMProvider.supports_tools()` and
    `build_tool_turn_messages()`; wired for Ollama.
  - Added `ToolSchemaError` and `ToolLoopError`.
EOF
```

(If `scripts/changelog-add.sh` is absent, prepend by hand per `changelog-entries.md` — anchor the edit on the title line, never on an existing `### v` header.)

- [ ] **Step 7: Commit and push**

```bash
git add -A
git commit -m "feat: tool/function calling for agents (vX.Y.Z)"
git push
```

---

## Self-Review

**Spec coverage:**
- Tool abstraction (`Tool`, `schema_from_callable`, `ToolRegistry`) → Tasks 1–2. ✓
- Result-dict `tool_calls` + `tools=` param + provider hooks + Ollama wiring → Task 3. ✓
- Engine/dispatcher/cache threading → Task 4. ✓
- `ToolAgent` (sync+async, bounded loop, metric accumulation, `from_yaml`) → Task 5. ✓
- Exceptions `ToolSchemaError` / `ToolLoopError` → Tasks 1 & 5 usage. ✓
- Unit tests (schema, registry, fake-provider-driven agent loop) + integration test → Tasks 1,2,5,6. ✓
- README + changelog + minor bump → Task 6. ✓
- Backward compatibility (additive `tool_calls`/`tools`, cache key stable when no tools) → Tasks 3,4 + `make_key` test. ✓

**Deviations from spec (intentional, noted):**
- `tool_calls` are plain `{"name","arguments"}` dicts, not a `ToolCall` dataclass — required for `SqliteCache` JSON serialization at `temperature=0`.
- `tools=None` added to all three providers' `chat` signatures (Anthropic/OpenAI accept-and-ignore) so the generic `LLMEngine.chat` can always forward it.

**Type consistency:** `supports_tools`, `build_tool_turn_messages(assistant_text, tool_calls, results)`, `schemas()`, `execute(name, arguments)`, and the `{"name","arguments"}` tool-call shape are used identically across Tasks 2–5. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code.
