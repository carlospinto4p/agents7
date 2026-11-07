# LLM providers

## OpenAI-Compatible Provider

The library supports any server exposing the OpenAI chat/completions API (OpenAI, Azure OpenAI, vLLM, LM Studio, Ollama in OpenAI mode, etc.). Install the optional `openai` package:

```bash
uv pip install agents7[openai]
```

**Python configuration:**

```python
from agents7.llm import LLMEngine

# OpenAI
engine = LLMEngine(
    provider="openai_compat",
    model_name="gpt-4o",
    api_key="sk-...",
)

# Local vLLM / LM Studio
engine = LLMEngine(
    provider="openai_compat",
    model_name="meta-llama/Llama-3-8b",
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)
```

**YAML configuration:**

```yaml
engine:
    provider: openai_compat
    model: gpt-4o
    base_url: http://localhost:8000/v1
    api_key: sk-...
```

**Environment-variable configuration (external providers, e.g. DeepSeek):**

To point the provider at an external OpenAI-compatible API without hardcoding the model, URL, or key, set these environment variables:

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENAI_COMPAT_MODEL_NAME` | Model identifier | `deepseek-chat` |
| `OPENAI_COMPAT_BASE_URL` | API base URL | `https://api.deepseek.com` |
| `OPENAI_COMPAT_API_KEY` | API key | `sk-...` |

Any constructor argument left unset falls back to the matching variable, so the engine can be built with just the provider name:

```bash
export OPENAI_COMPAT_MODEL_NAME=deepseek-chat
export OPENAI_COMPAT_BASE_URL=https://api.deepseek.com
export OPENAI_COMPAT_API_KEY=sk-...
```

```python
from agents7.llm import LLMEngine

# Model, base_url and api_key all come from the environment.
engine = LLMEngine(provider="openai_compat")
```

The three variables are also exposed as an `OpenAICompatSettings` dataclass (`from agents7.settings import OpenAICompatSettings`), read at instantiation time.

**Precedence** (per field): explicit argument → `OPENAI_COMPAT_*` variable → the `openai` SDK's own fallback (`OPENAI_API_KEY` / `OPENAI_BASE_URL`). A missing model raises `ValueError`. All agent features (streaming, caching, retry, fallback, batch) work identically with the OpenAI-compatible provider.

## Anthropic/Claude Provider

Native support for Claude models using the official `anthropic` SDK.
Install the optional `anthropic` extra:

```bash
uv pip install agents7[anthropic]
```

```python
from agents7.llm import LLMEngine

engine = LLMEngine(
    provider="anthropic",
    model_name="claude-sonnet-4-20250514",
    # api_key falls back to ANTHROPIC_API_KEY env var
)

# With thinking mode and fallback
engine = LLMEngine(
    provider="anthropic",
    model_name="claude-sonnet-4-20250514",
    fallback_model="claude-haiku-4-5-20251001",
    default_max_tokens=8192,
)
```

YAML configuration:

```yaml
engine:
    provider: anthropic
    model: claude-sonnet-4-20250514
```

Handles system messages correctly (extracted as a separate `system` parameter per the Anthropic API). Extended thinking maps to Anthropic's native `thinking` parameter with `budget_tokens`.

## Model Selection

Automatically find the smallest Ollama model that passes all your test cases. This avoids manual trial-and-error when choosing between models of different sizes.

```python
from agents7.model_selection import find_best_model, TestCase

# Define test cases covering every expected output
cases = [
    TestCase(
        input={"text": "It is -10 degrees outside."},
        expected={"reason": "TOO_COLD"},
    ),
    TestCase(
        input={"text": "The temperature is 35 degrees."},
        expected={"reason": "TOO_HOT"},
    ),
    TestCase(
        input={"text": "She bought a new book."},
        expected={"reason": "NOT_APPLICABLE"},
    ),
]

# Find the smallest model that passes all cases
result = find_best_model(
    "configs/demo_temperature_detector.yml",
    test_cases=cases,
)

print(result.best_model)       # e.g. "qwen2.5:3b"
print(result.best_model_size)  # size in bytes
for score in result.scores:
    print(f"{score.model_name}: {score.pass_rate:.0%}")
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `yaml_path` | — | Path to the agent's YAML config |
| `test_cases` | — | List of `TestCase(input, expected)` |
| `models` | `None` | Candidate model names (None = all local) |
| `runs` | `1` | Runs per test case (for reliability) |
| `include_batch` | `False` | Also test `act_in_batch` |
| `stop_on_first` | `True` | Stop after first passing model |

Models are tested smallest-first. Only the fields present in `expected` are checked — enum fields are compared by name, other fields by value.

## Fallback Model

**Retrying never changes the model.** A retry re-issues the same call to the same model on the same provider — that is all `RetryConfig` does. The only thing that puts a different model in front of a request is `fallback_model`, and it is opt-in: an engine that does not name one has no fallback, and nothing substitutes a model behind your back.

When a `fallback_model` *is* named and the primary fails (after its retries), the engine tries the call again with that model. If the fallback also fails, the primary error is raised.

**A fallback answer says so.** `metrics.model` carries the fallback's name rather than the configured one, and so does `CallRecord.model` in the call log — see [Which model answered](operations.md#which-model-answered). The engine's own `model_id` stays the primary's, because that is what the agent was configured with; reading only that is how a run half-served by a fallback used to be recorded entirely against the model that failed.

`fallback_model` is a different **model**, not a different **provider**: the engine builds it on the same provider as the primary. It carries a model name and nothing else, so the call is priced at the primary's rates — right when the two models bill alike, wrong when they do not.

**YAML configuration:**

```yaml
engine:
    provider: ollama
    model: gemma3:12b
    fallback_model: qwen2.5:3b
```

**Python configuration:**

```python
from agents7.llm import LLMEngine

engine = LLMEngine(
    provider="ollama",
    model_name="gemma3:12b",
    fallback_model="qwen2.5:3b",
)
```

### Falling back to another provider

`fallback_engine` names a fallback as **a whole engine of its own** — its own `provider`, `base_url`, `api_key`, `pricing`, `timeout` and `retry`. That is what moves a call from a paid endpoint to a local one, which is the case worth having: an outage or a rate limit at a paid provider is exactly when a local model rescues a run, and a same-provider fallback cannot help with either.

**YAML configuration:**

```yaml
engine:
    provider: openai_compat
    model: deepseek-chat
    base_url: $OPENAI_COMPAT_BASE_URL
    api_key: $OPENAI_COMPAT_API_KEY
    pricing:
        input: 0.28
        output: 0.42
    fallback_engine:
        provider: ollama
        model: gemma3:12b
        pricing:
            input: 0.0
            output: 0.0
        retry:
            max_retries: 1
```

**Python configuration:**

```python
from agents7.llm import LLMEngine, Pricing, RetryConfig

engine = LLMEngine(
    provider="openai_compat",
    model_name="deepseek-chat",
    pricing=Pricing(input=0.28, output=0.42),
    fallback_engine=LLMEngine(
        provider="ollama",
        model_name="gemma3:12b",
        pricing=Pricing.flat(0.0),
        retry=RetryConfig(max_retries=1),
    ),
)
```

**A response the fallback answered is priced at the fallback's rates.** A run rescued by a free local model reports what it actually cost, rather than what the paid primary would have charged for the same tokens. An unpriced fallback reports `cost_cents` as `None`, never as `0.0` — see [Cost Estimation](operations.md#cost-estimation).

**The fallback's `retry` is its own budget.** The primary's attempts and the fallback's are still counted together in `metrics.retries`, because they were all made to produce the one response.

Two rules on the shape:

- **`fallback_model` and `fallback_engine` are mutually exclusive.** They mean different things, so naming both raises `ValueError`.
- **A `fallback_engine` must not declare a `cache`.** One cache covers the call whichever model answers it, and it belongs to the primary — a second one would never be read, so it is rejected rather than silently ignored.

## Thinking Mode

Thinking mode (`think`) lets the model reason before answering. It defaults to `False` for all agents, since constrained extraction tasks rarely benefit from reasoning (see [model selection](model-selection.md)). Enable it per agent when the task genuinely requires multi-step inference.

**YAML configuration:**

```yaml
think: true
```

**Python configuration:**

```python
agent = Labeler(
    name="ReasoningExtractor",
    version=1,
    engine=engine,
    system_prompt="Extract person info.",
    output_model=PersonInfo,
    think=True,
)
```

The `think` setting can still be overridden per call via `chat_kwargs={"think": True}`.

## Streaming

LLM responses can be streamed token-by-token using the `_stream` methods. Streaming bypasses retry logic (a connection drop mid-stream cannot be retried meaningfully).

**Sync streaming:**

```python
from agents7.llm import LLMEngine

engine = LLMEngine(provider="ollama", model_name="qwen3")

stream = engine.chat_stream(
    messages=[{"role": "user", "content": "Tell me a story."}],
)
for chunk in stream:
    print(chunk, end="", flush=True)

print()
print(f"Tokens: {stream.metrics['tokens']}")
print(f"Full response: {stream.response}")
```

**Async streaming:**

```python
stream = await engine.chat_stream_async(
    messages=[{"role": "user", "content": "Tell me a story."}],
)
async for chunk in stream:
    print(chunk, end="", flush=True)

print(stream.response)
```

After iteration, `StreamResult` / `AsyncStreamResult` expose:
- `.response` — full accumulated text
- `.thinking` — accumulated thinking text (not yielded as chunks), or `None`
- `.role` — message role (chat only), or `None`
- `.metrics` — dict carrying the same fields a non-streamed call
  reports: `tokens`, `prompt_tokens`, `cached_prompt_tokens`,
  `total_duration`, `cost_cents`, `cached`, `retries` and
  `retry_duration`

The `generate_stream` and `generate_stream_async` methods work identically but accept a `prompt` string instead of `messages`.

### Agent-Level Streaming

Agents expose `act_stream()` and `act_stream_async()` on `Labeler` (and all subclasses). These yield text chunks during generation and parse the structured output after the stream completes.

```python
from agents7.agents import Labeler

agent = Labeler.from_yaml("configs/demo_labeler.yml")

# Sync streaming
stream = agent.act_stream({"text": "Sarah, 28, from London."})
for chunk in stream:
    print(chunk, end="", flush=True)

print()
response = stream.agent_response   # AgentResponse with parsed model
print(response.model)
```

```python
# Async streaming
stream = await agent.act_stream_async({"text": "Sarah, 28."})
async for chunk in stream:
    print(chunk, end="", flush=True)

print(stream.agent_response.model)
```

The `agent_response` property is `None` before iteration completes. After iteration, it contains the full `AgentResponse` with the parsed Pydantic model (or `model=None` if parsing failed).

Iteration is also when a streamed call reaches the [call
log](operations.md#structured-call-logging), for the same reason: the outcome is not known
until the accumulated text has been parsed. Before v4.24.0 a
streamed call was never logged at all, so a run made entirely of
streamed calls wrote an empty log file.

## Retry / Resilience

LLM transport errors (connection failures, server 5xx) can be retried automatically with exponential backoff. Client errors (4xx) are **not** retried. Parse failures can be retried separately via `max_parse_retries` (see below).

**YAML configuration:**

```yaml
name: MyAgent
version: 1
engine:
    provider: ollama
    model: qwen3
    retry:
        max_retries: 3
        base_delay: 1.0
        max_delay: 30.0
        backoff_factor: 2.0
system_prompt: Analyze this text.
output_model: mymodule.MyOutput
```

All `retry` fields are optional and fall back to sensible defaults. An empty `retry: {}` enables retries with defaults.

**Omitting the `retry` key does not disable retrying.** It selects `DEFAULT_RETRY` — two retries, three attempts in all. Before v5.3.0 it meant no framework retry, and the `openai` and `anthropic` clients then retried twice by themselves, so a config that declared neither `retry` nor `timeout` had three attempts made on its behalf while `LLMMetrics.retries` reported zero and `retry_duration` reported nothing. Both SDKs are now switched off in every case and the default takes their place, so the same attempts are made by a layer that counts and times them.

To disable retrying, say so:

```yaml
engine:
    provider: ollama
    model: qwen3
    retry:
        max_retries: 0
```

**Python configuration:**

```python
from agents7.llm import LLMEngine, RetryConfig

engine = LLMEngine(
    provider="ollama",
    model_name="qwen3",
    retry=RetryConfig(max_retries=3, base_delay=1.0),
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_retries` | `3` | Maximum retry attempts (0 = disabled) |
| `base_delay` | `1.0` | Initial delay in seconds |
| `max_delay` | `30.0` | Delay cap in seconds |
| `backoff_factor` | `2.0` | Multiplier applied after each retry |
| `retry_connection_errors` | `False` | Retry a failure to reach a paid provider at all |

Delay formula: `min(base_delay * backoff_factor ^ attempt, max_delay)`.

**A failure to reach the provider at all is the one transient the
three providers treat differently, deliberately.** It carries no
status code, because nothing answered to give it one — so until
v5.5.0 the classifier's fall-through to `status_code` made
`openai.APIConnectionError` and `anthropic.APIConnectionError`
non-retryable, while the identical failure against Ollama was
retried unconditionally.

Ollama keeps that behaviour: it is a local server, an attempt costs
nothing but time, and the server restarting between two calls is the
ordinary case this covers. Reaching for a paid endpoint is now a
decision you make in the config:

```yaml
engine:
    provider: anthropic
    model: claude-sonnet-5
    retry:
        max_retries: 2
        retry_connection_errors: true
```

Worth knowing before setting it: a connection failure never reached
the provider, so **no tokens were billed** for the attempt that
failed. What a retry costs is the wait — a full backoff (1s + 2s
under `DEFAULT_RETRY`) before you learn an endpoint is simply down,
where today you learn immediately.

A **timed-out** call is retried whatever this is set to. Both SDKs
subclass their timeout error from their connection error, and the
classifier checks for a timeout first.

Those are the defaults of a `RetryConfig` you construct yourself. An engine given no `retry` at all gets `DEFAULT_RETRY` instead, which is the same thing with `max_retries=2` — the attempt count the provider SDKs used to make on their own:

```python
from agents7 import DEFAULT_RETRY

DEFAULT_RETRY.max_retries   # 2
```

## Request Timeout

`timeout` caps how long a single HTTP request to the provider may take
before it is abandoned. It is set on the provider's own client, so it
covers sync, async and streaming calls alike, and each retry attempt
gets its own fresh budget.

**YAML configuration:**

```yaml
engine:
    provider: openai_compat
    model: deepseek-chat
    timeout: 10        # seconds; omit for no limit
    retry:
        max_retries: 2
```

**Python configuration:**

```python
engine = LLMEngine(
    provider="openai_compat",
    model_name="deepseek-chat",
    timeout=10,
    retry=RetryConfig(max_retries=2),
)
```

**A timed-out call is retryable.** Running out of time is a stall, and
a second attempt is what a stall is for. With retries configured, a
call that would have hung indefinitely costs the timeout plus a retry
and still returns an answer; without them it raises.

**The SDK's own retrying is switched off in every case.** The
`openai` and `anthropic` clients both retry twice by default, which
made a declared limit mean three times what it says — measured
against a server that accepts and never answers, a 0.5s timeout took
3.39s — and hid those attempts from `LLMMetrics.retries`. Retry
belongs to `RetryConfig`, which counts what it does. Until v5.3.0 the
switch-off rode along with `timeout`, so a config that declared no
timeout kept the hidden layer; it is now unconditional, and
`DEFAULT_RETRY` takes that layer's place.

**Size it from measured latency, not from a guess**, and size it per
model and per provider — which is why there is no framework-wide
default. Read the durations out of a call log (see
[Structured Call Logging](operations.md#structured-call-logging)) and pick a figure
above the slowest real call. Two things move that figure:

- **How much the agent writes.** An agent that answers in ten tokens
  and one that answers in a thousand are not comparable. On a chat
  agent, which writes 10–80 tokens a turn, 11,036 DeepSeek calls ran a median
  of 1.42s and a 99.99th percentile of 5.19s, with a single call
  stalling at 603.57s and nothing at all in between. It ships
  `timeout: 10`.
- **Who serves the model.** A local Ollama model loads into memory
  inside its first call, and that call is far slower than every one
  after it. Measured across `store_assistant`'s call logs, the first
  call of each run took 17–28s against a median of 0.6–1.5s once
  loaded — so a limit sized for a hosted API would cut off the first
  call of every local run. Either measure the loaded and unloaded
  cases separately, or set the limit above the load.

## Parse Retry

When an LLM returns malformed JSON that can't be parsed, `max_parse_retries` re-prompts the model with the error message so it can self-correct. This is separate from transport retries and only applies to `act()` and `act_async()`.

**YAML configuration:**

```yaml
name: MyAgent
version: 1
engine:
    provider: ollama
    model: qwen3
system_prompt: Extract person info.
output_model: mymodule.PersonInfo
max_parse_retries: 2
```

**Python configuration:**

```python
agent = Labeler(
    name="ResilientExtractor",
    version=1,
    engine=engine,
    system_prompt="Extract person info.",
    output_model=PersonInfo,
    max_parse_retries=2,
)
```

On each retry, the conversation history is extended with the error feedback, allowing the model to see what went wrong and correct its output. If all retries fail, `model=None` is returned.

## Exception Hierarchy

The library provides structured exceptions for fine-grained error handling:

```
Agents7Exception
├── LLMException
│   ├── ParseError              — response parsing failed
│   └── ProviderError           — LLM provider returned an error
│       ├── ModelUnavailableError — model not found (404)
│       └── RateLimitError       — rate limit exceeded (429)
├── AgentConfigurationError
├── PipelineConfigurationError
├── PropertyConfigurationError
└── PromptParameterError
```

```python
from agents7.exceptions import (
    ProviderError,
    RateLimitError,
    ModelUnavailableError,
    ParseError,
)

try:
    response = agent.act({"text": "..."})
except RateLimitError:
    print("Rate limited — back off")
except ModelUnavailableError:
    print("Model not found")
except ProviderError:
    print("Other provider error")
```

Note: `act()` catches `ParseError` internally and returns `model=None` for backward compatibility. The exception is available for direct use with `ResponseParser.parse()`.

## Model Warm-Up

Ollama loads models lazily on first call, causing a cold-start delay. Use `warm_up()` to preload the model at startup:

```python
agent = Labeler.from_yaml("configs/my_agent.yml")
agent.warm_up()  # blocks until model is loaded

# First real call is fast
response = agent.act({"text": "..."})

# Async version
await agent.warm_up_async()
```

For non-Ollama providers (e.g. `openai_compat`), `warm_up()` is a no-op.

## Closing Async Clients

Each provider opens an async HTTP client. Left for the garbage
collector, it tries to close on a loop that has already ended, and
`asyncio.run` prints *"Task exception was never retrieved ...
RuntimeError: Event loop is closed"* after a run that otherwise
finished cleanly. Close it before the loop ends:

```python
async def main():
    agent = Chatter.from_yaml("configs/my_chatter.yml")
    try:
        await agent.chat_async("hola")
    finally:
        await agent.aclose()

asyncio.run(main())
```

`agent.aclose()` closes the agent's engine and those of any agent it
holds (a guard, a fact extractor), each once. `LLMEngine.aclose()`
closes the primary provider and its fallback. Nothing stops working:
each provider swaps in a fresh async client that connects on its next
call. `engine.check_available()` does this itself, since the loop it
runs its check in ends with the call.
