
# Agents7

A Python library for building agents that process and label
data using structured outputs (Pydantic models). Agents are
typically LLM-powered, but the framework's models and output
structure work equally well for rule-based or hybrid agents
where deterministic logic is faster or more reliable.

## Table of Contents

- [Installation](#installation)
- [Design](#design)
- [Architecture](#architecture)
  - [LLM Layer](#llm-layer-agents7llm)
  - [Agents Layer](#agents-layer-agents7agents)
  - [Models Layer](#models-layer-agents7models)
- [Quick Start](#quick-start)
  - [Using a Labeler](#using-a-labeler-entity-extraction)
  - [Using a Classifier](#using-a-classifier-sentiment-analysis)
  - [Using a Regressor](#using-a-regressor-value-extraction)
  - [Defining Custom Models](#defining-custom-models)
  - [Prompt Parameterization](#prompt-parameterization)
  - [Per-Conversation Context](#per-conversation-context)
  - [YAML Configuration](#yaml-configuration)
  - [Retry / Resilience](#retry--resilience)
  - [Request Timeout](#request-timeout)
  - [Exception Hierarchy](#exception-hierarchy)
  - [Fallback Model](#fallback-model)
    - [Falling back to another provider](#falling-back-to-another-provider)
  - [Thinking Mode](#thinking-mode)
  - [Parse Retry](#parse-retry)
  - [OpenAI-Compatible Provider](#openai-compatible-provider)
  - [Anthropic/Claude Provider](#anthropicclaude-provider)
  - [Model Selection](#model-selection)
  - [Streaming](#streaming)
    - [Agent-Level Streaming](#agent-level-streaming)
  - [Cost Estimation](#cost-estimation)
  - [Caching](#caching)
  - [Validation Hooks](#validation-hooks)
  - [Agent Pipelines](#agent-pipelines)
  - [Batch Concurrency Control](#batch-concurrency-control)
  - [Agent Introspection](#agent-introspection)
  - [Agent Cloning with Overrides](#agent-cloning-with-overrides)
  - [Batch Export](#batch-export)
  - [Evaluation Workflow](#evaluation-workflow)
  - [Scoring a Chat Agent's Style](#scoring-a-chat-agents-style)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Input Preprocessor](#input-preprocessor)
  - [Model Warm-Up](#model-warm-up)
  - [Structured Call Logging](#structured-call-logging)
  - [Personality Chatter](#personality-chatter-roleplay-agent)
  - [Scene Chatter](#scene-chatter-contextual-roleplay-agent)
  - [Character Facts](#character-facts-memory-across-conversations)
  - [Testing Utilities](#testing-utilities)
  - [Simple Classifier](#simple-classifier)
  - [Tool Calling](#tool-calling)
- [Examples](#examples)
- [Requirements](#requirements)
- [Model Compatibility](#model-compatibility)
- [Development](#development)
  - [Integration Test Models](#integration-test-models)

## Installation

The package is not on PyPI. Install a tagged release from this
repository:

```bash
pip install "agents7 @ git+https://github.com/carlospinto4p/agents7@v9.25.9"
```

The UI and the agents live in repositories of their own and depend on
a tag of this one; `docs/repository-split.md` explains the layout.
The `samples/` folder holds example agent configs, tested by
`tests/test_sample_golden_sets.py`.

Requires Python >= 3.11.

## Design

See [docs/design.md](docs/design.md) for the rationale behind the
library's architecture: why reasons are required, how NOT_APPLICABLE
works, the structured output flow, and other design decisions.

See [docs/model-selection.md](docs/model-selection.md) for guidance
on choosing the right LLM for structured extraction tasks.

## Architecture

### LLM Layer (`agents7.llm`)

- `LLMEngine`: Facade class that routes to specific providers (supports an optional `fallback_model` on the same provider, or a `fallback_engine` on any provider)
- `LLMProvider`: Abstract base class defining the interface
- `OllamaProvider`: Concrete implementation using Ollama (local LLM server)
- `AnthropicProvider`: Native Claude provider using the official `anthropic` SDK
- `OpenAICompatProvider`: Provider for any OpenAI-compatible API (OpenAI, Azure, vLLM, LM Studio, etc.)

Supports both sync and async methods: `chat`, `chat_async`, `generate`, `generate_async`, plus streaming variants: `chat_stream`, `chat_stream_async`, `generate_stream`, `generate_stream_async`.

Features: temperature control, structured output via JSON schema, "thinking" mode, seed for reproducibility, streaming responses, in-memory LRU caching, fallback model support.

### Agents Layer (`agents7.agents`)

```
BaseAgent
  ├── Labeler (any Pydantic model)
  │     └── Detector (Property outputs)
  │           ├── Classifier (ClassificationProperty)
  │           ├── Regressor (Property[T])
  │           └── ContextGuard (response validation)
  └── Chatter (multi-turn conversation; optional persona + scene)
        └── GuardedChatter (+ ContextGuard pipeline)
              └── GuardedSceneChatter (persona + scene + guard)

Pipeline (chains multiple agents sequentially)
```

- `BaseAgent`: Abstract base defining common interface (`name`, `version`, `engine`, `system_prompt`, `prompt_params`, `context_params`, `prompt_variants`)
- `Labeler`: General-purpose agent that outputs structured data via a Pydantic `output_model`
  - Single processing: `act()`, `act_async()`
  - Streaming: `act_stream()`, `act_stream_async()` (yields text chunks, then parses structured output)
  - Batch processing: `act_in_batch()`, `act_in_batch_async()` (with optional `BatchSettings` concurrency control)
  - YAML configuration support via `from_yaml()`
- `Detector`: Specialized `Labeler` for property extraction (value + reason)
- `Classifier`: For categorical classification with category + reason enums
- `Regressor`: For numeric value extraction with reason enum
- `Pipeline`: Chains multiple agents sequentially with optional custom mappers and conditional step execution (async batch supports `BatchSettings`). Use `Pipeline.from_agents([a1, a2])` for simple sequential chains without custom mappers

### Models Layer (`agents7.models`)

- `AgentResponse`: Wraps LLM response with parsed model, thinking, metrics, and index
- `LLMMetrics`: Token counts, duration, and estimated cost — see [Cost Estimation](#cost-estimation)
- `Property[T]`: Generic base for value extraction (value of type T + reason enum)
- `ClassificationProperty`: Base for classification tasks (category enum + reason enum)

Both `Property` and `ClassificationProperty` automatically add `NOT_APPLICABLE` to their enum fields by default. Set `NOT_APPLICABLE = False` on the subclass to disable this (e.g., when a NEUTRAL category already covers the "no match" case).

## Quick Start

### Using a Labeler (Entity Extraction)

```python
from agents7.agents import Labeler

extractor = Labeler.from_yaml("configs/demo_labeler.yml")

texts = [
    "Sarah, a 28-year-old data scientist from London.",
    "Dr. Chen has been practicing medicine in Singapore.",
]

# Use batch processing for multiple inputs
inputs = [{"text": text} for text in texts]
responses = extractor.act_in_batch(inputs)

for response in responses:
    print(response.model)
```

### Using a Classifier (Sentiment Analysis)

```python
from agents7.agents import Classifier

classifier = Classifier.from_yaml("configs/demo_sentiment_classifier.yml")

texts = [
    "I can't believe how amazing this experience was!",
    "This is the worst customer service ever.",
    "The report contains the quarterly sales figures.",
]

inputs = [{"text": text} for text in texts]
responses = classifier.act_in_batch(inputs)

for response in responses:
    print(f"Category: {response.model.category.name}")
    print(f"Reason: {response.model.reason.name}")
```

### Using a Regressor (Value Extraction)

```python
from agents7.agents import Regressor

regressor = Regressor.from_yaml("configs/demo_temperature_detector.yml")

texts = [
    "It's freezing outside at -5 degrees!",
    "The room is at a comfortable 22 degrees.",
]

inputs = [{"text": text} for text in texts]
responses = regressor.act_in_batch(inputs)

for response in responses:
    print(f"Value: {response.model.value}")
    print(f"Reason: {response.model.reason.name}")
```

### Defining Custom Models

**For Classification (Classifier):**

```python
from enum import StrEnum
from agents7.models import ClassificationProperty

class SentimentCategory(StrEnum):
    POSITIVE = "Positive sentiment expressed."
    NEGATIVE = "Negative sentiment expressed."
    NEUTRAL = "Neutral or objective tone."

class SentimentReason(StrEnum):
    ENTHUSIASTIC = "Expresses enthusiasm or excitement."
    GRATEFUL = "Shows gratitude or appreciation."
    FRUSTRATED = "Expresses frustration or annoyance."
    FACTUAL = "Primarily factual, objective tone."

class Sentiment(ClassificationProperty):
    category: SentimentCategory
    reason: SentimentReason

    # Define valid reasons for each category
    CATEGORY_REASONS = {
        SentimentCategory.POSITIVE: [
            SentimentReason.ENTHUSIASTIC,
            SentimentReason.GRATEFUL,
        ],
        SentimentCategory.NEGATIVE: [
            SentimentReason.FRUSTRATED,
        ],
        SentimentCategory.NEUTRAL: [
            SentimentReason.FACTUAL,
        ],
    }
```

To disable the automatic `NOT_APPLICABLE` member (e.g., when NEUTRAL already covers the "no match" case):

```python
class Sentiment(ClassificationProperty):
    NOT_APPLICABLE = False  # don't add NOT_APPLICABLE
    category: SentimentCategory
    reason: SentimentReason
```

This also works on `Property` subclasses for detectors/regressors.

**For Value Extraction (Regressor):**

```python
from enum import StrEnum
from agents7.models import Property

class TemperatureReason(StrEnum):
    COLD = "Temperature is cold (below 15C)."
    COMFORTABLE = "Temperature is comfortable (15-25C)."
    HOT = "Temperature is hot (above 25C)."

class Temperature(Property[float]):
    reason: TemperatureReason
```

### Prompt Parameterization

Use `$variable` placeholders in `system_prompt` and resolve them with `prompt_params`:

```python
from datetime import date
from agents7.agents import Labeler

agent = Labeler(
    name="DailyAnalyzer",
    version=1,
    engine=engine,
    system_prompt="Today is $today. Analyze data for $company.",
    output_model=MyOutput,
    prompt_params={
        "today": lambda: str(date.today()),  # evaluated each call
        "company": "Acme Corp",              # static value
    },
)
```

Unmatched `$placeholders` are left as-is (safe_substitute). Callable values are evaluated fresh each time the prompt is accessed.

### Per-Conversation Context

`prompt_params` holds values the config fixes once. `context_params` declares values that change per conversation — who the agent is talking to — or per message — the time of day. Both substitute the same `$variable` placeholders; only the source of the value differs.

```yaml
system_prompt: |
  Hablas con $nick. Es $moment.
context_params:
    nick:
        default: "alguien"
        description: The other person's nickname
    moment:
        callable: agents7.agents.context.spanish_time_of_day
```

```python
chatter = Chatter.from_yaml("config.yml").set_context(nick="lector42")
chatter.chat("hola")
# System prompt: "Hablas con lector42. Es por la noche."
```

A value resolves from `set_context()` first, then the declared `callable` (run again before every message), then `default` — so a declared name never reaches the model as the literal `$name`. A keyword that matches no declaration raises `PromptParameterError` rather than being silently dropped.

Supplied values follow a copy made with `with_overrides()`. They are not a dataclass field, so a bare `dataclasses.replace(agent, ...)` drops them and the copy answers with each default — copy an agent with `with_overrides()`.

A `GuardedSceneChatter`'s guard reads the same values. `agent.guard` keeps a scene's context names as placeholders, so it is identical for every conversation, and each reply is checked by `agent.current_guard()`, which fills them with what the agent's own prompt carries at that moment. To run the guard by hand, call `current_guard()` rather than `guard`.

`callable` is a dotted path to a zero-argument function, so YAML configs are no longer limited to static values. Three clock helpers ship with the framework: `agents7.agents.context.time_of_day`, `spanish_time_of_day` and `spanish_weekday`.

A value can be worked out from another one. `from` names the declared param it reads, and its `callable` then takes that param's current value as its one argument:

```yaml
context_params:
    nick:
        default: "alguien"
    nick_hint:
        callable: mypackage.hints.read_nick
        from: nick
```

```python
chatter.set_context(nick="lector42")
# $nick_hint is now read_nick("lector42"), with nothing else to call
```

Supplying the source is enough, so the two cannot fall out of step; a value passed to `set_context()` for the derived name still wins. `from` must name a declared param that is not itself derived, and needs a `callable`. In Python the same declaration is `ContextParam(name=..., resolver=..., derived_from="nick")`.

`resolved_context` returns what the declared params currently resolve to — the mapping the prompt itself is built from, so the two cannot drift. It exists because those values are otherwise unrecoverable: a `callable` runs inside substitution and its result is discarded, so nothing afterwards can say which time of day or which nickname a finished conversation ran with, and at temperature a re-run is a different conversation. A `ConversationRecorder` captures it once when the conversation opens and saves it in the recording, alongside the character facts the store already held; a read then shows both above the transcript. Reading it twice can give different values, since a resolver is called each time.

A context param's **declaration** enters `system_prompt_identity`, never its resolved value — the name, the default and where a `callable` comes from. A run's fingerprint hashes that identity, so an agent whose prompt carries the time of day keeps one measurement baseline instead of gaining a separate one for each part of the day, and a nickname a caller supplies is treated as input rather than as a different agent. Changing a `default` or swapping a `callable` does change the identity, because those are configuration.

### Prompt Variants

`prompt_variants` lists alternative texts for a placeholder. The agent picks one of them when it is built, so each conversation shows a single example and different conversations show different ones.

```yaml
system_prompt: |
  User: fine
  David: $fine_reply
prompt_variants:
    fine_reply:
        - "Fine as in good, or fine as in don't ask?"
        - "Fine is what everyone says. I'll take it."
```

Listing every alternative in a fixed prompt does not work: on a chat agent the model answered with whichever came first in 8, 9 and 10 conversations of 10.

- `variant_values` returns what was picked, and a `ConversationRecorder` saves it in the recording.
- A style contract keeps what each run picked in `StyleReport.variants`, and the history record in `run_variants`. `StyleRunRecord.by_variant(rule_id, name)` adds a rule's score up per text, to check whether a rate moved with what was shown rather than with the config.
- A `Chatter` picks again on `reset()`. With a memory backend it picks from `conversation_id`, so an agent rebuilt for each turn keeps the same texts.
- `draw_variants(seed=...)` repeats a pick exactly.
- `system_prompt_identity` is the prompt with the placeholders left in and every alternative listed. The style history hashes it, so runs that picked different texts still share a baseline. `context_params` are left in and declared after it for the same reason — see [Per-Conversation Context](#per-conversation-context).
- Each entry needs at least two different, non-empty texts, and a name cannot also be a `prompt_params` or `context_params` name.

### YAML Configuration

```yaml
base_config: configs/shared_base.yml  # optional: inherit from parent config
name: MyClassifier
version: 1
engine:
    provider: ollama
    model: qwen3
    fallback_model: qwen2.5:3b  # optional: retry with this model on failure
    timeout: 10  # optional: seconds per request (default: no limit)
think: true  # optional: enable thinking mode (default: false)
system_prompt: |
  Analyze the sentiment of text collected on $date.
prompt_params:
    date: "2026-02-01"
output_model: mymodule.Sentiment
system_examples:
    - input: "This is fantastic!"
      output:
        category: POSITIVE
        reason: ENTHUSIASTIC
```

Note: YAML `prompt_params` only supports static values (cast to strings). For a value computed fresh on every call, declare it under `context_params` with a `callable` (see [Per-Conversation Context](#per-conversation-context)), or pass a callable when instantiating in Python.

Note: structured `system_examples` inputs and outputs are rendered into the prompt as JSON, matching the JSON the model is asked to produce. Scalar values are rendered as-is.

**Config inheritance:** When `base_config` is specified, the base YAML is loaded first and the current file's keys override it. Nested dicts (e.g. `engine`) are merged recursively. Chains are supported (A inherits B inherits C) and circular references are detected.

### Command-Line Interface

Installing the package exposes an `agents7` console script with two
subcommands for running and scoring agents headlessly, without any UI.
`run` is thin glue over the YAML loaders; `eval` is a thin wrapper over
[`run_eval_suite`](#evaluation-workflow), the same composed route a
Python caller uses.

```bash
# Run a YAML-configured agent against a single input (inline JSON):
agents7 run --config config.yml --input '{"text": "I love this!"}'

# …or from a file, printing the full response (text, thinking, metrics):
agents7 run -c config.yml --input-file input.json --full

# Evaluate against a labeled golden set and print accuracy:
agents7 eval --config config.yml --data data/golden_set.jsonl

# …and write a Markdown report with a confusion matrix:
agents7 eval -c config.yml -d golden_set.csv --report report.md

# …logging every call made, summarised in the output and the report:
agents7 eval -c config.yml -d golden_set.csv -r report.md \
    --call-log calls.jsonl
```

The agent class is chosen from the config's `agent_type` field
(`labeler`, `detector`, `classifier`, `regressor`, or `tool`),
overridable with `--agent-type`. A `tool` agent is scored on its
**trajectory** rather than on output fields — see
[Evaluating Tool Agents](#evaluating-tool-agents). Golden sets may be
`.jsonl`, `.json`, or `.csv`,
and rows may be nested (`{"input": {...}, "category": "..."}`) or flat
(input and expected columns side by side); both are normalised
automatically. Expected fields are derived from the golden set and the
output model, or set explicitly with `--fields`.

The whole `eval` route is importable as one call — see
[Evaluation Workflow](#evaluation-workflow) — and its individual steps
stay public for finer control:

```python
from agents7.eval import build_agent, load_eval_rows, normalise_rows

agent = build_agent("config.yml", agent_type="classifier")
rows = load_eval_rows("data/golden_set.jsonl")
eval_rows, fields = normalise_rows(rows)
```

### Retry / Resilience

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

### Request Timeout

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
[Structured Call Logging](#structured-call-logging)) and pick a figure
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

### Exception Hierarchy

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

### Fallback Model

**Retrying never changes the model.** A retry re-issues the same call to the same model on the same provider — that is all `RetryConfig` does. The only thing that puts a different model in front of a request is `fallback_model`, and it is opt-in: an engine that does not name one has no fallback, and nothing substitutes a model behind your back.

When a `fallback_model` *is* named and the primary fails (after its retries), the engine tries the call again with that model. If the fallback also fails, the primary error is raised.

**A fallback answer says so.** `metrics.model` carries the fallback's name rather than the configured one, and so does `CallRecord.model` in the call log — see [Which model answered](#which-model-answered). The engine's own `model_id` stays the primary's, because that is what the agent was configured with; reading only that is how a run half-served by a fallback used to be recorded entirely against the model that failed.

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

#### Falling back to another provider

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

**A response the fallback answered is priced at the fallback's rates.** A run rescued by a free local model reports what it actually cost, rather than what the paid primary would have charged for the same tokens. An unpriced fallback reports `cost_cents` as `None`, never as `0.0` — see [Cost Estimation](#cost-estimation).

**The fallback's `retry` is its own budget.** The primary's attempts and the fallback's are still counted together in `metrics.retries`, because they were all made to produce the one response.

Two rules on the shape:

- **`fallback_model` and `fallback_engine` are mutually exclusive.** They mean different things, so naming both raises `ValueError`.
- **A `fallback_engine` must not declare a `cache`.** One cache covers the call whichever model answers it, and it belongs to the primary — a second one would never be read, so it is rejected rather than silently ignored.

### Thinking Mode

Thinking mode (`think`) lets the model reason before answering. It defaults to `False` for all agents, since constrained extraction tasks rarely benefit from reasoning (see [docs/model-selection.md](docs/model-selection.md)). Enable it per agent when the task genuinely requires multi-step inference.

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

### Parse Retry

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

### OpenAI-Compatible Provider

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

### Anthropic/Claude Provider

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

### Model Selection

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

### Streaming

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

#### Agent-Level Streaming

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
log](#structured-call-logging), for the same reason: the outcome is not known
until the accumulated text has been parsed. Before v4.24.0 a
streamed call was never logged at all, so a run made entirely of
streamed calls wrote an empty log file.

### Cost Estimation

Every response carries an `LLMMetrics`, and its `cost_cents` is `None` until the engine is told what the model costs. Declare `pricing` and every call estimates its own cost.

Rates are **US dollars per million tokens**, which is how providers publish them — copy the published number straight in:

```yaml
engine:
    provider: openai_compat
    model: deepseek-chat
    pricing:
        input: 0.44          # prompt tokens read fresh
        output: 1.32         # generated tokens
        cached_input: 0.014  # prompt tokens served from the provider's cache
```

Or in Python:

```python
from agents7.llm import LLMEngine
from agents7.llm.pricing import Pricing

engine = LLMEngine(
    "openai_compat",
    "deepseek-chat",
    pricing=Pricing(input=0.44, output=1.32, cached_input=0.014),
)
```

**`cached_input` is the one that matters for a chat agent.** A persona chatter resends its whole system prompt — persona, scene, style examples, remembered facts — on every turn, so after the first turn nearly all of it is a prompt-cache hit. Measured on a chat agent: 6,144 of 6,239 prompt tokens, 98%. At the rates above a cache hit is **31× cheaper** than a fresh read, so pricing all input at the fresh rate overstates a turn by roughly 20×. `cached_input` defaults to `input`, which is correct for a provider that does not discount repeated prompts.

`cached_prompt_tokens` on the metrics reports how many prompt tokens were cache hits. It is a *subset* of `prompt_tokens`, on every provider — `openai_compat` reads OpenAI's `prompt_tokens_details.cached_tokens` or DeepSeek's `prompt_cache_hit_tokens`, and Anthropic's cache figures are added into `prompt_tokens` so the subset relationship holds there too. A provider that reports nothing gives 0, and the estimate falls back to charging everything fresh.

A rate the provider varies by time of day is not modelled — declare the rate you want the estimate against. `cost_cents` is an estimate, which is what it is named.

**An unpriced call reports `None`, never `0.0`.** That distinction is load-bearing: a zero cost and a zero duration are what a *missing* measurement looks like, and telling the two apart is the whole point. `tests/test_metrics_completeness.py` enforces it, along with the rule that every provider and every response path fills every declared metric — see `.claude/rules/metrics-completeness.md`.

#### Guarded agents: two calls, two costs

A `GuardedChatter` reply is at least two LLM calls — the chatter writes, the guard reads it and votes — and a rejected reply is four. Both halves are reported, and adding them up is the caller's job:

```python
events = []
agent.guard_callback = events.append

resp = agent.chat("hola")

chatter = resp.metrics.cost_cents
guard = sum(e.metrics.cost_cents for e in events if e.metrics)
print(f"turn cost: {chatter + guard:.5f} cents")
```

Two things worth knowing:

- **`resp.metrics` covers every chatter call the turn made**, not just the first. When the guard rejects a reply the chatter is called again, and those retries are added in — otherwise a rejected turn, which is the expensive one, would under-report.
- **The guard's engine is priced separately.** A guard that declares no `engine` of its own inherits the agent's, pricing included; one that declares its own needs its own `pricing` block, since it may well run a different model.

On one guarded chat agent, where both run the same model, the guard costs roughly as much as the chatter — so a guarded agent is about twice the price of an unguarded one, not a rounding error on top.

#### Retries

Two different things can make a response take more than one LLM call, and the metrics report them differently because they cost differently.

A **transport retry** is a call that never reached the model: a connection failure, a 500, a 429. `retry.py` waits and tries again. The attempt raises before the provider reports any usage, so it has no tokens and no price — what it cost is time, and that is what is measured:

```python
resp = agent.act({"text": "..."})

resp.metrics.retries          # attempts that failed and were retried
resp.metrics.retry_duration   # seconds they took, backoff included
```

`retry_duration` is **not** part of `total_duration`, which is the successful attempt's own figure — the wall clock a caller waited is the two added together. A cache hit reports `0` for both: no request left the machine, so nothing could fail.

A **parse retry** is different: the call succeeded and the model answered, but the answer did not match the schema, so `max_parse_retries` asks again. That is a whole extra LLM call with its own tokens and its own price, and since v4.26.0 its metrics are added into the response — an agent that needed two retries used to report one call's cost while having paid for three. It is *not* counted in `retries`, which is only for attempts that produced nothing.

A retry always goes back to the **same model on the same provider**. Only a `fallback_model` changes which model answers, and it is never automatic — see below.

#### Which model answered

`metrics.model` names the model that actually produced the response, as the provider named it in its own reply:

```python
resp = agent.act({"text": "..."})
resp.metrics.model      # 'openai_compat::deepseek-v4-flash'
agent.engine.model_id   # 'openai_compat::deepseek-chat'
```

It is a separate field from the configured `engine.model_id` because the two differ in two situations that were both invisible before v5.4.0:

- **A provider resolving an alias.** Ask DeepSeek for `deepseek-chat` and `deepseek-v4-flash` answers. Only the reply knows.
- **A fallback answering.** When the primary fails after its retries and the fallback succeeds, `metrics.model` is the fallback's — under `fallback_model` and under `fallback_engine` alike, so a fallback on another provider names itself too. Every call used to be recorded against the configured name, so a run half-answered by a fallback logged all of it against the model that had failed.

The same split is in the call log: `CallRecord.model_id` is what the agent was configured with, `CallRecord.model` is what answered. A `CallStats` bucket carries `models`, the distinct names its calls were answered by — more than one entry means the bucket was not all served by the same model.

On a response that adds several calls together, `model` holds only when every one of them named the same model, the way `cached` does — a guarded turn's reply and its guard usually run different models, and one field cannot report two. The per-call truth is in the call log, one row per call.

#### Adding calls up

Anything that makes several LLM calls for one response adds them with `LLMMetrics.sum`:

```python
from agents7.models import LLMMetrics

total = LLMMetrics.sum([first, second])
```

Counters and durations add. `cost_cents` adds only when at least one part carries it and stays `None` otherwise, so an unpriced response reads as unpriced rather than as free. `cached` holds only when every part was a hit. Use it rather than summing field by field: it is the one place a new metric has to be handled, which is why a guarded turn, a tool loop and a pipeline all agree.

### Caching

Deterministic LLM calls (`temperature=0`) can be cached in memory with an LRU eviction policy. Cached responses are returned instantly without hitting the LLM. When `temperature != 0`, caching is bypassed automatically.

Streaming responses are also cached: on a miss the result is stored after the stream is fully consumed; on a hit the full response is returned as a single chunk.

The default in-memory cache is lost on process restart. For persistent caching, use the SQLite backend:

```python
from agents7.llm import LLMEngine, CacheConfig

engine = LLMEngine(
    provider="ollama",
    model_name="qwen3",
    cache=CacheConfig(backend="sqlite", path="cache.db"),
)
```

```yaml
engine:
    provider: ollama
    model: qwen3
    cache:
        backend: sqlite
        path: cache.db
```

**YAML configuration:**

```yaml
name: MyAgent
version: 1
engine:
    provider: ollama
    model: qwen3
    cache:
        max_size: 128
system_prompt: Analyze this text.
output_model: mymodule.MyOutput
```

All `cache` fields are optional and fall back to defaults. An empty `cache: {}` enables caching with defaults. Omitting the `cache` key entirely disables caching (backward compatible).

**Python configuration:**

```python
from agents7.llm import LLMEngine, CacheConfig

engine = LLMEngine(
    provider="ollama",
    model_name="qwen3",
    cache=CacheConfig(max_size=256),
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_size` | `128` | Maximum cached entries (0 = unlimited) |
| `backend` | `"memory"` | `"memory"` for in-memory LRU, `"sqlite"` for persistent |
| `path` | `"cache.db"` | SQLite file path (ignored for memory backend) |

**What a cache hit reports about itself.** A hit is a real call
with numbers of its own, so it does not replay the stored ones:

- `metrics.cached` is `True`. On a response that adds several
  calls up — a `ToolAgent` loop, a `GuardedChatter` turn — it is
  `True` only when every one of them was a hit.
- `metrics.total_duration` is how long the lookup took, typically
  well under a millisecond, not how long the original call took.
- `metrics.cost_cents` is `0.0` on a priced engine, because no
  request left the machine. On an engine with no declared pricing
  it stays `None`, so an unpriced run keeps reading as unpriced
  rather than as free.
- `metrics.tokens` and `metrics.prompt_tokens` are the original
  call's — they describe the response being returned. `cached` is
  what tells you they were not billed a second time, and
  `CallStats.cached_calls` counts them across a run.

Until v4.24.0 a hit returned the stored duration and cost
verbatim, so any total that included one reported time it did not
take and money it did not spend.

### Validation Hooks

Validators are optional callables that run after the LLM response is parsed into a Pydantic model. Each validator receives the model, can transform it (return modified), or reject it (raise an exception, which sets the model to `None`). Validators run in list order; the output of one feeds into the next, and the first failure short-circuits the pipeline.

Since `Detector`, `Classifier`, and `Regressor` all inherit from `Labeler`, they get validators for free.

In batch mode, validators run on each individual item (not the batch wrapper).

**Python configuration:**

```python
from pydantic import BaseModel
from agents7.agents import Labeler


def require_non_empty(model: BaseModel) -> BaseModel:
    """Reject models with empty name."""
    if not model.name:
        raise ValueError("Name must not be empty")
    return model


def normalize_name(model: BaseModel) -> BaseModel:
    """Title-case the name field."""
    model = model.model_copy(update={"name": model.name.title()})
    return model


agent = Labeler(
    name="ValidatedExtractor",
    version=1,
    engine=engine,
    system_prompt="Extract person info.",
    output_model=PersonInfo,
    validators=[require_non_empty, normalize_name],
)
```

**YAML configuration:**

```yaml
name: ValidatedExtractor
version: 1
engine:
    provider: ollama
    model: qwen3
system_prompt: Extract person info.
output_model: mymodule.PersonInfo
validators:
    - mymodule.require_non_empty
    - mymodule.normalize_name
```

YAML `validators` is a list of dotted import paths, resolved the same way as `output_model`.

### Agent Pipelines

A `Pipeline` chains multiple agents sequentially. The output of each step is mapped into the input for the next step via an optional custom mapper function. Pipelines support single and batch modes, both sync and async.

Use cases:
- Multi-stage extraction (extract entities, then classify them)
- Enrichment chains (extract data, then augment with computed fields)
- Validation pipelines (extract, then verify with a second agent)

If any step produces `model=None` (parse failure or validator rejection) or a mapper raises, remaining steps are skipped and the pipeline returns `success=False`.

**Python configuration:**

```python
from pydantic import BaseModel
from agents7.agents import Labeler, Pipeline, PipelineStep
from agents7.models import AgentResponse

class StepOneOutput(BaseModel):
    name: str

class StepTwoOutput(BaseModel):
    greeting: str

def name_to_greeting(response: AgentResponse) -> dict:
    return {"text": f"Hello, {response.model.name}!"}

agent1 = Labeler(
    name="Extractor", version=1, engine=engine,
    system_prompt="Extract the person's name.",
    output_model=StepOneOutput,
)
agent2 = Labeler(
    name="Greeter", version=1, engine=engine,
    system_prompt="Generate a greeting.",
    output_model=StepTwoOutput,
)

pipeline = Pipeline(steps=[
    PipelineStep(agent=agent1, mapper=name_to_greeting),
    PipelineStep(agent=agent2),
])

# Single input
result = pipeline.act({"text": "Sarah from London"})
print(result.model.greeting)  # "Hello, Sarah!"
print(result.success)         # True

# Batch input
results = pipeline.act_in_batch([
    {"text": "Sarah from London"},
    {"text": "Dr. Chen from Singapore"},
])
for r in results:
    print(r.model.greeting if r.success else "Failed")
```

**YAML configuration:**

```yaml
steps:
  - agent: configs/extractor.yml
    mapper: mymodule.name_to_greeting
  - agent: configs/greeter.yml
    agent_type: agents7.agents.labeler.Labeler
```

- `agent`: Path to the agent's YAML config file (required)
- `agent_type`: Dotted import path to the agent class (optional, defaults to `Labeler`)
- `mapper`: Dotted import path to a mapper function (optional, defaults to `model.model_dump()`)
- `condition`: Dotted import path to a condition function (optional, defaults to always execute)

**Conditional steps:** Add a `condition` callable to `PipelineStep` to gate execution. The condition receives the current input dict and returns `bool`. If `False`, the step is skipped and the input passes unchanged to the next step:

```python
def has_name(input_data: dict) -> bool:
    return bool(input_data.get("name"))

pipeline = Pipeline(steps=[
    PipelineStep(agent=extractor),
    PipelineStep(agent=enricher, condition=has_name),
    PipelineStep(agent=formatter),
])
```

In batch mode, conditions are evaluated per-item: items that fail the condition skip the step while others proceed normally.

For simple sequential chains without custom mappers, use `from_agents`:

```python
pipeline = Pipeline.from_agents([agent1, agent2, agent3])
result = pipeline.act({"text": "input"})
```

**Pipeline streaming:** Stream the final step's output while running intermediate steps normally:

```python
stream = pipeline.act_stream({"text": "input"})
for chunk in stream:
    print(chunk, end="", flush=True)

print(stream.agent_response.model)

# Async version
stream = await pipeline.act_stream_async({"text": "input"})
async for chunk in stream:
    print(chunk, end="", flush=True)
```

If an intermediate step fails, the stream yields nothing and `agent_response.model` is `None`.

### Batch Concurrency Control

When using `act_in_batch_async`, you can limit the number of concurrent LLM requests via a `BatchSettings` instance. This prevents overwhelming the LLM server with large input lists.

```python
from agents7.settings import BatchSettings

# Labeler: limit to 3 concurrent chunk requests
responses = await agent.act_in_batch_async(
    inputs,
    batch_settings=BatchSettings(max_concurrent=3),
)

# Pipeline: limit to 2 concurrent requests per step
results = await pipeline.act_in_batch_async(
    inputs,
    batch_settings=BatchSettings(max_concurrent=2),
)

# Unlimited concurrency
results = await pipeline.act_in_batch_async(
    inputs,
    batch_settings=BatchSettings(max_concurrent=None),
)
```

| `batch_settings` | `max_concurrent` | Behavior |
|-------------------|------------------|----------|
| `None` (default) | `5` | Sensible throttling out of the box |
| `BatchSettings(max_concurrent=None)` | `None` | Unlimited concurrency |
| `BatchSettings(max_concurrent=N)` | Positive `int` | Caps concurrent LLM requests via `asyncio.Semaphore` |

The default `max_concurrent` value can be overridden via the `BATCH_MAX_CONCURRENT` environment variable.

The semaphore is created per call (not stored on the instance), so different calls can use different limits.

### Agent Introspection

Inspect agent configuration at runtime:

```python
agent = Classifier.from_yaml("configs/sentiment.yml")

# JSON schema dict for the output model
agent.output_schema
# {"properties": {"category": ..., "reason": ...}, ...}

# StrEnum field → member names mapping
agent.enum_values()
# {"category": {"POSITIVE": "Positive.", "NEGATIVE": "Negative."}, ...}

# Human-readable summary
print(agent.describe())
# Agent: SentimentClassifier (v1)
# Model: ollama::qwen3
# Output: Sentiment
# Enum fields:
#   category: POSITIVE, NEGATIVE, NOT_APPLICABLE
#   reason: ENTHUSIASTIC, FRUSTRATED, NOT_APPLICABLE
```

### Agent Cloning with Overrides

Create agent variants with selected fields replaced, without rebuilding from scratch. Useful for A/B testing, model comparison, and prompt experimentation:

```python
agent = Labeler.from_yaml("configs/my_agent.yml")

# Swap model
variant = agent.with_overrides(
    engine=LLMEngine(provider="ollama", model_name="llama3")
)

# Change prompt
variant = agent.with_overrides(
    system_prompt="New extraction instructions."
)

# Enable parse retries
variant = agent.with_overrides(max_parse_retries=2)

# Multiple overrides at once
variant = agent.with_overrides(
    name="experiment-v2",
    think=True,
    max_parse_retries=1,
)
```

The original agent is unchanged (frozen dataclass). Defined on `BaseAgent`, so it works with **every** agent type — the structured-output agents (`Labeler`, `Detector`, `Classifier`, `Regressor`), the chatters, and `ToolAgent`. Accepted keyword arguments are whatever fields the concrete class defines.

### Batch Export

Export agent responses to JSONL or CSV files after `act_in_batch`:

```python
from agents7.eval.export import export_jsonl, export_csv

responses = agent.act_in_batch(inputs)

# Export to JSONL (one JSON object per line)
export_jsonl(responses, "results.jsonl")

# Export to CSV (auto-detects model fields)
export_csv(responses, "results.csv")

# Export specific fields only
export_csv(responses, "results.csv", fields=["category", "reason"])

# Export to Polars DataFrame (requires polars)
from agents7.eval.export import export_dataframe

df = export_dataframe(responses)
df = export_dataframe(responses, fields=["category", "reason"])
```

### Evaluation Workflow

`run_eval_suite()` is the composed route from a golden set to a report — one call that loads and normalises the eval data, builds the agent from its YAML config, scores every row, aggregates metrics, summarises the calls it made, and writes a Markdown report. Reach for this first; the primitives in the next section are for when you need finer control.

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
- **Row shape** — both nested rows (`{"input": {...}, "category": "..."}`) and flat rows (`{"text": "...", "category": "..."}`) are normalised automatically.
- **The confusion matrix** defaults to `category` when that field is scored, otherwise the first scored field. Override with `confusion_field=`.
- **The call log** is written by a `JsonCallLogger` attached to a clone of the agent (the agent you passed is never mutated), then summarised into the report's *Call Log* section — one record per real LLM call, so a batched run reports the number of requests, not the number of rows.
- **Report notes** always cover the eval set and the accuracy. Pass `notes=[...]` to append domain caveats a reader needs to interpret the numbers (e.g. a category the output model deliberately disables).

The same route backs the CLI, so `agents7 eval` and a Python caller produce identical results:

```bash
agents7 eval -c configs/sentiment.yml -d data/golden_set_40.jsonl \
    --report reports/sentiment.md --call-log reports/sentiment_calls.jsonl
```

### Evaluating Tool Agents

A `ToolAgent` answers in free text, so there is no label to compare
against — two correct answers to the same question rarely share a
wording. What *is* checkable is the **trajectory** it took: which
tools it called, in what order, and with what arguments. Set
`agent_type: tool` and `run_eval_suite` routes the run through the
trajectory scorer automatically; everything else — the call log, the
metrics, the report — is unchanged.

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
- **Arguments match on the stated keys only** — pin the ones that
  matter (`sku`, `quantity`) and leave the rest out rather than
  spelling out every default. Omitting `arguments` scores the call on
  its name alone, which is the right choice for an argument the model
  legitimately phrases differently (a search `query`).
- **`defaults` pins an argument as absent, or equal to a value** —
  `{"name": "order_total", "arguments": {"quantity": 4},
  "defaults": {"express": false}}` passes a model that omits
  `express` and fails one that invents `express=true`. Neither of the
  other two options can express that: left out entirely, an invented
  argument is ignored and the row passes; pinned under `arguments`,
  a model that correctly omits it fails.
- **A row may expect no calls at all** (`"tools": []`) — that is how
  you assert an agent *doesn't* reach for a tool it doesn't need. Such
  a row is scored under `exact` whatever mode the run is in: with no
  expectation to miss, `in_order` would tolerate every call as merely
  extra and the row could never fail.
- **A call the tool rejected satisfies nothing.** A row naming
  `find_product` asserts the agent looked the product up, so a call
  whose arguments the tool threw out leaves that expectation
  unmatched; it is reported among the extras, marked `[failed]`, and
  counted in `run.metrics["trajectory"]["failed_calls"]`. This is the
  case pinning arguments cannot cover — a row leaves the search
  `query` unpinned precisely because its wording is the model's to
  choose, which is where a malformed value would otherwise slip
  through on the tool name alone.
- **`allow_failure` is for a row that tests the error path** —
  `{"name": "check_stock", "arguments": {"sku": "NOPE-99"},
  "allow_failure": true}` is a row where the tool is *meant* to
  raise and the agent is meant to relay it, so the call counts as
  made either way. Leave the key off everywhere else; it is the one
  opt-out of the rule above.

Alongside the usual accuracy, the report gains a **Trajectory** section
with per-tool recall, and each failing case names exactly which calls
were missing and which were extra. The confusion matrix is omitted — a
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

### Scoring a Chat Agent's Style

A conversational agent has neither a label nor a tool call to score
— two good replies to "how are you" share no wording. So the only
instrument it had was a person reading a transcript, which is slow,
subjective, and blind to the half of a persona that is a plain
**string property of the output**: message count, line length,
forbidden words, how a reply opens, whether two replies in a row
end on a question.

A **style contract** scores that half with no model and no person.
What is not checkable that way — is the agent good company, is the
tone right — stays with the human reading the transcript, which is
the point: the read gets spent on the questions only a person can
answer.

A third thing comes out of that read and is neither: **the
operator's correction of a specific line, and the rule it becomes.**
`eval.chat_feedback` is the format for it. One entry carries a
rewrite, a rule, or both — a real correction is usually both — and
three things are derived from the one file: `rule_changes()` groups
the rules by the config file they land in and flags the ones that
have not shipped, `eval_rows()` turns corrected turns into
machine-readable targets, and `render_feedback()` writes the
operator's report. `agents7 chat-feedback <file>` does all three.
See **Feedback on a chat agent** below.

Two things differ from the other scorers:

- **The counterpart is scripted, deliberately.** Judging whether an
  agent is good company needs a *live* counterpart, since a reply
  written before the agent spoke proves nothing. Detecting that a
  rule stopped holding needs the opposite — the same stimulus every
  time, so the only thing that changed is the config.
- **A rule scores as a rate, not a pass.** Chat agents run at
  temperature, so one run of one script is an anecdote. Each rule
  declares the rate it tolerates via `budget`: `0.0` is a hard rule,
  `0.5` tolerates a tendency, and `None` measures without ever
  failing — for a number worth watching that has no agreed
  threshold yet.

The contract lives beside the agent as data, like a golden set:

```yaml
repeats: 5
rules:
    - id: one_message_when_dry
      description: When the user writes three words or fewer, one message.
      check: max_messages
      args: {limit: 1, when_prompt_words_at_most: 3}
      budget: 0.0
    - id: ends_on_a_question
      description: Asking is one move, not the default.
      check: ends_with_question
      budget: null           # measured, never failed
scripts:
    - id: cold
      opens: true
      lines: ["bien", "normal", "ya", "no"]
```

```python
from agents7.eval import load_style_contract, run_style_contract

contract = load_style_contract("data/style_contract.yml")
report = run_style_contract(lambda: build_agent(), contract)
print(report.summary())
assert report.passed
```

**A script can hand the agent context.** A rule that depends on a
[context param](#per-conversation-context) — the other person's
nickname, say — cannot be exercised by a script that leaves every
param at its default. A script's optional `context` mapping is
passed to `set_context` before its first turn:

```yaml
scripts:
    - id: nick
      opens: true
      context: {nick: Marta_87}
      lines: ["hola"]
```

The context is part of the stimulus, so it is part of the run's
fingerprint: changing it starts a fresh baseline. A script that
declares none hashes exactly as it did before.

**Use `run_style_contract_async` for anything real.** The
conversations are independent — each gets its own agent from the
factory and shares no state — so running them one after another
costs wall clock and buys nothing. Turns *within* a conversation
stay ordered either way, so the report is identical run for run,
labels included:

```python
from agents7.eval import run_style_contract_async

report = await run_style_contract_async(
    build_agent, contract, concurrency=5
)
```

This is what decides whether a contract gets run at all. On a
chat agent, 20 conversations took about 17 minutes serially, which
is too slow to put on every version. The ceiling on `concurrency`
is the provider's rate limit, not the machine — every one of these
is waiting on a network call.

The async runner closes every agent it built (`aclose()`, see
[Closing Async Clients](#closing-async-clients)) before it returns,
so an `asyncio.run` of it ends without an *"Event loop is closed"*
traceback. The agents stay usable; their next async call connects
afresh.

**A conversation that raises loses only itself.** Both runners
catch it, leave that conversation unscored, and score the rest —
so a provider dropping one connection costs one conversation
rather than the whole invocation. On a chat agent it used to cost
about seventeen minutes and seventeen cents, with nothing
recorded. The runs that did not finish are in `report.lost`, are
named in the summary, and `report.complete` says whether there
were any. Their replies are **not** scored even where they got
some: a five-turn prefix of a sixteen-turn script is a different
stimulus, and folding it into a band built from whole
conversations compares two different things. The invocation still
raises when nothing at all finished, because a report over no runs
would pass every budget and say nothing.

Recovery sits at the conversation and not at the turn: the engine
already retries the provider call and counts what that cost in
`LLMMetrics.retries`, so a turn that reaches the runner has
exhausted that mechanism. What was actually being lost was the
other nineteen conversations.

**A run stops when the provider stops answering.** Once two
conversations in a row are lost to a timeout or a dropped
connection, no further conversation starts, and the ones in flight
end before their next turn — a turn already sent is never cut off.
The rest appear in `report.lost` as not played, and
`report.stopped_early` says the run stopped. A finished
conversation, or one lost to any other error, resets the count.
Pass `stop_after_unavailable=` to change the number, or `None` to
play every conversation regardless.

To avoid starting a run while the provider is down at all, ask it
first. `check_available()` sends one small request with a hard time
limit, straight to the provider — no retry, no cache, no fallback:

```python
check = agent.engine.check_available(timeout=15)
if not check.answered:
    print(f"Not answering: {check.error}")
```

```
Style contract: 42 replies over 3 runs, 0 LLM calls
  BROKEN one_message_when_dry            46% (6/13, <=0%)  runs 42%-100% !
  BROKEN opens_in_lower_case             45% (18/40, <=25%)  runs 19%-73% !
    ok   ends_on_a_question              55% (23/42, measure)  runs 50%-64%
  !  = runs disagreed by more than half. The pooled rate is not a baseline.
```

**Every row carries the range across runs, not only the pooled
rate**, because a pooled rate cannot distinguish one bad run from
several middling ones — and those call for opposite responses. A
rule whose runs disagree by more than half is marked `!` and
listed by `report.volatile`: its pooled figure is an average over
runs that behaved differently, so it predicts nothing about the
next one and must not be quoted as a baseline. This is not
hypothetical. The first contract written with this module scored a
rule at 81% on one repeat and 23% on three, and the pooled report
could not say which. `RuleResult.per_run` holds each run's score,
labelled `script#repeat`; a run the rule never applied to is left
out of the range rather than counted as zero, which would make a
rule look steadier than the runs that actually tested it.

**A rate divides by the turns the rule applied to, not by every
reply.** `when_prompt_words_at_most` above means the rule only
looks at turns where the counterpart was terse; without it the rate
would be diluted by every turn the rule was never about.
`forbidden_pattern` takes `when_prompt_matches` for the same
reason: a regular expression the counterpart's line must match
for the rule to apply, for a rule about how the agent answers one
kind of question. A rule
that no reply triggered is reported as `??` rather than passing
silently — a script that never reaches a rule is a gap in the
script, not evidence about the agent.

Built-in checks, all declarable by name in YAML: `forbidden_text`,
`forbidden_pattern`, `max_messages`, `max_line_chars`,
`ends_with_question`, `repeats_question_ending`,
`first_char_lowercase`, `capitalises_after_stop`, `max_emoji`,
`greets_after_opening`, `repeats_opener`,
`reproduces_verbatim`, `copies_example`. A rule may also be built in code with any
callable — it takes a `ReplyContext` and returns True for a
violation, False for a clean turn, and None when the rule does not
apply.

**`repeats_opener` compares conversations with each other.** It
takes no arguments and scores only each conversation's first reply:
a violation when another conversation in the same invocation opened
with the same line (ignoring casing and end punctuation). The rate
is the share of conversations whose opening was not their own —
a chat agent opened five of five with the same line while told
to vary it. Because it needs the whole invocation, the runner binds it
before scoring; calling it on one `ReplyContext` raises. A check of
your own can do the same by giving its callable a `bind_runs(runs)`
attribute that returns the per-reply check. The rate grows with the
number of conversations, so compare it only between invocations of
the same size.

**`copies_example` measures copying, not just identical lines.**
`reproduces_verbatim` only fires on a message equal to an example,
so a copy with one word swapped or a sentence appended passes. In
a chat agent's 13th September 2026 read that was most of them.
`copies_example(lines, min_similarity=0.7, min_chars=16)` counts a
message that contains an example line, or that scores at least
`min_similarity` against one (`difflib`). At 0.7 that read split
cleanly: every copy scored 0.71 or above, and the closest line the
agent wrote itself scored 0.58. Lines and messages under
`min_chars` are skipped, because a short stock reply ("recibido")
is not a copied example.

**`first_char_lowercase` and `capitalises_after_stop` are two
rules, and a persona usually insists on only the second.** Opening
a chat message in lower case is a register choice. Capitalising a
new sentence *inside* the message is not. One chat persona asks
for both in one paragraph and calls only the second non-optional —
and its contract read the insistence as attaching to the first, so
for weeks it enforced a preference at `budget: 0.25` while the
actual rule went unmeasured. When it was finally measured, the
first recorded read had broken it in 33% of the replies it applied
to, including the exact sentence the persona uses as its example of
what not to write.

To score conversations that already happened — a past read, a real
conversation — use `score_transcripts`, which makes no LLM calls:

```python
from agents7.eval import score_transcripts

report = score_transcripts(contract, [
    [(None, "¡Buenas!"), ("bien", "¿bien o bien de no preguntes?")],
])
```

Pass one list per conversation rather than one flat list. Each
conversation is numbered from zero and its first reply has nothing
before it, so a second conversation's legitimate opening greeting
is not read as a greeting mid-chat, and its first reply is not read
as following the previous conversation's last.

The report carries every LLM call the run made — guard checks and
regenerations included, via the agent's `call_logger` — so a
contract that has to be run repeatedly can say what it cost.
`report.cost_cents` is `None` rather than `0.0` when no call was
priced, so an unpriced engine is never reported as a free one.

#### Comparing runs over time

**One invocation is not a baseline.** Measured on a real agent:
three invocations of the *same* config, the *same* scripted
counterpart and the *same* code scored one rule at 23%, then 70%,
then 41%. Ten runs inside a single invocation agreed well enough to
look like a measurement; the invocations did not agree with each
other. So the unit worth persisting is the whole invocation:

```python
from agents7.eval import (
    append_style_run, compare_to_history,
    load_style_history, record_style_run,
)

record = record_style_run(report, contract, build_agent(), "v0.1.19")
history = load_style_history("data/style_history.jsonl")
print(compare_to_history(record, history).summary())
append_style_run(record, "data/style_history.jsonl")
```

```
Against 2 prior invocation(s) of the same configuration (4f8e4071c8dd):
  better opens_in_lower_case      8%  (was 12%-20%, margin 4%)
         cold 12%  warm 3%
    ==   one_message_when_dry    41%  (was 23%-70%, margin 29%)
    ==   ends_on_a_question      48%  (was 44%-55%, margin 7%)
         cold 44%  warm 53%
  Indented lines split a rate by script. Only the combined rate is judged.
  margin = how far this rate could sit from its own truth by
  chance. A rate outside its band by less than this is not a change.
```

The indented line appears when two or more scripts exercised a
rule and at least one reply broke it (since v9.4.0). Scripts are replayed in a fixed proportion,
so a move in one of them can hide inside the combined rate; the
split is there to be read, and the verdict stays on the combined
rate. `RuleRecord.by_script` and `RuleComparison.script_split` give
the same figures to code.

Six things make that trustworthy rather than merely tidy:

- **Configuration identity is computed, not claimed.**
  `fingerprint_run` hashes the agent's resolved system prompt with
  the contract's rule ids and script lines. A baseline is built
  only from records sharing it, so editing a persona starts a fresh
  baseline instead of being judged against a band that described a
  different agent. A version label would not do — it is asserted by
  whoever types it, and a config can change without one moving.
  **A rule's budget is not in the hash** (since v8.0.0): it is read
  once, against a rate that has already been measured, so it cannot
  move a number. Keying on it meant that writing down the threshold
  a baseline had just justified discarded that baseline — the one
  edit guaranteed to change nothing also reset every band. Adding
  or removing a rule still changes the hash, because the ids do.
- **Each rule keeps its own history** (since v9.1.0). Comparison
  keys on `configuration_key`, the fingerprint without the rule
  ids, so adding a rule to a contract leaves every other rule's
  band in place. Each rule's band then holds only runs scored
  under the same `definition` — the check's name and arguments,
  which `build_rule` hashes onto `StyleRule` and
  `record_style_run` stores on `RuleRecord`. Changing what a rule
  checks under an unchanged id starts that rule afresh instead of
  judging new numbers against the old check's band. Records
  written before v9.1.0 carry neither field and compare exactly as
  they did.
- **The guard counts, and whether it ran is read from the run.** A
  guard rewrites replies before they are scored, so a run made
  without one measures something else: on a chat agent, dropping
  the guard moved one rule from 50% to 56%. When a guard took part, its resolved
  prompt and configured model go into the hash, and
  `StyleRunRecord`'s `guard_model_id` names that configured model —
  the guard is often a different model at a different price, and on
  that agent it is 59% of the bill. Since v9.3.0
  `guard_answered_by` records the model the provider says answered
  the guard, and a band only holds runs whose guard was answered by
  the same one, as below for the chatter. Participation cannot be read off the agent,
  because an unguarded run is made by replacing the guard's check
  while the agent keeps a fully configured `guard`; so
  `fingerprint_run` takes the report and reads the calls the run
  actually made. Pass it: `fingerprint_run(contract, agent,
  report)`. An agent with no guard hashes exactly as it did
  before, so unguarded history keeps its baseline.
- **A band only holds runs answered by the same model** (since
  v9.0.0). The chatter's model stays out of the hash, because it
  is the thing under test. Instead, `StyleRunRecord.answered_by`
  records the models the provider says answered (`CallRecord.model`),
  and `compare_to_history` leaves out runs of the same configuration
  answered by any other model. It lists them in `other_models` and
  names them in the summary, so a switch is still reported. This
  matters because a provider can move an alias under an unchanged
  config: DeepSeek moved `deepseek-chat` to `deepseek-flash` on
  10th September 2026, a chat agent's casing rule went from 75% to
  10%, and a band spanning both would pass almost anything. Runs
  that recorded no model are left out as soon as the current one
  records a model.
- **No verdict below two prior invocations.** One earlier run is a
  point, and a point has no spread. The comparison returns
  `Verdict.NO_BASELINE` and says so.
- **A change means outside the band *and* outside it by more than
  the rate's own sampling margin** (`sampling_margin`, twice the
  spread, floored at half an event). Without the margin,
  four stray events out of 175 flagged as a regression against a
  band built from one and one.
- **A rarely broken rule is shown, not judged** (since v9.6.0). When
  the prior invocations broke a rule fewer than
  `RARE_VIOLATIONS_PER_INVOCATION` (one) times each on average, its
  verdict is `Verdict.RARE`: the summary prints its rate marked
  `rare`, and it is never in `worse` or `better`. A band built mostly
  from zeros cannot tell a regression from bad luck: a chat agent's
  gate failed `no_trailing_laugh` twice at 3 of 111 replies on an
  unchanged config, with the laughs in different conversations each
  time. The price is that a quiet rule which jumps is not caught
  either — the same agent once scored `one_message_when_dry` at 10 of 48 after
  runs of 1, 1, 0 and 0 — so read the `rare` lines.
- **The margin is measured from the prior invocations themselves**,
  once there are four of them (`MIN_SPREAD_PRIORS`). What is being
  judged is whether the *invocation's* score moved, so the honest
  bar is how much that score has moved before: the prior rates are
  draws of exactly that quantity, with no independence assumption
  and no pooling of one script with another. Below four priors the
  margin falls back to the proxies below, and `margin_source` says
  which of the three it came from — the summary marks the two
  fallbacks with `*` and `**`, because they disagree with the real
  figure by two to three times.

  The priors' spread is read at the **upper end of its 90%
  confidence interval** (`SPREAD_CONFIDENCE`), not at its point
  value. Four values carry three degrees of freedom, so four that
  happen to agree used to give a margin of a point or so: a chat
  agent's gate failed an unchanged config at 18.4% against 14.9%-16.5%.
  Replayed over that gate's history, the bound took false gate
  failures on unchanged configs from six to none and the median
  margin from 9.2 to 16.7 points. The widening fades as priors
  accumulate.
- **Fallback one, the spread across this invocation's own runs.** A
  contract plays a script, so its replies are not drawn
  independently — every reply after the first answers an agent that
  has read its own previous ones. The independent unit is the run.
  On a chat agent, `ends_on_a_question` varies barely at all between
  conversations while `one_message_when_dry` scored 42%, 82% and
  81% across three, roughly three times the variance a pooled
  denominator assumes: the agent settles into a rhythm early and
  keeps it. Treating its 16 dry turns as 16 independent draws is
  what made three invocations of one unchanged config read as two
  regressions. Each record keeps every run's label, so this spread
  is measured *within* each script and pooled across them rather
  than taken around the grand mean — every invocation replays every
  script in the same proportion, so the difference between two of
  them is fixed structure rather than noise.
- **Fallback two, the binomial figure**, for a record written
  before per-run rates were persisted. It assumes the replies were
  drawn independently, which they are not, so it understates the
  uncertainty.

  **Why the fallbacks are fallbacks.** A chat agent has six gate
  invocations of one configuration, so the real
  invocation-to-invocation movement can be read off and compared
  against what the within-run proxy printed:
  `two_questions_in_a_row` 20% against a real 6%,
  `ends_on_a_question` 16% against 9%, `capitalises_after_stop` 18%
  against 11%, `one_message_when_dry` 37% against 19% — four rules
  held to a bar two to three times too wide, where a real
  regression passes silently. And `opens_in_lower_case` printed 12%
  where the real movement is 20%, so it is not wrong in one
  direction only and cannot be corrected by moving a constant.
  Reading the margin off the priors puts all five within a point or
  two of the measured movement.

  **The proxy is dropped, not kept as a floor.** Taking whichever
  of the two figures is larger would leave every too-wide rule
  exactly as wide as it was, which is the failure this closes. The
  half-an-event floor stays, so four priors that happen to agree
  still cannot gate on a single event.
- **The history is append-only**, because it is the only thing that
  can say how much a figure moves when nothing changed.

Each record also keeps the budget it was scored under — a threshold
tightened later would otherwise make every earlier invocation look
like a failure it never was — the model id, because a provider
serving a different model explains a moved number as readily as a
config edit does, and `lost_runs`, because an invocation that
measured fewer runs than it planned carries more uncertainty than
its `repeats` implies.

#### Sizing a contract

A contract that cannot see the move you care about is a contract
that reports nothing, expensively. `runs_for_shift` answers how
many runs that takes; `detectable_shift` answers the same question
from the other end. Both return a `ShiftEstimate` carrying the
move, the run count that goes with it, and where the spread behind
it came from:

```python
from agents7.eval import detectable_shift, runs_for_shift

rates = result.run_rates                    # one rate per run
runs_for_shift(0.10, rates).runs            # runs to see ten points
detectable_shift(rates, 10).rate            # what ten runs would see
```

**Which spread it uses is the whole accuracy of the answer**, and
both take the same three sources as `sampling_margin`, in the same
order — earlier invocations of this configuration first, the
labelled runs inside one invocation next, bare rates last:

```python
prior = [r.rule("ends_on_a_question") for r in history]

detectable_shift(
    planned_runs=20,
    prior_rates=[p.rate for p in prior],     # draws of the real thing
    prior_runs_each=len(prior[0].run_rates), # the size they were at
    prior_invocations=len(prior),
)   # -> ShiftEstimate(rate=0.142, runs=20, source=MarginSource.PRIORS)
```

`prior_runs_each` is required rather than optional: a spread
measured at ten runs an invocation means something different at
twenty, so without it the figure cannot be rescaled and is
refused. Passing `runs=` a sequence of `RunScore` instead of
`run_rates` keeps two scripts' different true rates apart, which
the bare rates cannot do — see `sampling_margin`.

Measured on a chat agent's five committed invocations, the weakest
source disagreed with the strongest by between 0.38 and 3.76
times on the same rules. It claimed a 2.9-point move was visible
on `no_paragraph_lines` where the real bar is 7.6, and demanded
16.5 points on `capitalises_after_stop` where 4.6 would do. Wrong
in both directions, so no constant corrects it — which is why the
source is returned rather than assumed.

The bar a rate has to clear is the top of the band plus the
margin, and both shrink with the square root of the *runs* — so
four times the runs sees half the move, and a longer script sees
almost nothing extra. Asked how to make a contract more sensitive,
the intuitive answer is to lengthen the scripts; it is the wrong
one.

Two things the result does not say. It is a coin flip, not a
guarantee: a move exactly that size clears the bar about half the
time, and catching it reliably takes roughly twice the runs. And
**more history raises the bar** — the maximum of five prior
invocations sits further above the truth than the maximum of two,
which is a property of a min/max band rather than a defect, so
`prior_invocations` is part of the calculation. It also answers a
question the function invites: adding invocations does not buy
sensitivity, so the runs inside one are the only lever.

Sizing a chat agent's contract this way split it in two. Eleven
of its fifteen rules reach ten points inside the 20 runs an
invocation already has, and `capitalises_after_stop` — the
tightest rule that is actually broken — needs only 5. The rule
measuring how the agent answers a monosyllable needs 161, because
its spread is the agent picking a rhythm per conversation and
holding it.
That is real behaviour rather than measurement error, so it is the
price of the question — and worth knowing before reading a
26-point move as a regression.

#### Comparing two configurations against each other

`compare_to_history` answers *"did this invocation leave its
band"* — one configuration judged against its own past. It has no
answer for *"do these two configurations wobble by different
amounts"*, because a band is a property of one configuration.
`compare_arms` is that second instrument: two sets of invocations,
read against each other.

```python
from agents7.eval import compare_arms, load_style_history

guarded = load_style_history(path, fingerprint="30d996ff9270")
plain = load_style_history(other, fingerprint="526788c30730")
print(compare_arms(guarded, plain, "guarded", "unguarded").summary())
```

It reports three things, and keeping them apart is the point:

- **Level** — the mean rate each way. An apparatus that repairs
  faults shows up here, not in the spread.
- **Invocation spread** — how far whole-invocation scores sit from
  one another. This is what `compare_to_history` builds a band
  from, so it is what decides whether an apparatus change affects
  a band.
- **Conversation spread** — the wobble between individual
  conversations, with binomial noise removed. Two arms at
  different mean rates carry different amounts of that noise, so
  comparing raw run spreads would report a difference in *level*
  as a difference in wobble.

**Every p-value comes with the size of difference the test could
have seen** — `RuleArms.spread_threshold` per rule and
`ArmComparison.panel_threshold` for the panel. A null result is
unreadable without it: it is the difference between having ruled
something out and having been unable to see it. That is the same
discipline `ShiftEstimate.source` applies to a figure's
provenance, applied here to its power.

**Significance is by exact permutation of whole invocations**,
never of conversations — two conversations inside one invocation
are not exchangeable with two inside another, for the same reason
`sampling_margin` takes the run rather than the reply as the
independent unit.

**Only rules that actually fire enter the panel.** A rule with no
violations either way has a spread of zero on both sides, which is
a ratio of exactly one carrying no information; a dozen of those
drag the median to a dead heat whatever the live rules did. The
first version of this comparison reported 1.00x at p=1.000 over
fifteen of a chat agent's rules, seven of which never fired.

`trim=True` (the default) cuts the longer arm to the shorter one's
size, keeping the newest, so two deliberately interleaved arms sit
in the same window of time and a change in what the provider is
serving cannot masquerade as the difference under test. What it
removes is reported as `drift_left` / `drift_right` rather than
discarded silently. Pass `trim=False` for a before/after, where
the arms are whatever sizes they are.

Run on a guarded chat agent (5th September 2026) it settled the question it
was built for: across the eight rules that fire, the guarded arm
moved more on five and less on three, a median spread ratio of
1.07x at p = 0.794, where a consistent difference above 1.65x
would have registered. The guard is not itself a source of
variance.

#### Where inside a conversation a rule breaks

A contract scores a rule per invocation and per run, which cannot
say *when* in a conversation the rule breaks. `decompose_rhythm`
takes one 0/1 row per conversation — one column per turn the rule
applied to — and splits the violations into three effects:

- **Position** — the rate at each turn, and first half against
  second. A rule that holds early and decays shows up here.
- **Conversation** — how far conversation means sit apart once the
  position drift is removed, against what independent coin flips
  would give. A ratio well above 1 means whole conversations run
  high or low.
- **Momentum** — whether a broken turn still predicts the next one
  once both of the above are removed, by shuffling turns *within*
  each conversation.

```python
from agents7.eval import (
    StyleContract,
    decompose_rhythm,
    run_style_contract,
    turn_matrix,
)

cold = StyleContract(rules=contract.rules, scripts=[script], repeats=10)
report = run_style_contract(factory, cold)
rule = next(r for r in contract.rules if r.id == "one_message_when_dry")
rows = turn_matrix(report.replies, rule)
print(decompose_rhythm(rows).summary())
```

**Play one script.** Two scripts are two stimuli with different
true rates, and mixing them puts that gap into the conversation
spread. **Save `rows`**: a re-run is a different set of
conversations, while re-analysing a saved matrix costs nothing and
gives the same figures for the same `seed`.

The summary ends with the uncontrolled figure — the rate after a
violation against the rate after a clean turn — marked as not to be
quoted alone, because most of that gap is position and conversation
rather than memory.

Run on a chat agent's `one_message_when_dry` (5th September 2026, two
runs of ten cold conversations) it found the rule clean on the
first dry turn in 20 of 20 conversations, the rate roughly doubling
by the second half, and conversation means 2.6x wider than
independence allows — a decay rather than the examples, which
pointed the next fix somewhere else.

### Feedback on a Chat Agent

A style contract says a rule stopped holding. A read says whether
the agent is good company. Neither produces the thing that actually
improves it: the operator's correction of a line, and the rule that
correction becomes. That has been written by hand, in prose, in two
shapes — a rewrite (*"this is what he should have said"*) and a
rule (*"he is too verbose and does not mirror the user"*) — and neither
was machine-readable.

`eval.chat_feedback` is one format for both. An entry carries a
rewrite, a rule, or both, because a real correction is usually
both:

```yaml
read_id: 2026-08-28-first-read
agent: david
agent_version: 0.1.4
golden: [warm]          # accepted as a target to imitate in full
entries:
    - conversation: warm
      turn: 3
      verdict: wrong                   # right | weak | wrong
      he_said: best question anyone has asked me all year
      should_say: good question, give me a second
      why: overstates the investment three messages in
      rule: No overstating how interested he is early on.
      lands_in: characters/david.yml
      shipped: v0.1.5                  # empty means outstanding
```

Points worth knowing:

- **`turn` is optional.** Leaving it out makes the entry
  structural — about the whole conversation. *"He writes more the
  less the user gives him"* is the motivating case: the most useful kind
  of feedback, anchored to no single line.
- **`right` is not the absence of criticism.** It marks a line good
  enough to imitate, and such a line becomes a target in its own
  right.
- **`he_said` sits beside the rewrite** so a correction reads as a
  pair without opening the recording, and so a rewrite cannot drift
  from the line it was correcting.
- **`golden` names the conversations accepted whole.** The rest
  contribute corrections only — a correction taken from a
  conversation nobody accepted is still binding.

Three outputs, one source:

```python
feedback = load_feedback("docs/reads/<date>/feedback.yml")

feedback.rule_changes()   # grouped by config file, unshipped flagged
feedback.eval_rows()      # corrected turns as machine-readable targets
render_feedback(feedback) # the operator's report, as Markdown
```

Or from the CLI, which makes no LLM calls:

```bash
agents7 chat-feedback docs/reads/<date>/feedback.yml \
    --report docs/reads/<date>/feedback.md \
    --rows data/golden/<name>.jsonl
```

`merge_entries` replaces every entry on a turn the operator
touched and leaves every other turn alone, so an editing UI can
read a file, add to it and write it back without losing anything
hand-written.

### Evaluation Metrics

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

### Input Preprocessor

An optional callable that transforms input data before it reaches the LLM. Applied consistently across `act()`, `act_async()`, `act_stream()`, `act_stream_async()`, and batch methods.

```python
def clean_input(data):
    """Strip HTML and truncate."""
    data = dict(data)
    data["text"] = strip_html(data["text"])[:2000]
    return data

agent = Labeler(
    ...,
    preprocessor=clean_input,
)

# Preprocessing happens automatically
response = agent.act({"text": "<p>Very long HTML...</p>"})
```

**YAML configuration:**

```yaml
preprocessor: mymodule.clean_input
```

### Model Warm-Up

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

### Closing Async Clients

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

### Structured Call Logging

Optional per-call logging that captures trace ID, agent name, model, tokens, duration, cost, success/failure, and — on a call that failed — the prompt that produced it. Useful for debugging, observability, and cost tracking.

```python
from agents7.call_log import JsonCallLogger

# Log to file (JSON lines)
logger = JsonCallLogger("calls.jsonl")
agent = Labeler(..., call_logger=logger)

# Log to stdout
logger = JsonCallLogger()
agent = Labeler(..., call_logger=logger)

# Custom logger — any callable accepting CallRecord
agent = Labeler(..., call_logger=lambda r: print(r.to_dict()))
```

When logging to a file, `JsonCallLogger` holds a single append
handle (opened lazily on first write, flushed per record) instead
of reopening the file on every call. Call `close()` or use it as a
context manager to release the handle:

```python
with JsonCallLogger("calls.jsonl") as logger:
    agent = Labeler(..., call_logger=logger)
    ...
```

**One `CallRecord` is emitted per LLM call**, so the log counts the
calls actually made. An `act()`, `act_async()`, `act_stream()` or
`act_stream_async()` that parsed first time made one call and emits
one record. An `act()` or `act_async()` that needed **parse retries**
(`max_parse_retries`) emits one record per attempt: a retry is a
whole extra call to the provider, with its own tokens, its own price
and its own messages. They share the answer's `trace_id`, only the
last one can be a success — the retry loop stops at the first call
that parses — and each failed attempt carries the messages *it* sent,
which include the parse error fed back to the model. Batch runs
(`act_in_batch()` / `act_in_batch_async()`) emit one record per **LLM
request** — that is, per chunk — not one per input item.

Each record reports its own call, never a total over several, so
summing a log never counts the same tokens twice. The
`AgentResponse.metrics` returned to the caller *is* the total across
every call the answer took.

A **streamed** call emits its record once the caller has finished
iterating the stream, not when `act_stream()` returns: whether the
call succeeded depends on parsing the streamed text, which is not
known before then. A caller that abandons a stream without reading
it records nothing.

**A chat turn emits one record per LLM call it makes.** `Chatter`
takes a `call_logger` like any other agent. A plain turn is one
call. A `GuardedChatter` turn is at least two — the reply and the
guard's check — and each regeneration a rejection forces adds
another pair, so a turn rejected once is four records. The guard's
calls get their own records, carrying the guard's name, version
and model: a guard usually runs a different model from the
chatter, and folding its tokens into the reply's record would
report one call made on two models.

Every call in one turn shares that turn's `trace_id`, so the
records of a single reply can be grouped and added up. A
structured-output agent that parsed first time made one call, so
that id is unique to its record; one that needed parse retries
shares it across the attempts.

**Above the turn sits `run_id`**, shared by every call of one
conversation or one eval run — what a log needs to answer *what
did this conversation cost*, which grouping by turn cannot. A
`Chatter` fills it from its `conversation_id` automatically.
Every other agent takes it from the caller:

```python
agent = Labeler(..., call_logger=logger, run_id="sweep-2026-08-30")
```

`run_evaluation_suite()` sets one for you, so each eval run's
calls group even when several runs share a log file. Left unset it
is `null`, which means nobody named a run — not that every call
belongs to one.

```python
from agents7.call_log import JsonCallLogger

with JsonCallLogger("chat.jsonl") as logger:
    agent = GuardedChatter(..., call_logger=logger)
    agent.chat("hola")
    # -> 2 lines: the reply, then the guard, same trace_id
```

```json
{
  "trace_id": "a1b2c3d4e5f6",
  "agent_name": "Extractor",
  "agent_version": 1,
  "model_id": "ollama::qwen3",
  "success": true,
  "tokens": 42,
  "prompt_tokens": 128,
  "cached_prompt_tokens": 0,
  "total_duration": 1.23,
  "cost_cents": 0.0124,
  "cached": false,
  "retries": 0,
  "retry_duration": 0.0,
  "error": null,
  "prompt": null,
  "run_id": "sweep-2026-08-30",
  "recorded_at": "2026-08-30T09:14:02.4813+00:00"
}
```

`recorded_at` is stamped by the producer when the record is
built, immediately after the call returned — a log without it can
be read in order but not against a date, so it cannot say what a
week cost or whether a model got slower. A row read back from a
log written before this field existed reads as `null`: unstamped,
rather than having happened at the moment it was read.

**The prompt is recorded on a call that failed, and only then.**
`prompt` holds the messages that were sent, as a JSON string, so a
row that failed to parse can be reproduced by hand — a re-run at
temperature would produce a different call, so a prompt not
captured during the run is gone. A call that succeeded records
`null`: its prompt is derivable from the agent's config, and
carrying it on every row would multiply the log by the size of the
system prompt.

```json
{
  "trace_id": "9f8e7d6c5b4a",
  "agent_name": "Extractor",
  "success": false,
  "error": "ParseError: expecting value at line 1 column 1",
  "prompt": "[{\"role\": \"system\", \"content\": \"Extract ...\"}, {\"role\": \"user\", \"content\": \"Process the following:\\n{'text': 'hello'}\"}]"
}
```

Batch runs record the chunk's messages, not one item's.
`ToolAgent` records no prompt: it reports no per-call failure the
field could key off yet. Nor does a chatter's own reply, which has
nothing to parse and so cannot fail this way — but its **guard**
can, and an unparseable verdict records the guard input that
produced it.

**Two sinks: a file or a database.**

`JsonCallLogger` appends JSON lines. It suits one eval run and
not months of chat turns — reading it means parsing the whole
file, and two processes appending at once interleave.
`SqliteCallLogger` writes one row per call to a `calls` table
whose columns are `CallRecord`'s fields, indexed on `run_id`,
`trace_id` and `recorded_at`, in WAL mode so the log can be
queried while a run is still writing to it:

```python
from agents7.call_log import SqliteCallLogger

with SqliteCallLogger("data/calls.db") as logger:
    agent = Chatter(..., call_logger=logger)
```

Neither creates its file until the first record is written, so a
run that logged nothing leaves nothing behind.

**YAML configuration:**

```yaml
call_log:
  backend: sqlite        # json (default) | sqlite
  path: data/calls.db    # relative paths sit beside this config
```

A relative `path` is resolved against the directory holding the
config, not the working directory — so the same agent logs to the
same file whether it is loaded by a script or by a web backend.
Written bare (`call_log:`) it defaults to `calls.jsonl` beside the
config.

The older form still works and is for a sink you built yourself:

```yaml
call_logger: mymodule.my_logger
```

The `call_logger` value is a dotted import path to any callable
accepting a `CallRecord`. It is an *alternative* to `call_log`,
not an addition — declaring both raises
`AgentConfigurationError`, since only one sink can be attached
and picking one silently would leave a config that reads as
logging to a database writing somewhere else.

**Reading a log back:**

```python
from agents7.call_log import read_call_log

for record in read_call_log("data/calls.db"):
    print(record.run_id, record.cost_cents)
```

`read_call_log()` and `summarize_call_log()` both accept either
sink's output, told apart by the file's own header rather than by
its name.

**Analyzing a call log:**

`summarize_call_log()` reads a JSONL file written by
`JsonCallLogger` (or any iterable of `CallRecord` / dict / JSON
line) and returns a `CallLogSummary` with overall stats plus
per-agent, per-model and per-run breakdowns and error counts:

```python
from agents7.call_log import summarize_call_log

summary = summarize_call_log("calls.jsonl")
print(summary.format())

# Programmatic access:
summary.overall.success_rate   # fraction of calls that parsed
summary.overall.tokens         # total completion tokens
summary.by_agent["Extractor"]  # CallStats for one agent
summary.by_model["ollama::qwen3"].avg_duration
summary.by_run["cold-read"].cost_cents  # what one run cost
summary.errors                 # {error message: count}
summary.overall.cost_cents     # None when nothing was priced
summary.overall.cached_calls   # how many came from the cache
summary.overall.retries        # attempts that failed and were retried
summary.overall.retried_calls  # how many calls needed one
summary.overall.retry_duration # seconds those attempts took
```

`cost_cents` sums only the records that carried one and stays
`None` when none did, so a run on an engine with no declared
pricing reads as *unpriced* rather than as free. `cached_calls`
counts the calls served from the response cache: their tokens are
in the totals above but were never billed a second time, so a run
with many hits has spent far less than its token totals suggest.
`by_run` holds only the records that named a run, so a log written
before anyone did comes back empty rather than as one run called
`None`.

`retries` and `retry_duration` are what a slow run needs to
explain itself. A run that hit rate limits and retried its way
through used to be indistinguishable in the log from one that
succeeded first time — same records, same tokens, same cost, and
nothing to account for the wall clock. `retried_calls` separates
one call retried five times from five calls retried once.

During an evaluation you do not need to wire this up by hand — pass
`call_log=` to [`run_eval_suite`](#evaluation-workflow) and the route
attaches the logger, summarises the result, and folds it into the
report's *Call Log* section.

### Persona-Driven Chatter (Roleplay Agent)

Give a `Chatter` a structured `Persona` for consistent character
roleplay — the system prompt is auto-generated from biographical
data. `persona` is an optional plugin on the base `Chatter` (the
former `PersonalityChatter` subclass was removed in v4.0.0):

```python
from agents7.agents import Chatter, Persona, PhysicalTraits

persona = Persona(
    full_name="David García",
    birthdate="1990-12-04",
    nationality="Spanish",
    occupation="journalist",
    physical=PhysicalTraits(
        height="178 cm",
        build="medium",
        hair="dark brown, short",
        eyes="brown",
    ),
    personality=["curious", "empathetic", "direct"],
    backstory="Grew up in Madrid, studied journalism at UCM.",
    interests=["politics", "football", "cooking"],
    speaking_style="casual and direct",
)

agent = Chatter(
    name="david",
    version=1,
    engine=engine,
    system_prompt="",
    persona=persona,
    extra_instructions="Always respond in Spanish.",
)

response = agent.chat("¿Cómo te llamas?")
```

Every `Chatter` reply is returned and stored without empty lines.
A persona that writes "two or three short lines" gets them as
consecutive lines; a model that separates them with an empty line
would otherwise hand a chat client an empty message between them.
A live stream still yields chunks as generated, and its
`chat_response` carries the cleaned reply.

YAML configuration:

```yaml
name: david
version: 1
engine:
    provider: anthropic
    model: claude-sonnet-4-20250514
persona:
    full_name: David García
    birthdate: "1990-12-04"
    nationality: Spanish
    occupation: journalist
    physical:
        height: 178 cm
        build: medium
        hair: dark brown, short
        eyes: brown
    personality:
        - curious
        - empathetic
        - direct
    backstory: |
        Grew up in Madrid, studied journalism at UCM.
    interests:
        - politics
        - football
        - cooking
    speaking_style: casual and direct
extra_instructions: |
    Always respond in Spanish.
```

The persona includes: name, birthdate, nationality, occupation, physical traits, personality adjectives, backstory, interests, speaking style, and a freeform `extra` dict for any additional details.

### Scene-Driven Chatter (Contextual Roleplay Agent)

Add a `Scene` alongside the persona to place the character in a
specific **scene** — a location, situation, time period, and set
of behavioral rules. The system prompt frames everything as a
role-play scenario. A `scene` requires a `persona` (the former
`SceneChatter` subclass was removed in v4.0.0):

```python
from agents7.agents import Chatter, Persona, Scene

persona = Persona(
    full_name="Old Thomas",
    occupation="tavern keeper",
    personality=["gruff", "practical", "shrewd"],
)
scene = Scene(
    setting="A medieval tavern at night",
    situation="The tavern is about to close.",
    time_period="Medieval England, 14th century",
    rules=[
        "No knowledge of anything after the 14th century.",
        "Suspicious of strangers.",
    ],
    goal="Close the tavern and get everyone out.",
)
agent = Chatter(
    name="thomas",
    version=1,
    engine=engine,
    system_prompt="",
    persona=persona,
    scene=scene,
)

response = agent.chat("Good evening!")
```

The scene includes: setting, situation, time period, rules, other characters present, goal, and a freeform `extra` dict. See `docs/chatter.md` for the full YAML schema.

### Splitting a Roleplay Agent into Character and Chat Goal

A roleplay agent has two independent axes, and putting them in one
file ties them together: rename the character and you edit a file
full of scene rules; reuse the character in a different
conversation and you copy its whole scenario with it.

`character` and `chat_goal` each name a YAML file, and the loader
folds both into the config:

```yaml
# config.yml — the deployment: model, prices, stores, guard
character: characters/david.yml
chat_goal: goals/source_interview.yml
name: david
version: 1
agent_type: guarded_scene_chatter
engine:
    provider: openai_compat
    model: deepseek-chat
```

```yaml
# characters/david.yml — who he is. No scene, ever.
persona:
    full_name: David García López
    language: Spanish
    speaking_style: |
        Casual y directo. Frases cortas.
extra_instructions: |
    ## Cómo escribe
    Usuario: ¿qué tal?
    $character: liado con un artículo, como siempre
```

```yaml
# goals/source_interview.yml — what he is doing. No persona, ever.
scene:
    setting: Un café ruidoso en Malasaña, por la tarde.
    rules:
        - Nunca revelas quién es tu fuente.
    goal: Conseguir detalles verificables para publicar la historia.
can_open: true
extra_instructions: |
    ## Cómo trabaja
    Fuente: ¿vas a publicar mi nombre?
    $character: no, eso no sale de aquí
```

Three rules govern the composition:

- **Neither file may hold the other's section.** A `character`
  declaring `scene`, or a `chat_goal` declaring `persona`, raises
  `AgentConfigurationError`. The split is enforced rather than
  suggested, because a config that has been split can otherwise
  grow quietly back into one file and the swap stops working with
  nothing failing.
- **`extra_instructions` and `opening_instructions` accumulate**,
  in a fixed order — character, then chat goal, then whatever the
  agent config says itself. Both axes have something to say about
  how the agent writes, so neither silently replaces the other.
  Every other key is an ordinary deep merge with the agent config
  winning, so a deployment can pin what the fragments left open.
- **`$character` resolves to the persona's `full_name`.** That is
  what lets a chat-goal file write a worked example without saying
  who is in it. Swap `characters/david.yml` for another one and every
  example in the goal reads correctly, including the ones the
  guard sees.

Fragments are one level deep: neither may declare `base_config`,
`character`, or `chat_goal` of its own. `base_config` inheritance
still works on the agent config itself, and resolves first.

To read the composed result without building the agent — a
config-contract test, a catalog listing — use
`agents7.agents.resolve_config(path)`.

### Opening the Conversation (Agent Speaks First)

`chat()` answers a message. A character whose persona says how it
greets a stranger needs the other call: one that produces a first
turn with nothing to answer. Set `can_open=True` and use
`open_chat()`:

```python
agent = Chatter(
    name="david",
    version=1,
    engine=engine,
    system_prompt="",
    persona=persona,
    can_open=True,
)

opening = agent.open_chat()   # the agent writes first
print(opening.response)       # "¡Hola! ¿Qué tal la tarde?"

agent.chat("Bien, ¿y tú?")    # the conversation continues normally
```

Without `can_open=True` the call raises
`AgentConfigurationError` rather than inventing an opening the
agent was never given. The async and streaming siblings are
`open_chat_async()`, `open_chat_stream()`, and
`open_chat_stream_async()`; on `GuardedChatter` the opening turn
runs through the guard like any other.

[Per-conversation context](#per-conversation-context) reaches the
opening turn like any other, which matters most here — the first
line is written before the other person has said anything, so
`set_context` is all the agent has to go on:

```python
agent.set_context(nick="lector42")
agent.open_chat()   # greets knowing who it is writing to
```

The model is asked for that first message by an instruction sent
as a user message, which is **not** stored — the transcript
begins with the agent's own line. Set `opening_prompt` to replace
the built-in English instruction, which is what a persona writing
in another language usually wants:

```yaml
can_open: true
opening_prompt: "[Escribe el primer mensaje de esta conversación.]"
```

#### Where the opening lines themselves go

`opening_prompt` is the instruction that asks for a first message.
Guidance on *how* the character opens — its written opening lines,
their length, their register — goes in `opening_instructions`,
which is appended to the system prompt on the opening turn and on
no other:

```yaml
can_open: true
opening_instructions: |
    Cuando le toca abrir a él, siempre muy corto: una línea y para.

    $character: ¡Buenas! ¿Qué tal?
    $character: Hola, ¿tienes un minuto?
```

`$character` resolves to the persona's `full_name` here as it
does in the standing prompt, so the same lines can be written
once and read under any character.

Put those lines in `extra_instructions` instead and they are in
front of the model on every single turn — and a bank of opening
lines the model can read on every turn is a bank it can copy on
every turn. The failure that produced this field: a character who
had just said goodnight answered the other person's *"vale"* with
*"¡Buenas! ¿Qué tal?"*, the first line of its own opening
section, fifteen turns into the conversation.

On `GuardedChatter` a rejected opening turn regenerates from the
same messages, so the retry still knows it is opening.

### Character Facts (Memory Across Conversations)

A `MemoryBackend` stores the *transcript* under a
`conversation_id`, so anything a character improvises about
itself dies with that conversation. A **fact store** keys what
the character said to the *character* instead. Recorded facts are
folded into the system prompt on every turn, in every later
conversation.

`remember()` runs a `FactExtractor` over the finished transcript
and writes what it found. It makes an LLM call, so call it when a
conversation is over, not inside a chat turn:

```python
from agents7.agents import (
    Chatter,
    FactExtractor,
    JsonFileFactStore,
    Persona,
)

store = JsonFileFactStore("facts/")

agent = Chatter(
    name="david",
    version=1,
    engine=engine,
    system_prompt="",
    persona=Persona(full_name="David García López", language="Spanish"),
    facts=store,
    character_id="david",
    fact_extractor=FactExtractor(
        name="david-facts", version=1, engine=engine
    ),
)

agent.chat("¿Cuál es tu película favorita?")
# → "El secreto de sus ojos, sin duda."

agent.remember()
# [CharacterFact(text='Su película favorita es El secreto de sus
#                ojos', source_conversation='default')]

# A different conversation, later — same character:
later = Chatter(..., facts=store, character_id="david")
later.chat("¿Te gusta el cine?")
# David answers as somebody whose favourite film is already
# El secreto de sus ojos.
```

The extractor records only what the **character** said about
itself, only when it is durable, and only when it is not already
known. It ignores what the other person said, and passing states
("estoy cansado", the weather).

**Facts are stored in the character's own language, not in
English.** The language comes from `persona.language` — David
declares `Spanish`, so his facts are stored in Spanish even when
the other person writes in English. That keeps the system prompt
in one language throughout, so the facts read as things David
would say rather than as English notes about him. Proper nouns
are kept as the character wrote them: a film he called *El
secreto de sus ojos* is not rendered as *The Secret in Their
Eyes*. Without a persona,
or without a declared language, the extractor follows whatever
language the transcript is in.

Facts are stored as one JSON file per character, so an entry that
comes out wrong can be corrected by hand:

```json
[
  {
    "text": "Su película favorita es El secreto de sus ojos",
    "source_conversation": "default"
  }
]
```

Other methods and fields:

- `agent.character_facts` — the facts currently in the prompt.
- `agent.forget()` — erase every fact for this character.
  `reset()` clears the conversation and leaves the facts alone.
- `store.add(character_id, ["..."])` — write a fact by hand,
  without an LLM call.
- `character_id` defaults to the agent's `name`, so several
  agents can share one character by setting it explicitly.
- `banned_fact_phrases` — phrases the character must never say.
  A fact containing one is not written by `remember()`, and one
  already in the file is left out of the prompt with a warning
  naming it. The file itself is not edited. Matching ignores
  case and spacing.

A stored fact outlives the config it was written under. Without a
ban list, a phrase the persona forbids today can come straight
back from a fact recorded before the rule existed.

YAML configuration — works on `Chatter`, `GuardedChatter`, and
`GuardedSceneChatter`. The extractor reuses the agent's engine
unless you name another:

```yaml
character_facts:
    path: facts/
    character_id: david
    banned_phrases:
        - sin remedio
    extractor:
        engine:
            provider: ollama
            model: gemma3
```

A relative `path` is resolved against **the directory holding the
config file**, not the directory the process runs from. The folder
above lands beside the YAML that declared it, so a web backend and
a command-line script loading the same config read and write the
same facts. Give an absolute path to put the store somewhere else.
`memory.path` is anchored the same way.

### Testing Utilities

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

### Simple Classifier

The `agents7.simple` module lets you create a `Classifier` from a list of category strings without defining Pydantic models manually:

```python
from agents7.simple import simple_classifier

agent = simple_classifier(
    categories=["SPAM", "HAM"],
    system_prompt="Is this email spam or not?",
    engine=engine,
)
result = agent.act({"text": "Buy now!!!"})
print(result.model.category)  # SPAM
```

Auto-generates `StrEnum` and `ClassificationProperty` subclasses. Supports custom reasons, `not_applicable` toggle, `think` mode, and extra kwargs (`examples`, `prompt_params`).

### Tool Calling

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

#### Streaming the tool loop

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

## Examples

The `examples/` directory contains runnable scripts demonstrating each feature:

**Agents** (`examples/agents/`):

| Script | Feature |
|--------|---------|
| `01_labeler.py` | Entity extraction with batch processing |
| `02_detector.py` | Temperature detection |
| `03_classifier.py` | Sentiment classification |
| `04_regressor.py` | Numeric value extraction |

**Chatter** (`examples/chatter/`):

| Script | Feature |
|--------|---------|
| `01_chatter.py` | Multi-turn chat, streaming, reset |
| `02_chatter_memory.py` | Persistent memory with multiple conversations |
| `03_personality_chatter.py` | Roleplay with structured persona |
| `04_guarded_chatter.py` | Context guard against prompt injection |

**Features** (`examples/features/`):

| Script | Feature |
|--------|---------|
| `01_streaming.py` | Token-by-token streaming (sync + async) |
| `02_export.py` | JSONL and CSV export |
| `03_labeler_params.py` | Prompt parameter substitution |
| `04_sqlite_cache.py` | Persistent SQLite cache |
| `05_config_inheritance.py` | YAML `base_config` inheritance |

**Pipelines** (`examples/pipelines/`):

| Script | Feature |
|--------|---------|
| `01_pipeline.py` | Multi-step pipeline with custom mapper |
| `02_conditional_pipeline.py` | Conditional step execution |

Run from the relevant subdirectory:

```bash
cd examples/agents
python 01_labeler.py
```

## Requirements

- **Ollama provider**: Ollama running locally (default: `http://localhost:11434`)
- **OpenAI-compatible provider**: `uv pip install agents7[openai]` (requires `openai` package)
- Core dependencies: `pydantic`, `ollama`, `tqdm`, `pyyaml`

## Model Compatibility

The library includes automatic fallback handling for models with varying
levels of structured output support:

- **Full support** (e.g., `qwen3`): Uses Ollama's native JSON schema format
- **Partial support** (e.g., `gpt-oss:20b`): Falls back to prompt-based JSON
  generation with automatic parsing fixes

Common LLM output issues are handled automatically:
- Markdown code blocks in responses
- Malformed JSON (double braces, etc.)
- Enum names vs values mismatch

## Development

```bash
# Install dev dependencies (using uv)
uv pip install -e ".[dev]"

# Run unit tests (fast, no LLM required)
uv run pytest tests/ -v

# Run integration tests (requires Ollama running)
uv run pytest integration_tests/ -v

# Run linter
ruff check .
```

### Integration Test Models

The integration tests call Ollama with the following models.
Pull them before running the suite:

| Test file | Config | Model | Size |
|---|---|---|---|
| `test_detector` | `demo_temperature_detector.yml` | `qwen2.5:3b` | 1.9 GB |
| `test_regressor` | `demo_temperature_detector.yml` | `qwen2.5:3b` | 1.9 GB |
| `test_classifier` | `demo_sentiment_classifier.yml` | `gemma3:12b` | 8.1 GB |
| `test_async` | all three configs above | `qwen2.5:3b`, `gemma3:12b` | — |
| `test_multi_model` | built programmatically | `qwen2.5:3b`, `qwen3:4b` | 1.9 / 2.6 GB |
| `test_pipeline` | `demo_pipeline.yml` (chains `demo_labeler.yml` → `demo_bio_generator.yml`) | `qwen2.5:3b` | 1.9 GB |
| `test_prompt_params` | `demo_labeler_params.yml` | `qwen3:4b` | 2.6 GB |

```bash
ollama pull qwen2.5:3b
ollama pull qwen3:4b
ollama pull gemma3:12b
```
