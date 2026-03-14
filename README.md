
# Agents7

A Python library for managing LLM-based agents that process and label data using structured outputs (Pydantic models).

## Installation

```bash
pip install agents7
```

Requires Python >= 3.11.

## Design

See [docs/design.md](docs/design.md) for the rationale behind the
library's architecture: why reasons are required, how NOT_APPLICABLE
works, the structured output flow, and other design decisions.

See [docs/model-selection.md](docs/model-selection.md) for guidance
on choosing the right LLM for structured extraction tasks.

## Architecture

### LLM Layer (`agents7.llm`)

- `LLMEngine`: Facade class that routes to specific providers (supports optional `fallback_model`)
- `LLMProvider`: Abstract base class defining the interface
- `OllamaProvider`: Concrete implementation using Ollama (local LLM server)
- `OpenAICompatProvider`: Provider for any OpenAI-compatible API (OpenAI, Azure, vLLM, LM Studio, etc.)

Supports both sync and async methods: `chat`, `chat_async`, `generate`, `generate_async`, plus streaming variants: `chat_stream`, `chat_stream_async`, `generate_stream`, `generate_stream_async`.

Features: temperature control, structured output via JSON schema, "thinking" mode, seed for reproducibility, streaming responses, in-memory LRU caching, fallback model support.

### Agents Layer (`agents7.agents`)

```
BaseAgent
  └── Labeler (any Pydantic model)
        └── Detector (Property outputs)
              ├── Classifier (ClassificationProperty)
              └── Regressor (Property[T])

Pipeline (chains multiple agents sequentially)
```

- `BaseAgent`: Abstract base defining common interface (`name`, `version`, `engine`, `system_prompt`, `prompt_params`)
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
- `LLMMetrics`: Token counts and duration
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

### YAML Configuration

```yaml
base_config: configs/shared_base.yml  # optional: inherit from parent config
name: MyClassifier
version: 1
engine:
    provider: ollama
    model: qwen3
    fallback_model: qwen2.5:3b  # optional: retry with this model on failure
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

Note: YAML `prompt_params` only supports static values (cast to strings). For callable values, use Python code instantiation.

**Config inheritance:** When `base_config` is specified, the base YAML is loaded first and the current file's keys override it. Nested dicts (e.g. `engine`) are merged recursively. Chains are supported (A inherits B inherits C) and circular references are detected.

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

All `retry` fields are optional and fall back to sensible defaults. An empty `retry: {}` enables retries with defaults. Omitting the `retry` key entirely disables retries (backward compatible).

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

Delay formula: `min(base_delay * backoff_factor ^ attempt, max_delay)`.

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

When the primary model fails (after retries, if configured), the engine can automatically retry with a fallback model. If the fallback also fails, the primary error is raised.

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

The `api_key` falls back to the `OPENAI_API_KEY` environment variable when omitted. All agent features (streaming, caching, retry, fallback, batch) work identically with the OpenAI-compatible provider.

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
- `.metrics` — dict with `tokens`, `prompt_tokens`, `total_duration`

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

The original agent is unchanged (frozen dataclass). Works with all agent types (`Labeler`, `Detector`, `Classifier`, `Regressor`).

### Batch Export

Export agent responses to JSONL or CSV files after `act_in_batch`:

```python
from agents7.export import export_jsonl, export_csv

responses = agent.act_in_batch(inputs)

# Export to JSONL (one JSON object per line)
export_jsonl(responses, "results.jsonl")

# Export to CSV (auto-detects model fields)
export_csv(responses, "results.csv")

# Export specific fields only
export_csv(responses, "results.csv", fields=["category", "reason"])

# Export to Polars DataFrame (requires polars)
from agents7.export import export_dataframe

df = export_dataframe(responses)
df = export_dataframe(responses, fields=["category", "reason"])
```

### Evaluation Metrics

The `agents7.metrics` module provides evaluation primitives usable without Streamlit. Run an agent against labeled data and compute accuracy, per-field metrics, and confusion matrices.

`run_evaluation` uses `act_in_batch()` internally for efficient processing. If the batch fails, it falls back to sequential `act()` calls so individual errors don't break the entire evaluation. An async variant `run_evaluation_async` is also available.

```python
from agents7.metrics import (
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

**Evaluation reports:** Generate a formatted Markdown report from evaluation results:

```python
from agents7.export import export_eval_report

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

The UI evaluation module (`agents7.ui.evaluation`) re-exports from `agents7.metrics`, so existing UI code continues to work unchanged.

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

### Structured Call Logging

Optional per-call logging that captures trace ID, agent name, model, tokens, duration, and success/failure. Useful for debugging, observability, and cost tracking.

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

Each `act()` / `act_async()` call emits one `CallRecord`:

```json
{
  "trace_id": "a1b2c3d4e5f6",
  "agent_name": "Extractor",
  "agent_version": 1,
  "model_id": "ollama::qwen3",
  "success": true,
  "tokens": 42,
  "prompt_tokens": 128,
  "total_duration": 1.23,
  "error": null
}
```

**YAML configuration:**

```yaml
call_logger: mymodule.my_logger
```

The `call_logger` value is a dotted import path to any callable accepting a `CallRecord`.

### Agent Builder UI

> Full workflow documentation: [docs/agent_builder.md](docs/agent_builder.md)

An optional Streamlit-based UI for creating, testing, and evaluating agents interactively. Install with:

```bash
pip install agents7[ui]
```

Launch the UI:

```bash
agents7-ui
```

Custom host and port (useful for Docker / remote environments):

```bash
agents7-ui --host 0.0.0.0 --port 9000
```

Or from Python:

```python
from agents7.ui import launch
launch()                                  # defaults
launch(host="0.0.0.0", port=9000)        # custom
```

The wizard guides you through 5 steps:

| Step | Description |
|------|-------------|
| **Define** | Choose agent type (Classifier, Detector, Regressor, Labeler) and define the output model. Optionally use **Discover from Data** to auto-suggest the schema from sample JSONL |
| **Configure** | Select LLM model (auto-detected from Ollama), write the system prompt, add examples. Optionally **Generate Prompt & Examples** from intent and sample data |
| **Test** | Quick-test with single or batch inputs (text, JSON, or JSONL file upload), save prompt versions, and run A/B comparisons side-by-side |
| **Evaluate** | Upload CSV/JSONL or add labeled rows, run the agent, see accuracy, per-field metrics, and confusion matrix. Download results as CSV or JSONL |
| **Export** | Save generated `models.py`, `config.yml`, `eval_data.csv`, `eval_results.csv`, `eval_results.jsonl`, and `usage_example.py` to `workspace/<agent_name>/` |

The UI constructs agents in-memory during Test/Evaluate (no files written until Export). This enables fast prompt iteration without import path issues.

## Examples

The `examples/` directory contains runnable scripts demonstrating each feature:

| Script | Feature |
|--------|---------|
| `labeler.py` | Entity extraction with batch processing |
| `classifier.py` | Sentiment classification |
| `detector.py` | Temperature detection |
| `regressor.py` | Numeric value extraction |
| `labeler_params.py` | Prompt parameter substitution |
| `pipeline.py` | Multi-step pipeline with custom mapper |
| `config_inheritance.py` | YAML `base_config` inheritance |
| `conditional_pipeline.py` | Conditional step execution |
| `streaming.py` | Token-by-token streaming (sync + async) |
| `sqlite_cache.py` | Persistent SQLite cache |
| `export.py` | JSONL and CSV export |

Run from the `examples/` directory:

```bash
cd examples
python labeler.py
```

## Docker

Run the Agent Builder UI in Docker, connecting to Ollama on your host
machine:

```bash
# 1. Make sure Ollama is running locally
ollama serve

# 2. Copy and fill in your GitLab token
cp .env.example .env

# 3. Build and start the app
docker compose up --build
```

The Streamlit UI will be available at `http://localhost:8501`.
The container connects to Ollama on the host via
`host.docker.internal:11434`.

To stop the service:

```bash
docker compose down
```

## Requirements

- **Ollama provider**: Ollama running locally (default: `http://localhost:11434`)
- **OpenAI-compatible provider**: `uv pip install agents7[openai]` (requires `openai` package)
- Core dependencies: `pydantic`, `ollama`, `tqdm`, `pyyaml`, `cpinto`

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
