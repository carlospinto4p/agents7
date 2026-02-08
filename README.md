
# Agents7

A Python library for managing LLM-based agents that process and label data using structured outputs (Pydantic models).

## Installation

```bash
pip install agents7
```

Requires Python >= 3.11.

## Architecture

### LLM Layer (`agents7.llm`)

- `LLMEngine`: Facade class that routes to specific providers
- `LLMProvider`: Abstract base class defining the interface
- `OllamaProvider`: Concrete implementation using Ollama (local LLM server)

Supports both sync and async methods: `chat`, `chat_async`, `generate`, `generate_async`, plus streaming variants: `chat_stream`, `chat_stream_async`, `generate_stream`, `generate_stream_async`.

Features: temperature control, structured output via JSON schema, "thinking" mode, seed for reproducibility, streaming responses, in-memory LRU caching.

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
  - Batch processing: `act_in_batch()`, `act_in_batch_async()` (with optional `BatchSettings` concurrency control)
  - YAML configuration support via `from_yaml()`
- `Detector`: Specialized `Labeler` for property extraction (value + reason)
- `Classifier`: For categorical classification with category + reason enums
- `Regressor`: For numeric value extraction with reason enum
- `Pipeline`: Chains multiple agents sequentially with optional custom mappers between steps (async batch supports `BatchSettings`)

### Models Layer (`agents7.models`)

- `AgentResponse`: Wraps LLM response with parsed model, thinking, metrics, and index
- `LLMMetrics`: Token counts and duration
- `Property[T]`: Generic base for value extraction (value of type T + reason enum)
- `ClassificationProperty`: Base for classification tasks (category enum + reason enum)

Both `Property` and `ClassificationProperty` automatically add `NOT_APPLICABLE` to their reason enums.

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

detector = Regressor.from_yaml("configs/demo_detector.yml")

texts = [
    "It's freezing outside at -5 degrees!",
    "The room is at a comfortable 22 degrees.",
]

inputs = [{"text": text} for text in texts]
responses = detector.act_in_batch(inputs)

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
name: MyClassifier
version: 1
engine:
    provider: ollama
    model: qwen3
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

### Retry / Resilience

LLM transport errors (connection failures, server 5xx) can be retried automatically with exponential backoff. Client errors (4xx) and parse failures are **not** retried.

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

### Caching

Deterministic LLM calls (`temperature=0`) can be cached in memory with an LRU eviction policy. Cached responses are returned instantly without hitting the LLM. When `temperature != 0`, caching is bypassed automatically.

Streaming responses are also cached: on a miss the result is stored after the stream is fully consumed; on a hit the full response is returned as a single chunk.

The cache is in-memory only and lost on process restart.

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

### Agent Builder UI

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
| **Define** | Choose agent type (Classifier, Detector, Regressor, Labeler) and define the output model |
| **Configure** | Select LLM model (auto-detected from Ollama), write the system prompt, add examples |
| **Test** | Quick-test with single or batch inputs, view structured output and metrics |
| **Evaluate** | Upload CSV or add labeled rows, run the agent, see accuracy, per-field metrics, and confusion matrix |
| **Export** | Save generated `models.py`, `config.yml`, `eval_data.csv`, and `usage_example.py` to `workspace/<agent_name>/` |

The UI constructs agents in-memory during Test/Evaluate (no files written until Export). This enables fast prompt iteration without import path issues.

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

- Ollama running locally (default: `http://localhost:11434`)
- Dependencies: `pydantic`, `ollama`, `tqdm`, `pyyaml`, `cpinto`

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
cd integration_tests && uv run pytest -v

# Run linter
ruff check .
```
