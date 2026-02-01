
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

Supports both sync and async methods: `chat`, `chat_async`, `generate`, `generate_async`.

Features: temperature control, structured output via JSON schema, "thinking" mode, seed for reproducibility.

### Agents Layer (`agents7.agents`)

```
BaseAgent
  └── Labeler (any Pydantic model)
        └── Detector (Property outputs)
              ├── Classifier (ClassificationProperty)
              └── Regressor (Property[T])
```

- `BaseAgent`: Abstract base defining common interface (`name`, `version`, `engine`, `system_prompt`, `prompt_params`)
- `Labeler`: General-purpose agent that outputs structured data via a Pydantic `output_model`
  - Single processing: `act()`, `act_async()`
  - Batch processing: `act_in_batch()`, `act_in_batch_async()`
  - YAML configuration support via `from_yaml()`
- `Detector`: Specialized `Labeler` for property extraction (value + reason)
- `Classifier`: For categorical classification with category + reason enums
- `Regressor`: For numeric value extraction with reason enum

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

# Run tests
cd tests && uv run pytest -v

# Run linter
ruff check .
```
