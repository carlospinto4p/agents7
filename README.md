
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

- `BaseAgent`: Abstract base defining common interface (`name`, `version`, `engine`, `system_prompt`)
- `Labeler`: General-purpose agent that outputs structured data via a Pydantic `output_model`
  - Single processing: `act()`, `act_async()`
  - Batch processing: `act_in_batch()`, `act_in_batch_async()`
  - YAML configuration support via `from_yaml()`
- `Detector`: Specialized `Labeler` with output constrained to `Property` subclass (value + reason enum)

### Models Layer (`agents7.models`)

- `AgentResponse`: Wraps LLM response with parsed model, thinking, metrics, and index
- `LLMMetrics`: Token counts and duration
- `Property`: Base class for detectable properties (value + reason enum)

## Quick Start

### Using a Labeler

```python
from agents7.agents import Labeler

labeler = Labeler.from_yaml("path/to/config.yml")
response = labeler.act({"text": "Process this text"})
print(response.model)
```

### Using a Detector

```python
from agents7.agents import Detector

detector = Detector.from_yaml("path/to/config.yml")
response = detector.act({"text": "It's freezing outside!"})
print(f"Value: {response.model.value}, Reason: {response.model.reason}")
```

### YAML Configuration

```yaml
name: MyAgent
version: 1
engine:
    provider: ollama
    model: qwen3
system_prompt: |
  Your system prompt here.
output_model: mymodule.MyPydanticModel
system_examples:
    - input: Example input
      output: Example output
```

## Requirements

- Ollama running locally (default: `http://localhost:11434`)
- Dependencies: `pydantic`, `ollama`, `tqdm`, `pyyaml`, `cpinto`

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check .
```
