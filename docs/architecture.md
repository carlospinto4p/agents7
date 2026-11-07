# Architecture

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
- `LLMMetrics`: Token counts, duration, and estimated cost — see [Cost Estimation](operations.md#cost-estimation)
- `Property[T]`: Generic base for value extraction (value of type T + reason enum)
- `ClassificationProperty`: Base for classification tasks (category enum + reason enum)

Both `Property` and `ClassificationProperty` automatically add `NOT_APPLICABLE` to their enum fields by default. Set `NOT_APPLICABLE = False` on the subclass to disable this (e.g., when a NEUTRAL category already covers the "no match" case).
