# Development

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
