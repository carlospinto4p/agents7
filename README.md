# agents7

A Python library for LLM agents that answer with a Pydantic model
instead of free text. You describe the agent in a YAML file: the model,
the prompt, the output model and a few examples. agents7 calls the
model and checks its answer against the output model.

It runs on Ollama by default, and on any OpenAI-compatible API or on
Anthropic's.

## Agent types

| Class | Returns |
|-------|---------|
| `Labeler` | any Pydantic model, e.g. the fields of a person in a text |
| `Classifier` | a category and the reason for it |
| `Regressor` | a value and the reason for it |
| `Chatter` | replies in a multi-turn chat, optionally in character |
| `ToolAgent` | a free-text answer after calling Python tools |
| `Pipeline` | the output of several agents run in sequence |

## Install

agents7 is not on PyPI. Install a tagged release from GitHub:

```bash
pip install "agents7 @ git+https://github.com/carlospinto4p/agents7@v9.25.9"
```

Add the `openai` or `anthropic` extra for those providers. Python 3.11
or later.

## Example

A sentiment classifier, from the config in
[`examples/configs/demo_sentiment_classifier.yml`](examples/configs/demo_sentiment_classifier.yml):

```python
from agents7.agents import Classifier

classifier = Classifier.from_yaml(
    "examples/configs/demo_sentiment_classifier.yml"
)
response = classifier.act({"text": "This is the worst service ever."})

print(response.model.category.name)  # NEGATIVE
print(response.model.reason.name)    # e.g. FRUSTRATED
print(response.metrics)              # tokens, duration, cost if priced
```

`act_in_batch` takes a list of inputs, and every method has an async
version.

## Documentation

- [Building agents](docs/agents.md): the agent types, custom output
  models, prompt parameters, YAML configs and the command line.
- [LLM providers](docs/providers.md): Ollama, OpenAI-compatible APIs,
  Anthropic, fallback models, streaming, retries and timeouts.
- [Cost, caching and logging](docs/operations.md)
- [Pipelines](docs/pipelines.md)
- [Tool calling](docs/tools.md)
- [Evaluation](docs/evaluation.md): scoring an agent against a labeled
  set.
- [Chat agents](docs/chat-agents.md): personas, scenes, memory and
  style checks. [docs/chatter.md](docs/chatter.md) has more detail.
- [Architecture](docs/architecture.md) and the reasons behind it in
  [docs/design.md](docs/design.md).
- [Choosing a model](docs/model-selection.md)
- [Agent packages](docs/agent-packages.md): an agent in its own
  repository.
- [Development](docs/development.md): tests and the models they need.

[`examples/`](examples) has runnable scripts for each feature, and
[`samples/`](samples) has example agent configs.

## Related

- [agents7-ui](https://github.com/carlospinto4p/agents7-ui): a web UI to
  run, chat with and evaluate agents.
- [agent-sentiment_classifier](https://github.com/carlospinto4p/agent-sentiment_classifier):
  an agent packaged in its own repository.

## License

MIT
