# Building agents

## Using a Labeler (Entity Extraction)

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

## Using a Classifier (Sentiment Analysis)

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

## Using a Regressor (Value Extraction)

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

## Defining Custom Models

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

## Simple Classifier

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

## Prompt Parameterization

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

## Per-Conversation Context

`prompt_params` holds values the config fixes once. `context_params` declares values that change per conversation, such as who the agent is talking to, or per message, such as the time of day. Both substitute the same `$variable` placeholders; only the source of the value differs.

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

A value resolves from `set_context()` first, then the declared `callable` (run again before every message), then `default`. A declared name therefore never reaches the model as the literal `$name`. A keyword that matches no declaration raises `PromptParameterError` rather than being silently dropped.

Supplied values follow a copy made with `with_overrides()`. They are not a dataclass field, so a bare `dataclasses.replace(agent, ...)` drops them and the copy answers with each default. Copy an agent with `with_overrides()`.

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

`resolved_context` returns what the declared params currently resolve to. That is the mapping the prompt itself is built from, so the two cannot drift. It exists because those values are otherwise unrecoverable: a `callable` runs inside substitution and its result is discarded, so nothing afterwards can say which time of day or which nickname a finished conversation ran with, and at temperature a re-run is a different conversation. A `ConversationRecorder` captures it once when the conversation opens and saves it in the recording, alongside the character facts the store already held; a read then shows both above the transcript. Reading it twice can give different values, since a resolver is called each time.

A context param's **declaration** enters `system_prompt_identity`, never its resolved value. The declaration is the name, the default and where a `callable` comes from. A run's fingerprint hashes that identity, so an agent whose prompt carries the time of day keeps one measurement baseline instead of gaining a separate one for each part of the day, and a nickname a caller supplies is treated as input rather than as a different agent. Changing a `default` or swapping a `callable` does change the identity, because those are configuration.

## Prompt Variants

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
- `system_prompt_identity` is the prompt with the placeholders left in and every alternative listed. The style history hashes it, so runs that picked different texts still share a baseline. `context_params` are left in and declared after it for the same reason. See [Per-Conversation Context](#per-conversation-context).
- Each entry needs at least two different, non-empty texts, and a name cannot also be a `prompt_params` or `context_params` name.

## YAML Configuration

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

## Command-Line Interface

Installing the package exposes an `agents7` console script with two
subcommands for running and scoring agents headlessly, without any UI.
`run` is thin glue over the YAML loaders; `eval` is a thin wrapper over
[`run_eval_suite`](evaluation.md#evaluation-workflow), the same composed route a
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
**trajectory** rather than on output fields. See
[Evaluating Tool Agents](evaluation.md#evaluating-tool-agents). Golden sets may be
`.jsonl`, `.json`, or `.csv`,
and rows may be nested (`{"input": {...}, "category": "..."}`) or flat
(input and expected columns side by side); both are normalised
automatically. Expected fields are derived from the golden set and the
output model, or set explicitly with `--fields`.

The whole `eval` route is importable as one call, which
[Evaluation Workflow](evaluation.md#evaluation-workflow) describes. Its individual steps
stay public for finer control:

```python
from agents7.eval import build_agent, load_eval_rows, normalise_rows

agent = build_agent("config.yml", agent_type="classifier")
rows = load_eval_rows("data/golden_set.jsonl")
eval_rows, fields = normalise_rows(rows)
```

## Validation Hooks

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

## Input Preprocessor

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

## Agent Introspection

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

## Agent Cloning with Overrides

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

The original agent is unchanged (frozen dataclass). Defined on `BaseAgent`, so it works with **every** agent type: the structured-output agents (`Labeler`, `Detector`, `Classifier`, `Regressor`), the chatters, and `ToolAgent`. Accepted keyword arguments are whatever fields the concrete class defines.
