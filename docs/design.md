# Design — agents7

agents7 turns **unstructured text into structured data**. The
typical path is through an LLM: you describe the output shape
(a Pydantic model), write a system prompt, and the library
handles prompt construction, LLM invocation, response parsing,
enum mapping, validation, batching, and async support.

However, **not every agent needs an LLM**. The framework's
output models (`Property`, `ClassificationProperty`) and
evaluation tooling work equally well for rule-based or hybrid
agents. When the signal space is well-defined — name
dictionaries, keyword matching, grammatical patterns — a
deterministic classifier can be orders of magnitude faster
and fully transparent. The `tagger_agent` agent in the
catalog demonstrates this: it started as an LLM-based
classifier and evolved into a pure rule-based pipeline that
is 1000x faster with the same accuracy.

This document explains the design decisions behind the library.


## Core idea: structured extraction as agents

An "agent" in agents7 is a frozen configuration that knows how to
turn a piece of text into a typed Pydantic model. It is not an
autonomous loop or a tool-using agent — it is a single,
deterministic LLM call wrapped in validation. The name "agent"
reflects that each one encapsulates a specific analytical task
(detect, classify, regress, label) and can be composed into
pipelines.


## Agent hierarchy

```
BaseAgent (abstract — name, version, engine, system_prompt)
├── Labeler (general-purpose structured output)
│     └── Detector (boolean property + reason)
│           ├── Classifier (category + reason)
│           ├── Regressor (numeric value + reason)
│           └── ContextGuard (response validation)
└── Chatter (multi-turn conversation; optional persona + scene)
      └── GuardedChatter (+ ContextGuard pipeline)
            └── GuardedSceneChatter (persona + scene + guard)
```

The tree has two branches: **Labeler** for structured data
extraction and **Chatter** for free-form conversation. See
[`docs/chatter.md`](chatter.md) for the conversational branch
(including `GuardedChatter` and `ContextGuard`).

Each level adds a constraint:

| Agent | Output model must be | Adds to prompt |
|-------|---------------------|----------------|
| **Labeler** | Any `BaseModel` | Output schema |
| **Detector** | `Property` subclass | "Stick to the reasons defined above" |
| **Classifier** | `ClassificationProperty` | "Classify into one of the categories" + category-reason constraints |
| **Regressor** | `Property` subclass | "Extract the numeric value" |

**Why inheritance instead of configuration?** Each agent class carries
its own prompt wording, validation logic, and property accessors
(`possible_reasons`, `possible_categories`). Inheritance lets
subclasses override only what differs while reusing all of
`Labeler`'s machinery: parsing, batching, async, caching, retry.


## Why reasons are required

Every `Property`-based agent (Detector, Classifier, Regressor)
requires a **reason enum** alongside the extracted value:

```python
class TemperatureReason(StrEnum):
    TOO_COLD = "The temperature is too cold."
    COMFORTABLE = "The temperature is comfortable."
    TOO_HOT = "The temperature is too hot."

class Temperature(Property[float]):
    reason: TemperatureReason
```

Reasons serve three purposes:

1. **Explainability.** A bare classification or detection is
   opaque — the consumer cannot tell *why* the LLM chose that
   label. Requiring a reason forces the model to justify its
   decision in terms the developer defined upfront.

2. **Constraint.** The reason enum limits the LLM to a closed
   set of justifications. Without it, the model might hallucinate
   novel explanations that downstream code cannot handle.

3. **Debuggability.** When an agent misclassifies, the reason
   field reveals the model's logic. "Category: NEGATIVE,
   Reason: FRUSTRATED" is far more actionable than
   "Category: NEGATIVE" alone.


## NOT_APPLICABLE: the universal fallback

All reason enums (and category enums in classifiers)
automatically receive a `NOT_APPLICABLE` member at class
definition time, via `Property.__init_subclass__()`:

```python
class Property(BaseModel, Generic[T]):
    def __init_subclass__(cls, **kwargs):
        # Extend the reason enum with NOT_APPLICABLE
        ...
```

**Why auto-inject it?**

- **Graceful degradation.** If the input text contains no
  relevant information, the LLM needs a way to say "this
  doesn't apply" instead of guessing.
- **Reduced boilerplate.** Users don't need to remember to add
  it to every enum — it's always there.
- **Test coverage.** Every test suite can include NOT_APPLICABLE
  cases without defining special enums.

The `validate_category_reason_link()` validator on
`ClassificationProperty` ensures NOT_APPLICABLE can pair with
any category, while non-NOT_APPLICABLE reasons must match their
category's allowed list.


## Structured output flow

Every `act()` call follows the same pipeline:

```
1. Build prompt
   ├── resolved_system_prompt (template substitution via $placeholders)
   ├── output model schema (auto-generated from Pydantic model)
   ├── agent-specific instruction (e.g., "Classify into categories")
   ├── extra sections (e.g., category-reason constraints)
   └── examples (from YAML config)

2. Call LLM
   ├── temperature = 0 (deterministic)
   ├── output_format = model JSON schema (structured output)
   └── think = agent.think (default False; opt-in reasoning)

3. Parse response
   ├── Fix malformed JSON (strip markdown fences, fix doubled braces)
   ├── Validate against Pydantic model
   ├── If validation fails: map enum names → values and retry
   └── If still fails: return None

4. Run validators
   └── Each callable receives and returns a BaseModel (or raises to reject)

5. Return AgentResponse
   ├── model: the validated Pydantic instance (or None)
   ├── response: raw LLM text
   ├── thinking: chain-of-thought (if the model supports it)
   └── metrics: tokens, prompt tokens, duration
```


### Why temperature = 0?

Agents are analytical tools, not creative generators.
Reproducibility matters: the same input should produce the same
output. Temperature 0 makes LLM calls as deterministic as the
model allows.


### Why enum name-to-value mapping?

Some LLMs return enum member names (`"TOO_COLD"`) instead of
values (`"The temperature is too cold."`). Since Pydantic
validates by value, this would fail. The parser detects this
pattern and maps names to values before a second validation
attempt.


## YAML configuration

Agents are instantiated from YAML files via `Agent.from_yaml()`:

```yaml
name: TemperatureDetector
version: 1
engine:
    provider: ollama
    model: qwen2.5:3b
system_prompt: |
    Analyze the input and determine the temperature.
output_model: mypackage.models.Temperature
system_examples:
    - input: "It's freezing outside."
      output:
          value: -5
          reason: TOO_COLD
```

**Why YAML over Python-only?**

- **Separation of concerns.** The output model is code (Pydantic
  class with enums and validation). The prompt and examples are
  configuration. Keeping them apart means you can iterate on
  prompts without touching code.
- **Version control.** YAML diffs are readable. Prompt changes
  show up clearly in git history.
- **Accessibility.** Domain experts can edit prompts and examples
  without writing Python.

The `output_model` field is a dotted import path
(`"mypackage.models.Temperature"`). The config loader dynamically
imports it at load time, so the YAML stays declarative.


### Prompt parameters

Static YAML prompts can include `$placeholder` variables filled
at runtime via `prompt_params`:

```yaml
prompt_params:
    language: English
    current_date: "2026-03-08"
```

Values can also be **callables** (in Python, not YAML), enabling
dynamic prompts like `{"today": lambda: date.today().isoformat()}`.


## Batch processing

Processing many inputs one-by-one is slow. Batch methods solve
this by **chunking** inputs and sending each chunk as a single
LLM call:

```
1000 inputs
  → 100 chunks of 10 (requests_per_prompt=10)
  → 100 LLM calls (each returns a JSON array of 10 outputs)
  → flatten back to 1000 AgentResponse objects
```

Two variants:

- `act_in_batch()` — synchronous, with `tqdm` progress bar
- `act_in_batch_async()` — concurrent, with configurable
  `max_concurrent` (default 5) to avoid overwhelming the LLM
  server

The batch prompt explicitly tells the LLM to return an array,
and the parser validates against a dynamically-created
`OutputModelList` wrapper.


## Pipeline

A `Pipeline` chains agents sequentially. Each step's output is
mapped to the next step's input:

```python
Pipeline(steps=[
    PipelineStep(agent=extractor),
    PipelineStep(agent=classifier, mapper=my_mapper),
])
```

The default mapper calls `response.model.model_dump()`, but
custom mappers can filter, transform, or reshape data between
steps.

**Failure handling:** If any step returns `None` (parsing
failure) or the mapper raises, the pipeline stops and marks
the result as failed. All executed steps are preserved for
debugging.


## Frozen dataclasses

All agent classes use `@dataclass(frozen=True)`:

- **Thread safety.** Immutable objects are safe to share across
  threads and async tasks.
- **Intentionality.** An agent is a fixed configuration. You
  don't mutate it after creation — you create a new one.
- **Hashability.** Frozen dataclasses can serve as cache keys.

Private attributes (parser, cached schemas) are set in
`__post_init__` via `object.__setattr__()` to bypass the
frozen restriction during initialization only.


## LLM engine architecture

```
LLMEngine (retry, cache, sync/async dispatch)
└── LLMProvider (abstract)
    └── OllamaProvider (Ollama API)
```

The engine adds cross-cutting concerns:

- **Retry** with exponential backoff for transient errors
- **Response cache** (LRU, configurable size) for temperature=0
  calls — since the output is deterministic, identical prompts
  return cached results
- **Streaming** support for real-time token output

The provider abstraction exists so that adding OpenAI, Claude,
or other backends requires only a new `LLMProvider` subclass,
with no changes to agent code.


## Thinking mode

The `think` field on `Labeler` (and all subclasses) controls whether
the LLM reasons before generating its structured output. It defaults
to `False`.

**Why default to False?**

Thinking mode asks the model to produce a chain-of-thought before
answering. For complex, open-ended tasks this improves quality.
But for constrained extraction — which is what agents7 is designed
for — thinking often hurts:

- **Over-reasoning.** The model considers edge cases that don't
  exist, second-guesses obvious classifications, and finds
  creative justifications for wrong answers (see
  [model-selection.md](model-selection.md)).
- **NOT_APPLICABLE resistance.** Extra reasoning gives the model
  more room to talk itself out of "not applicable" and into a
  wrong answer.
- **Latency.** Thinking tokens add to response time without
  improving accuracy on classification and extraction tasks.

Users can opt in per agent (`think: true` in YAML or
`think=True` in Python) when the task genuinely benefits from
reasoning — for example, tasks requiring domain knowledge or
multi-step inference. The setting can also be overridden per
call via `chat_kwargs={"think": True}`.


## NOT_APPLICABLE prompt guidance

All `Property`-based agents (Detector, Classifier, Regressor)
inject an explicit guidance section into their prompts:

> Use NOT_APPLICABLE when the input has no connection to the
> property being detected. Do not infer, speculate, or make
> assumptions beyond what is explicitly stated or clearly
> implied.

This guidance addresses a common failure mode: larger models
resist saying "not applicable" because their RLHF training
rewards providing an answer. The explicit instruction gives the
model permission to classify as NOT_APPLICABLE and discourages
speculative reasoning.

See [model-selection.md](model-selection.md) for detailed
analysis of why this matters.


## Fallback model

The `LLMEngine` supports an optional `fallback_model`:

```yaml
engine:
    provider: ollama
    model: gemma3:12b
    fallback_model: qwen2.5:3b
```

When the primary model fails after exhausting retries, the engine
automatically retries the same call with the fallback model. If
the fallback also fails, the primary error is raised.

**Use cases:**

- **Resilience.** If the primary model is unavailable or
  overloaded, the fallback provides continuity.
- **Model-specific failures.** Some models fail on certain inputs
  (structured output format issues, thinking mode not supported).
  A fallback with different characteristics can handle these
  cases.
- **Cost/quality tradeoff.** Use a larger model as primary for
  quality, with a smaller fallback for availability.


## Summary of design principles

| Principle | How |
|-----------|-----|
| **Structured over free-text** | Pydantic models define the output shape; the LLM fills it |
| **Explainable by default** | Reason enums force the LLM to justify every decision |
| **Fail-safe** | NOT_APPLICABLE auto-injected with prompt guidance; multi-stage parsing recovery; fallback model |
| **Deterministic** | Temperature 0; response caching; frozen agent configs |
| **Minimal reasoning** | Thinking mode off by default; smaller models preferred for constrained tasks |
| **Config-driven** | YAML for prompts and examples; Python for models and logic |
| **Scalable** | Batch chunking; async concurrency; semaphore limits |
| **Composable** | Pipelines chain agents; validators add post-processing |
