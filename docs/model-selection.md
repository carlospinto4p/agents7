# Model selection for structured extraction

When using agents7 to extract structured data from text, the
instinct is to reach for the largest, most capable model
available. In practice, **smaller models consistently outperform
larger ones** at the constrained extraction tasks this library
is designed for.

This document explains why, and offers guidance on choosing the
right model.


## The paradox: bigger is not better

agents7's integration tests compare models of different sizes on
identical structured extraction tasks (temperature detection,
sentiment classification, value regression). The results are
consistent: lightweight models like `qwen2.5:3b` handle every
test case reliably, while heavier models like `gemma3:12b`
struggle with edge cases — particularly NOT_APPLICABLE.

This is not a fluke. It reflects a fundamental mismatch between
what large models are optimized for and what structured
extraction requires.


## Why large models underperform

### 1. Large models are trained to be helpful, not constrained

Modern LLMs are heavily fine-tuned with RLHF (reinforcement
learning from human feedback) to be helpful, verbose, and
engaging. A large model presented with "She bought a new book"
and asked to detect a temperature will *try to find one* — it
has been trained to always produce a useful answer. It might
infer room temperature, guess based on the season implied by
"book shopping", or extrapolate from cultural context.

A smaller model, with less capacity for creative reasoning,
is more likely to take the instruction literally: "There is no
temperature information here. Reason: NOT_APPLICABLE."

**The very capability that makes large models better at open-ended
tasks makes them worse at knowing when to stop.**


### 2. Structured output is a constraint satisfaction problem

Structured extraction is not a generative task. The model is not
writing a story or answering a question — it is filling in a
form with a fixed schema and a closed set of enum values. This
is closer to classification than to generation.

For constraint satisfaction:
- The model needs to understand the schema (small models can)
- The model needs to pick from a fixed set of options (small
  models can)
- The model needs to output valid JSON (small models can, especially
  with structured output / format enforcement)
- The model does **not** need world knowledge, nuance, or
  creative reasoning

Large models bring excess capacity that manifests as
over-interpretation rather than better accuracy.


### 3. Over-reasoning leads to wrong answers

When a large model "thinks harder" about a simple classification,
it often talks itself into the wrong answer. Given "The meeting
is at 3 PM in conference room B" and asked to classify sentiment,
a large model might reason:

> "Meetings can be stressful. Conference rooms suggest a formal
> setting. This could be NEGATIVE with reason FRUSTRATED..."

A smaller model, lacking the capacity for this chain of
speculation, simply classifies it as NEUTRAL/FACTUAL — which is
correct.

This is especially visible with NOT_APPLICABLE cases. Large
models resist saying "not applicable" because their training
rewards providing an answer. They will find a tenuous connection
rather than admit the input doesn't match.


### 4. Structured output format enforcement helps small models

When agents7 sends a JSON schema as the `output_format`, the LLM
provider (Ollama) constrains the model's output to valid JSON
matching that schema. This eliminates the main advantage of large
models in raw generation (coherent free-text) and levels the
playing field. The model only needs to choose the right enum
value, not generate well-formed prose.

With format enforcement, a 3B parameter model produces the same
JSON structure as a 12B model — the only question is whether it
picks the right values. And for constrained classification tasks,
smaller models pick correctly more often.


### 5. Thinking mode amplifies the problem

agents7 enables thinking mode (`think=True`) by default, which
lets the model reason before answering. For large models, this
extra reasoning step often leads to over-analysis:

- The model considers edge cases that don't exist
- It second-guesses the obvious classification
- It finds creative justifications for wrong answers

Smaller models use the thinking step more directly: they
restate the task, check the options, and pick one. Less
reasoning capacity means less room for the model to convince
itself of the wrong answer.


## Practical guidance

### Choose the smallest model that handles your schema

Start with a small model (3B-4B parameters) and only move up if
it genuinely fails on your task. Signs you need a larger model:

- The schema requires understanding complex relationships
  between fields
- The input text is in multiple languages and the small model
  doesn't support them
- The task requires domain knowledge (medical, legal, technical
  terminology)

Signs you do **not** need a larger model:

- The task is classification into a fixed set of categories
- The task is detecting a property (yes/no + reason)
- The task is extracting a value from clearly stated text
- The schema has fewer than 10 enum members

### Test every enum value

The most common failure mode is a model that handles 90% of
cases but fails on NOT_APPLICABLE or a rare category. Always
test every enum value in your integration tests. Never weaken
tests to accommodate a model — switch to a model that passes.

### Prefer instruction-tuned models over raw models

Instruction-tuned models (the default for most Ollama models)
follow schemas more reliably than base models. Within
instruction-tuned models, prefer those explicitly trained for
structured output or function calling.

### Watch for signs of over-reasoning

If your model is getting creative with classifications —
finding sentiment in factual statements, detecting temperatures
in unrelated text, assigning categories through loose
associations — it is too large for the task. Drop to a smaller
model.


## Models tested with agents7

These observations come from the library's integration test
suite, which runs real LLM calls against Ollama:

| Model | Size | Structured extraction | NOT_APPLICABLE | Notes |
|-------|------|----------------------|----------------|-------|
| `qwen2.5:3b` | 3B | Reliable | Reliable | Good default for detection and regression |
| `qwen3:4b` | 4B | Reliable | Reliable | Good for labeling and pipelines |
| `gemma3:4b` | 4B | Reliable | Reliable | Works for single-case detection |
| `gemma3:12b` | 12B | Works | **Unreliable** | Over-interprets; avoids NOT_APPLICABLE |

The pattern is clear: 3B-4B models handle constrained extraction
tasks as well as or better than 12B models, at a fraction of the
compute cost and latency.


## Summary

Structured extraction is not a task that benefits from scale.
The ideal model for agents7 is small enough to follow
instructions literally, capable enough to understand the schema,
and disciplined enough to say "not applicable" when the input
doesn't match. Larger models bring creative reasoning that
actively works against these requirements.

**Use the smallest model that passes your full test suite —
including NOT_APPLICABLE cases.**
