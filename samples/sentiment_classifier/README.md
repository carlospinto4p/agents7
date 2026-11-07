# Sentiment Classifier

LLM-based classifier that labels text with a sentiment
category and a justification reason. Built with
[agents7](../..); requires an
LLM backend (Ollama by default).

## Categories

| Category | Meaning |
|----------|---------|
| POSITIVE | Positive sentiment expressed |
| NEGATIVE | Negative sentiment expressed |
| NEUTRAL | Neutral or objective tone |

Each category maps to a fixed set of reasons (see
`CATEGORY_REASONS` on `agents7.models.demo.Sentiment`):

| Category | Reasons |
|----------|---------|
| POSITIVE | `ENTHUSIASTIC`, `GRATEFUL`, `HOPEFUL` |
| NEGATIVE | `FRUSTRATED`, `DISAPPOINTED`, `ANGRY` |
| NEUTRAL | `FACTUAL`, `INFORMATIVE`, `DESCRIPTIVE` |

`NOT_APPLICABLE` is disabled — `NEUTRAL` already covers
text with no clear sentiment.

## Usage

This is a **sample**, not a package — there is nothing to
install. Point the agent at the config:

```python
from agents7.agents import Classifier

agent = Classifier.from_yaml(
    "samples/sentiment_classifier/config.yml"
)
response = agent.act({"text": "Best purchase ever!"})
print(response.model.category.name)  # POSITIVE
print(response.model.reason.name)    # ENTHUSIASTIC
```

The output model is `agents7.models.demo.Sentiment`, resolved
by dotted path from `config.yml`.

## Evaluation

Run it against its golden set with the framework's `agents7 eval`
command, from this sample's directory:

```bash
uv run --project ../.. agents7 eval \
    -c config.yml -d data/golden_set.jsonl -r reports/sentiment_classifier.md
```

The eval set is `data/golden_set.jsonl` (9 rows, scored on
`category`). The report is written to `reports/` (gitignored).

## Tests

The golden set is validated by
`tests/test_sample_golden_sets.py`, which checks every
sample's rows against its declared `output_model`.
