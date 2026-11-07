# Person Extractor

LLM-based labeler that extracts structured person
information (name, age, occupation, location) from free
text. Built with
[agents7](../..); requires an
LLM backend (Ollama by default).

Unlike `company_extractor`, this agent is unscoped — it
extracts any person mentioned, with no company filter.

## Output fields

| Field | Type | Notes |
|-------|------|-------|
| `name` | `str \| None` | Person's name |
| `age` | `int \| None` | Numeric age |
| `occupation` | `str \| None` | Concise job title |
| `location` | `str \| None` | City / country / region |

## Usage

This is a **sample**, not a package — there is nothing to
install. Point the agent at the config:

```python
from agents7.agents import Labeler

agent = Labeler.from_yaml(
    "samples/person_extractor/config.yml"
)
response = agent.act(
    {"text": "John is a 35-year-old software engineer in Berlin."}
)
print(response.model.name)        # John
print(response.model.location)    # Berlin
```

The output model is `agents7.models.demo.PersonInfo`, resolved
by dotted path from `config.yml`.

## Evaluation

```bash
# from this sample's directory
uv run --project ../.. agents7 eval \
    -c config.yml -d data/golden_set.jsonl -r reports/person_extractor.md
```

The eval set is `data/golden_set.jsonl`. The report is written
to `reports/` (gitignored).

## Tests

The golden set is validated by
`tests/test_sample_golden_sets.py`.
