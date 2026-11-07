# Company Person Extractor

LLM-based labeler that extracts structured person
information from free text, scoped to a configured
company. Built with
[agents7](../..); requires an
LLM backend (Ollama by default).

People who are not relevant to the configured company are
returned with every field `null`.

## Output fields

| Field | Type | Notes |
|-------|------|-------|
| `name` | `str \| None` | Person's name |
| `age` | `int \| None` | Numeric age |
| `occupation` | `str \| None` | Concise job title |
| `location` | `str \| None` | City / country / region |

## Configuration

The company and the reference date are injected via
`prompt_params` in `config.yml`:

```yaml
prompt_params:
    company: "Acme Corp"
    current_date: "2026-02-01"
```

`$company` and `$current_date` are substituted into the
system prompt at run time.

## Usage

This is a **sample**, not a package — there is nothing to
install. Point the agent at the config:

```python
from agents7.agents import Labeler

agent = Labeler.from_yaml(
    "samples/company_extractor/config.yml"
)
response = agent.act(
    {"text": "John is a 35-year-old engineer at Acme Corp in Berlin."}
)
print(response.model.name)        # John
print(response.model.occupation)  # engineer
```

The output model is `agents7.models.demo.PersonInfo`, resolved
by dotted path from `config.yml` — the same model
`person_extractor` uses. The two samples differ only in their
prompt: this one scopes extraction to a configured company via
`prompt_params`.

## Evaluation

```bash
# from this sample's directory
uv run --project ../.. agents7 eval \
    -c config.yml -d data/golden_set.jsonl -r reports/company_extractor.md
```

The eval set is `data/golden_set.jsonl`. The report is written
to `reports/` (gitignored).

## Tests

The golden set is validated by
`tests/test_sample_golden_sets.py`.
