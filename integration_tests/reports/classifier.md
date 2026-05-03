# Evaluation Report

- **Agent**: DemoSentimentClassifier
- **Model**: ollama::gemma3:12b
- **Date**: 2026-04-11 23:38 UTC

## Notes

- NOT_APPLICABLE is **disabled** on the Sentiment model (`NOT_APPLICABLE = False`). The NEUTRAL category covers inputs with no clear sentiment.

## Overall Metrics

| Metric | Value |
|--------|-------|
| Total | 3 |
| Passed | 3 |
| Failed | 0 |
| Accuracy | 100.0% |

## Per-Field Accuracy

| Field | Accuracy |
|-------|----------|
| category | 100.0% |

## Response Times

| Metric | Value |
|--------|-------|
| Total | 19.55s |
| Average | 6.52s |
| Min | 6.52s |
| Max | 6.52s |

## Confusion Matrix (category)

| Expected \\ Actual | NEGATIVE | NEUTRAL | POSITIVE |
|---|---|---|---|
| NEGATIVE | 1 | 0 | 0 |
| NEUTRAL | 0 | 1 | 0 |
| POSITIVE | 0 | 0 | 1 |

## Per-Case Results

### Case 1 — PASS (6.52s)

- **Input**: {'text': 'I absolutely love this! Best day ever!'}
- **Expected**: {'category': 'POSITIVE'}
- **Actual**: {'category': 'POSITIVE'}

### Case 2 — PASS (6.52s)

- **Input**: {'text': "This is terrible and I'm very disappointed."}
- **Expected**: {'category': 'NEGATIVE'}
- **Actual**: {'category': 'NEGATIVE'}

### Case 3 — PASS (6.52s)

- **Input**: {'text': 'The meeting is at 3 PM in conference room B.'}
- **Expected**: {'category': 'NEUTRAL'}
- **Actual**: {'category': 'NEUTRAL'}
