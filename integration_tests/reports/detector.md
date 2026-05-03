# Evaluation Report

- **Agent**: DemoTemperatureDetector
- **Model**: ollama::qwen2.5:3b
- **Date**: 2026-04-11 23:39 UTC

## Notes

- NOT_APPLICABLE is **enabled** on the output model and **tested** (included in test data).

## Overall Metrics

| Metric | Value |
|--------|-------|
| Total | 4 |
| Passed | 4 |
| Failed | 0 |
| Accuracy | 100.0% |

## Per-Field Accuracy

| Field | Accuracy |
|-------|----------|
| reason | 100.0% |

## Response Times

| Metric | Value |
|--------|-------|
| Total | 15.39s |
| Average | 3.85s |
| Min | 3.85s |
| Max | 3.85s |

## Per-Case Results

### Case 1 — PASS (3.85s)

- **Input**: {'text': 'This is a test text.'}
- **Expected**: {'reason': 'NOT_APPLICABLE'}
- **Actual**: {'reason': 'NOT_APPLICABLE'}

### Case 2 — PASS (3.85s)

- **Input**: {'text': 'The water in the lake is starting to freeze.'}
- **Expected**: {'reason': 'TOO_COLD'}
- **Actual**: {'reason': 'TOO_COLD'}

### Case 3 — PASS (3.85s)

- **Input**: {'text': "It's scorching outside; I can barely stay in the sun."}
- **Expected**: {'reason': 'TOO_HOT'}
- **Actual**: {'reason': 'TOO_HOT'}

### Case 4 — PASS (3.85s)

- **Input**: {'text': 'A light breeze makes the evening feel very pleasant.'}
- **Expected**: {'reason': 'COMFORTABLE'}
- **Actual**: {'reason': 'COMFORTABLE'}
