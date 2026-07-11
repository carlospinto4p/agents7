
# Backlog - Agents7

### Tool calling follow-ups — 21st June 2026 (v4.4.0 shipped Ollama-only v1)

- [ ] Streaming + tool calls — let `ToolAgent` stream intermediate steps/tokens during the loop. Out of v1 scope; each provider's streaming tool API is the hardest part. (LOW, L)


### Integration test timeouts — 11th July 2026

- [ ] Thinking models exceed the 60s per-test timeout in `integration_tests/test_multi_model.py` — `qwen3:4b` and `gemma4:e4b` both fail on `pytest-timeout` (>60s) because a cold model load plus a full thinking pass overruns `conftest._PER_TEST_TIMEOUT`. Surfaced now that Ollama is installed on the box (the suite used to skip entirely). Options: raise the per-test timeout for thinking models (parametrize a longer timeout marker), warm the model before the timed call, or pass `think=False` for the connectivity smoke test. Weigh against the 300s session cap. (MED, S)


### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.
