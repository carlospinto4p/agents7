
# Backlog - Agents7

### Integration tests — 23rd August 2026

- [ ] `integration_tests/test_prompt_params.py::test_rejects_non_company_person` fails consistently (reproduced twice, same output both times) against the current Ollama model: `CompanyPersonExtractor` returns `name: "Maria"` for a person the config's `prompt_params`-scoped prompt should null out as not belonging to the configured company, instead of `null`. Found while running the full integration suite for an unrelated change (`agents7.tools.demo`); confirmed unrelated — the test predates that change and doesn't touch `ToolAgent`/tools. Needs investigation: either the prompt-scoping instruction needs strengthening for the current model, or the model/config drifted since this test last passed. (HIGH, S — investigate root cause first)
