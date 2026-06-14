
# Backlog - Agents7

### Improvements — 14th June 2026

- [ ] CLI entry point — add an `agents7` console script (`[project.scripts]`) with `run` and `eval` subcommands to execute a YAML-configured agent against an input/file or a golden set, without spinning up the UI. Thin glue over the existing YAML loaders + `run_evaluation` / `export_eval_report`. (MED, M)
- [ ] Tool use / function calling — agents that can invoke registered tools (call APIs, read files, etc.). The single largest capability gap vs. typical agent frameworks, but it needs a design pass and per-provider plumbing, and sits outside the current extraction/roleplay focus. Confirm scope before starting. (HIGH, L)


### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.
