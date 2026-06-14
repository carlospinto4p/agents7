
# Backlog - Agents7

### Improvements — 14th June 2026

- [ ] CLI entry point — add an `agents7` console script (`[project.scripts]`) with `run` and `eval` subcommands to execute a YAML-configured agent against an input/file or a golden set, without spinning up the UI. Thin glue over the existing YAML loaders + `run_evaluation` / `export_eval_report`. (MED, M)
- [x] Call-log analysis helper — a reader/summary over `JsonCallLogger` JSONL output: aggregate success rate, total tokens + cost, and per-agent/per-model breakdown. Closes the loop on the structured call logging that already exists. (MED, S)
- [ ] Tool use / function calling — agents that can invoke registered tools (call APIs, read files, etc.). The single largest capability gap vs. typical agent frameworks, but it needs a design pass and per-provider plumbing, and sits outside the current extraction/roleplay focus. Confirm scope before starting. (HIGH, L)


### Found while working — 14th June 2026

- [x] An empty/incomplete `persona:` mapping in a Chatter YAML surfaces a raw `TypeError` (`full_name` missing) from `Persona.from_dict`, leaking out of `load_chatter_yaml_settings` (and `_load_base_chatter_config` for the guarded loaders). Wrap persona/scene construction to raise a friendly `AgentConfigurationError` instead. (MED, S)


### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.
