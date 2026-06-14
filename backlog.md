
# Backlog - Agents7

### Found while working — 14th June 2026

- [x] An empty/incomplete `persona:` mapping in a Chatter YAML surfaces a raw `TypeError` (`full_name` missing) from `Persona.from_dict`, leaking out of `load_chatter_yaml_settings` (and `_load_base_chatter_config` for the guarded loaders). Wrap persona/scene construction to raise a friendly `AgentConfigurationError` instead. (MED, S)


### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.
