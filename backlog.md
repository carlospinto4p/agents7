
# Backlog - Agents7

### Scan findings — 7th June 2026

Test coverage:

- [x] `forbidden_terms_for_language` case/whitespace handling (`.strip().lower()`) is untested — add lowercase, mixed-case, and whitespace cases (`context_guard.py`). (MED, S)
- [x] `Chatter.from_yaml` persona/scene parsing edge cases untested directly: empty `persona` dict, `extra_instructions` stripping, and scene-without-persona raising via YAML (`config_loader.py` `load_chatter_yaml_settings`). (MED, M)


### Found while working — 14th June 2026

- [ ] An empty/incomplete `persona:` mapping in a Chatter YAML surfaces a raw `TypeError` (`full_name` missing) from `Persona.from_dict`, leaking out of `load_chatter_yaml_settings` (and `_load_base_chatter_config` for the guarded loaders). Wrap persona/scene construction to raise a friendly `AgentConfigurationError` instead. (MED, S)


### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.
