
# Backlog - Agents7

### Scan findings — 7th June 2026

Refactoring:

- [x] Duplicated memory-config block across the two guarded loaders (`config_loader.py`) — extract `_build_memory_config()`. (LOW, S)
- [x] Repeated mapper try/except error handling at 3 sites in `pipeline.py` (`act_stream`, `act_stream_async`, `_evaluate_batch_step`) — extract `_call_mapper_safe()`. (LOW, S)
- [x] Broad `except Exception` in `run_validators` (`src/agents7/agents/parsing.py:106`) silently converts every validator error to `None` — narrow to expected types so unexpected errors surface. (MED, M)

Test coverage:

- [ ] `forbidden_terms_for_language` case/whitespace handling (`.strip().lower()`) is untested — add lowercase, mixed-case, and whitespace cases (`context_guard.py`). (MED, S)
- [ ] `Chatter.from_yaml` persona/scene parsing edge cases untested directly: empty `persona` dict, `extra_instructions` stripping, and scene-without-persona raising via YAML (`config_loader.py` `load_chatter_yaml_settings`). (MED, M)


### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.
