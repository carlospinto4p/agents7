
# Backlog - Agents7

### 2026.06.09

- [ ] Remove (or repair) the orphaned `agents7_framework/docker-compose.yml`. Its `app` service has `build: context: .` but there is **no Dockerfile** in `agents7_framework/` — the Dockerfile was deleted in v3.19.0 ("replace Streamlit UI with Svelte app scaffold"), so `docker compose build/up` here fails. The compose still targets the old Streamlit port (8501); the live app now builds from `agents7_ui/`. Delete this dead compose (and its `.dockerignore`) unless the framework is meant to ship its own image, in which case restore the Dockerfile. — LOW — S

### Scan findings — 7th June 2026

Performance:

- [ ] `JsonCallLogger.__call__` opens the log file per call with no buffering (`src/agents7/call_log.py:100`). For high-throughput logging, batch writes or hold a persistent handle with periodic flush. (MED, M)

Refactoring:

- [ ] Duplicated `cast_optional_fields` field maps in `load_guarded_yaml_settings` and `load_guarded_scene_yaml_settings` (`src/agents7/agents/config_loader.py`) — extract the shared field map to a constant/helper. (MED, M)
- [ ] Duplicated memory-config block across the two guarded loaders (`config_loader.py`) — extract `_build_memory_config()`. (LOW, S)
- [ ] Repeated mapper try/except error handling at 3 sites in `pipeline.py` (`act_stream`, `act_stream_async`, `_evaluate_batch_step`) — extract `_call_mapper_safe()`. (LOW, S)
- [ ] Broad `except Exception` in `run_validators` (`src/agents7/agents/parsing.py:106`) silently converts every validator error to `None` — narrow to expected types so unexpected errors surface. (MED, M)

Test coverage:

- [ ] `forbidden_terms_for_language` case/whitespace handling (`.strip().lower()`) is untested — add lowercase, mixed-case, and whitespace cases (`context_guard.py`). (MED, S)
- [ ] `Chatter.from_yaml` persona/scene parsing edge cases untested directly: empty `persona` dict, `extra_instructions` stripping, and scene-without-persona raising via YAML (`config_loader.py` `load_chatter_yaml_settings`). (MED, M)


### Chatter/Guard/Scene design review — 7th June 2026

- [x] Two idioms for persona/scene. Since `GuardedChatter` extends `Chatter` directly (composition plugins), `PersonalityChatter`/`SceneChatter` (enforcement via subclass) are a parallel idiom. Consider folding them into a validated factory over `Chatter`. Public-API change — major bump. (LOW, medium) (Done agents7 v4.0.0 — full removal; persona/scene are Chatter plugins, `Chatter.from_yaml` parses them; UI v0.17.1 remaps legacy agent_types; catalog example configs migrated)


### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.
