
# Backlog - Agents7

### Chatter/Guard/Scene design review — 7th June 2026

- [x] Two idioms for persona/scene. Since `GuardedChatter` extends `Chatter` directly (composition plugins), `PersonalityChatter`/`SceneChatter` (enforcement via subclass) are a parallel idiom. Consider folding them into a validated factory over `Chatter`. Public-API change — major bump. (LOW, medium) (Done agents7 v4.0.0 — full removal; persona/scene are Chatter plugins, `Chatter.from_yaml` parses them; UI v0.17.1 remaps legacy agent_types; catalog example configs migrated)


### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.
