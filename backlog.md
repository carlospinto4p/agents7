
# Backlog - Agents7

### Chatter/Guard/Scene design review — 7th June 2026

- [ ] Two idioms for persona/scene. Since `GuardedChatter` extends `Chatter` directly (composition plugins), `PersonalityChatter`/`SceneChatter` (enforcement via subclass) are a parallel idiom. Consider folding them into a validated factory over `Chatter`. Public-API change — major bump. (LOW, medium)


### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.
