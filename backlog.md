
# Backlog - Agents7

### Chatter/Guard/Scene design review — 7th June 2026

- [ ] Default guard prompt is hardcoded, Spanish-specific, and tuned to catalog personas. `_DEFAULT_GUARD_PROMPT` and `_add_feedback` bake in Spanish AI-vocabulary lists and domain-specific examples (David the journalist, tavern source-protection), limiting generality for other languages/use-cases. Derive forbidden-vocabulary from `persona.language` and/or move domain examples into config. (MEDIUM, medium)
- [ ] Guard `fix` used verbatim from a one-line summary. `_use_fix_or_text` returns the guard's `fix` as the final turn, but the guard only sees `persona_summary`, not the full persona, so the fallback can be off-character. Treat `fix` as guidance for one more chatter regeneration instead of the final word. (LOW, low-medium)
- [ ] Two idioms for persona/scene. Since `GuardedChatter` extends `Chatter` directly (composition plugins), `PersonalityChatter`/`SceneChatter` (enforcement via subclass) are a parallel idiom. Consider folding them into a validated factory over `Chatter`. Public-API change — major bump. (LOW, medium)


### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.
