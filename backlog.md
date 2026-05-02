
# Backlog - Agents7

### Pending Manual Testing

- [ ] End-to-end manual test of the full Agent Builder UI (all 5 steps: Define, Configure, Test, Evaluate, Export).
- [ ] Test prompt versioning and A/B comparison UI: save/load/delete versions, run A/B comparison, verify side-by-side results. Confirm Single Input, Batch Test, and History tabs still work.
- [ ] Test JSONL file upload in Step 3 (Batch Test) and Step 4 (Evaluate).
- [ ] Test auto-discovery flow: upload JSONL, write intent, click Discover Schema, verify form fills, then Generate Prompt & Examples in Step 2.

### Scan — 2nd May 2026

#### Refactoring

- [ ] `llm/base.py` — `LLMEngine` mixes caching, retry, fallback, cost injection, and sync/async/stream dispatch; extract into separate concerns. **HIGH/L**
- [ ] `agents/` — 5-level inheritance chain (`Chatter → PersonalityChatter → SceneChatter → GuardedChatter → GuardedSceneChatter`) plus `Labeler → Detector → Classifier/Regressor`; flatten with composition. **HIGH/M**
- [ ] `agents/guarded.py` + `agents/guarded_scene.py` — Two separate guarded agents duplicate guard mechanics; unify via a parameterised mixin or base. **MED/M**
- [ ] `agents/config_loader.py` — Monolithic loader handles all nine agent types in one file; split per agent type or use a factory registry. **MED/M**
- [ ] `agents/classifier.py:45` — Uses `super(Detector, self).__post_init__()` to skip Detector validation; fragile MRO workaround; redesign with composition. **MED/M**
- [ ] `agents/scene.py:72` — `SceneChatter.__post_init__` calls `Chatter.__post_init__` directly to skip `PersonalityChatter`; fragile super-chain, same root cause as above. **MED/M**
- [ ] `export.py:77` / `metrics.py:300` — `_extract_fields` + `_format_value` logic duplicated across both modules; extract to shared helper. **LOW/S**
- [ ] Root `src/agents7/` — 11 top-level modules mixing unrelated concerns (testing utils, benchmarking, suggestions, export, metrics); group into subpackages. **MED/M**

#### Test coverage

- [ ] `agents/config_loader.py:61` — `_resolve_inheritance()` circular-reference detection is untested. **HIGH/S**
- [ ] `agents/context_guard.py:211` — `ContextGuard.__post_init__` with empty `persona_summary` is untested. **HIGH/S**
- [ ] `agents/parsing.py:47` — `ResponseParser.parse()` enum-name-to-value fallback path untested. **HIGH/M**
- [ ] `export.py:145,252` — `export_eval_report()` and `export_benchmark_report()` missing tests for timing sections, missing fields, and `None` best model. **HIGH/M**
- [ ] `model_selection.py:338` — `_benchmark_model()` engine-init failure recovery untested. **HIGH/M**
- [ ] `agents/memory.py:198` — `SqliteMemory` concurrent read/write races untested; `_lock` contention behaviour unverified. **HIGH/M**
- [ ] `ui/ollama_utils.py:56,80` — `check_connection()` timeout and `list_available_models()` error handling both untested. **HIGH/M**
- [ ] `agents/config_loader.py` — No test file at all; `_deep_merge()`, `load_yaml_settings()`, `build_engine()` kwargs all uncovered. **MED/M**
- [ ] `agents/utils.py` — `run_with_semaphore()`, `resolve_strenum()`, `model_class_to_string()` have no tests. **MED/S**
- [ ] `agents/parsing.py:81` — `ResponseParser.run_validators()` path where a validator raises an exception is untested. **MED/M**
- [ ] `agents/memory.py:124` — `JsonFileMemory.load()` with corrupted JSON untested. **MED/M**
- [ ] `model_selection.py:204` — `find_best_model()` with `runs=0` or empty model list untested. **MED/M**
- [ ] `llm/streaming.py:56` — `_extract()` with missing chunk attributes untested. **MED/M**
- [ ] `agents/jitter.py:49` — `_inject_typo()` word-length boundary (exactly 4 chars) untested. **MED/S**
- [ ] `tests/test_labeler.py:261` — `assert result is not None` is a weak assertion; verify actual output model contents. **MED/S**

### Bugs — 2nd May 2026

- [ ] Benchmark bad_signal 'la capital' too broad — fires on in-character redirect phrases (e.g. "con la capital de Australia ya nos vamos a perder el hilo"). Narrow the signal or add an exclusion for redirect context.
- [ ] qwen3:4b case 5 injection guard miss — guard accepted a response containing 'pirata' in a humorous aside (rej=0). The persona-swap injection was not caught. Investigate guard prompt or threshold for this model.
- [ ] qwen3:4b case 8 char_break retry pattern — in earlier runs the model said 'soy un modelo' after guard rejection, exhausting retries. Tighten the fix instruction or increase max_guard_retries for weaker models.

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.
