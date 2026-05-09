
# Backlog - Agents7

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.


### 9th May 2026 — /scan

#### Refactoring

- [x] `agents7_ui/backend/routers/eval.py:73,103` & `catalog.py:40` — Replace broad `except Exception: pass` with specific exceptions — HIGH — S
- [ ] `agents7_framework/integration_tests/test_multi_model.py:31` & `conftest.py:25` — Replace bare `except Exception:` in availability checks with specific exceptions — HIGH — S
- [x] `agents7_ui/backend/routers/eval.py:100-104` — Log exception before swallowing in agent execution error handler — HIGH — M
- [ ] `agents7_framework/src/agents7/agents/labeler.py:81-132` & `detector.py:52-75` — Replace `_set_attr` frozen-bypass pattern with `__init__` or composition — HIGH — M
- [ ] `agents7_ui/backend/routers/catalog.py:26-54` & `eval.py:65-74` — Extract shared output model/schema parsing into `_load_output_schema()` helper — MED — M
- [ ] `agents7_ui/backend/routers/catalog.py:14-23` & `eval.py:53-54` — Extract duplicate YAML config loading into a shared helper — MED — M
- [ ] `agents7_framework/src/agents7/agents/detector.py:78`, `classifier.py:60`, `regressor.py:35` — Extract identical `from_yaml` logic to base class or mixin — MED — M
- [ ] `agents7_framework/src/agents7/agents/config_loader.py` & `chatter_config_loader.py` — Consolidate into single config loader module — MED — L

#### Test coverage

- [ ] `agents7_framework/src/agents7/agents/guarded.py:219,247` — Add tests for `chat_stream()` and `chat_stream_async()` including `_finalize` callbacks — HIGH — M
- [ ] `agents7_framework/src/agents7/agents/classifier.py`, `detector.py`, `regressor.py`, `context_guard.py` — Add unit tests for `from_yaml()` in all four agents — MED — S
- [ ] `agents7_framework/tests/test_model_selection.py:78-108` — Add tests for `_compare_output()` with `None` fields and partial field matching — MED — S
- [ ] `agents7_framework/src/agents7/llm/dispatcher.py:259` — Add dedicated test for `asyncio.TimeoutError` path in `execute_async()` — MED — S
- [ ] `agents7_framework/src/agents7/agents/batch.py:194` — Add tests for malformed chunk responses in `_flatten()` — MED — M
- [ ] `agents7_framework/src/agents7/agents/guarded_scene.py:35` — Test `from_yaml()` persona_summary derivation with scene context — MED — M
