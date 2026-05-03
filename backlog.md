
# Backlog - Agents7

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.


### Scan — 3rd May 2026

#### Refactoring

- [ ] `src/agents7/agents/labeler.py:81` — `__post_init__()` is 49 lines with 7 `object.__setattr__()` mutations; split into `_validate_output_model()`, `_initialize_parser()`, `_initialize_handlers()` — HIGH — M
- [ ] `src/agents7/agents/detector.py:52` — Subclass `__post_init__()` calls `_validate_output_model()` after `super().__post_init__()`; validation is too late if parent assumes validated model — HIGH — M
- [ ] `src/agents7/agents/chatter.py` — 4-level inheritance chain (Chatter → PersonalityChatter → SceneChatter → GuardedSceneChatter); consider composition (Persona, Scene, Guard as plugins) — HIGH — L
- [ ] `src/agents7/eval/metrics.py:114` — `except Exception` blocks log but do not re-raise and do not return an error sentinel; evaluation failures are silently swallowed — HIGH — S
- [ ] `src/agents7/llm/retry.py:120` — `retry_sync()` and `retry_async()` are near-identical; extract shared `_retry_impl()` accepting a sleep callback — MED — M
- [ ] `src/agents7/agents/chatter.py:185` — `object.__setattr__()` for frozen dataclass mutation repeated 6+ times across Chatter and subclasses; extract `FrozenMutable` mixin with `_set_attr()` — MED — M
- [ ] `src/agents7/agents/labeler.py:472` — Input preprocessing (`_preprocess()` + logging) duplicated in all 4 `act*()` methods; extract `_prepare_input()` helper — MED — S
- [ ] `src/agents7/agents/base.py:109` — Provider check is a hard-coded string comparison (`provider != "ollama"`); use a constant or `engine.supports_warmup()` method — MED — S
- [ ] `src/agents7/llm/dispatcher.py:140` — Broad `except Exception` in 4 locations; should catch specific provider/LLM exception types and re-raise unexpected errors — MED — S
- [ ] `src/agents7/agents/labeler.py:549` — `input_data: dict | list[dict]` not validated early; non-dict inputs silently bypass preprocessing — MED — S
- [ ] `src/agents7/agents/detector.py:77` — Heavy `@property` overuse; `_build_prompt()` re-runs on every access to `prompt`/`prompt_for_batch`; cache result in `__post_init__()` — MED — S
- [ ] `src/agents7/agents/detector.py:121` — `_extra_prompt_sections()` duplicated in Detector and Classifier with minor differences; parameterise Detector's version — LOW — S
- [ ] `src/agents7/export.py` — Pure re-export module with no logic; document as deprecated or remove — LOW — S

#### Test Coverage

- [ ] `src/agents7/agents/parsing.py` — No dedicated test file for `ResponseParser.parse()` or `run_validators()`; add tests for JSON malformation, enum fallback, validator chaining — HIGH — M
- [ ] `src/agents7/agents/retry.py` — No dedicated test file for agent retry logic; add tests for zero retries, max_retries=1, jitter, cumulative failures — HIGH — M
- [ ] `src/agents7/llm/dispatcher.py` — `EngineDispatcher` has no dedicated test file; add tests for fallback on primary failure, cache hit/miss with fallback, retry exhaustion — HIGH — M
- [ ] `src/agents7/agents/base.py` — `BaseAgent` base functionality untested directly; add tests for `describe()`, `resolved_system_prompt` with prompt_params edge cases — MED — M
- [ ] `src/agents7/agents/streaming.py` — `AgentStreamResult`/`AgentAsyncStreamResult` tested via Chatter only; add pure Labeler/Detector streaming tests — MED — M
- [ ] `src/agents7/models/base.py` — `Property` and `ClassificationProperty` lack unit tests; add direct tests for `NOT_APPLICABLE`, `CATEGORY_REASONS`, field interaction — MED — M
- [ ] `src/agents7/eval/metrics.py` — `run_evaluation_async()` only covered by integration tests; add unit tests for cancellation, timeout, async errors — MED — M
- [ ] `tests/test_model_selection.py:454` — `_benchmark_model(runs=0)` passes trivially with empty results; should raise a ValueError — MED — S
- [ ] `src/agents7/agents/memory.py` — Edge cases untested: corrupted JSON recovery, concurrent `SqliteMemory` writes, purge with missing `conversation_id` — MED — M
- [ ] `src/agents7/agents/parsing.py:61` — `_fix_malformed_json()` untested directly; add tests for trailing commas, missing quotes, unclosed brackets — MED — S
- [ ] `tagger_agent/tests/test_rules_classifier.py:80` — Weak assertions (enum equality only); add confidence bounds and detail field validation — MED — S
- [ ] `tests/test_chatter.py` — Missing edge cases: empty history, `max_history=0`, `conversation_id` mutation, very long messages — MED — M


### agents7_ui — 3rd May 2026

#### Infrastructure

- [ ] Backend: set up FastAPI router structure (`/api/catalog`, `/api/run`, `/api/engine`) and wire uvicorn dev server — HIGH — S
- [ ] Frontend: add Svelte routing (e.g. `svelte-routing` or SvelteKit migration) with placeholder pages for Catalog, Runner, Engine — HIGH — S
- [ ] Docker Compose: verify full `docker compose up --build` flow, backend + nginx-served frontend, Ollama on host — MED — S

#### Catalog Browser

- [ ] Backend: auto-discover all agents in `agents_catalog/` by scanning for `config.yml` files; return list with name, agent type, model, description — HIGH — M
- [ ] Frontend: Catalog page — list all discovered catalog agents as cards (name, type, model); click to open Agent Detail — HIGH — M
- [ ] Frontend: Agent Detail view — show system prompt, output schema (rendered from YAML config), and YAML source — MED — M

#### Agent Runner

- [ ] Backend: `POST /api/run/{agent_name}` — instantiate catalog agent from YAML config, run `act()` with provided input, return structured output + metrics — HIGH — M
- [ ] Frontend: Single-input runner — JSON input form derived from agent's input fields; submit, see structured output rendered as a table/card — HIGH — M
- [ ] Frontend: Batch runner — paste or upload JSONL, run against selected agent, show results table with pass/fail per row — MED — L
- [ ] Frontend: Response history — store last N run results in a Svelte store, persisted to `localStorage`; show in a collapsible sidebar — MED — M

#### Engine / Models

- [ ] Backend: `GET /api/engine/models` — query Ollama for loaded/available models and return list — MED — S
- [ ] Frontend: Engine status panel — show available Ollama models, which is loaded, memory usage if available — MED — S

#### Evaluation

- [ ] Backend: `POST /api/eval/{agent_name}` — run agent against its golden set (if present), return accuracy + per-category breakdown — MED — M
- [ ] Frontend: Evaluation view — trigger eval run, show accuracy bar, per-category table, and list of mismatches — MED — M
