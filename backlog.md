
# Backlog - Agents7

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.


### Scan — 3rd May 2026

#### Test Coverage

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
