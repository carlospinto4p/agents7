
# Backlog - Agents7

### Improvements — 29th March 2026

- [ ] A/B testing harness — `run_ab_test(agent_a, agent_b, test_cases)` that runs both agents on same inputs and returns comparative metrics.


### agents7_ui — 3rd May 2026

#### Engine / Models

- [x] Backend: `GET /api/engine/models` — query Ollama for loaded/available models and return list — MED — S
- [x] Frontend: Engine status panel — show available Ollama models, which is loaded, memory usage if available — MED — S

#### Evaluation

- [x] Backend: `POST /api/eval/{agent_name}` — run agent against its golden set (if present), return accuracy + per-category breakdown — MED — M
- [ ] Frontend: Evaluation view — trigger eval run, show accuracy bar, per-category table, and list of mismatches — MED — M
