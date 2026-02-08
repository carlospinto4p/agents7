
# Backlog - Agents7

Pending manual testing:

- End-to-end manual test of the full Agent Builder UI (all 5 steps: Define, Configure, Test, Evaluate, Export).
- Test prompt versioning and A/B comparison UI: save/load/delete versions, run A/B comparison, verify side-by-side results. Confirm Single Input, Batch Test, and History tabs still work.

Low priority:

- Add more LLM providers (e.g., OpenAI, Anthropic) beyond Ollama.
- Add persistent disk-based cache option (SQLite) alongside the in-memory LRU.
- Publish Docker image to a container registry for easier deployment.
