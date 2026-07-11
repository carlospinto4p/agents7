
## Changelog - Agents7


### v4.8.0 - 11th July 2026

- Added env-var configuration for the OpenAI-compatible provider (external providers like DeepSeek):
  - `settings.py`: added `OpenAICompatSettings` reading `OPENAI_COMPAT_MODEL_NAME`, `OPENAI_COMPAT_BASE_URL`, and `OPENAI_COMPAT_API_KEY` from the environment at instantiation.
  - `OpenAICompatProvider`: `model_name`, `base_url`, and `api_key` now fall back to `OpenAICompatSettings` when omitted; raises `ValueError` when no model is configured.
  - `LLMEngine`: `model_name` is now optional and resolves from `OPENAI_COMPAT_MODEL_NAME` for the `openai_compat` provider.
- Updated `README.md`: documented the `OPENAI_COMPAT_*` env vars and a DeepSeek end-to-end setup.


### v4.7.4 - 1st July 2026

- Rotated changelog: archived 1 entries to `changelog/2026.md`.


### v4.7.3 - 28th June 2026

- Rotated changelog: archived 1 entries to `changelog/2026.md`.


### v4.7.2 - 25th June 2026

- Rotated changelog: archived 1 entries to `changelog/2026.md`.


### v4.7.1 - 25th June 2026

- Rotated changelog: archived 16 entries to `changelog/2026.md`.


### v4.7.0 - 21st June 2026

- Updated `tools/`:
  - `Tool.run()`: serializes structured returns (`dict`, `list`,
    `tuple`, or a Pydantic model) to JSON before feeding them back
    to the model; plain strings pass through, other types fall back
    to `str`.
  - Added `serialize_tool_result()`: the standalone serializer
    backing `Tool.run()`, exported from `agents7.tools`.


### v4.6.0 - 21st June 2026

- Wired tool calling for the OpenAI-compatible provider in `src/agents7/llm/openai_compat.py`:
  - `OpenAICompatProvider.supports_tools()`: now returns `True`.
  - `OpenAICompatProvider.build_tool_turn_messages()`: emits the assistant `tool_calls` array plus one `role: "tool"` message per call, linked by `tool_call_id`.
  - `_parse_tool_calls()`: decodes `function.arguments` JSON strings into dicts (malformed strings degrade to `{}`).
  - `_build_params()`: forwards `tools` unchanged (the registry already emits OpenAI's native schema); `chat`/`chat_async` thread `tools` through.
- `ToolAgent` now works on Ollama, Anthropic, and OpenAI-compatible providers.


### v4.5.1 - 21st June 2026

- Added the `anthropic` optional-dependency extra in `pyproject.toml`:
  - Install Claude support with `pip install agents7[anthropic]`.
  - `uv sync --all-extras` now installs `anthropic`, so the
    `AnthropicProvider` tests run instead of skipping.
- Updated `README.md`: Anthropic install uses the `[anthropic]`
  extra instead of a bare `anthropic` install.


### v4.5.0 - 21st June 2026

- Added tool calling to the Anthropic provider in `llm/anthropic.py`:
  - `AnthropicProvider.supports_tools()`: now returns `True`.
  - `AnthropicProvider.build_tool_turn_messages()`: emits the
    `tool_use` / `tool_result` content blocks Claude expects,
    linking each result to its call by `tool_use_id`.
  - `_convert_tools()`: maps neutral OpenAI-style tool schemas to
    Anthropic's `input_schema` format.
  - `chat` / `chat_async` now forward `tools` and parse `tool_use`
    blocks into `tool_calls`, so `ToolAgent` works on Claude models.
- Updated `README.md`: tool calling now lists Ollama and Anthropic.


### v4.4.0 - 21st June 2026

- Added tool / function calling (Ollama provider):
  - `agents7.tools`: `Tool`, `ToolRegistry`, `schema_from_callable`.
  - `ToolAgent`: bounded call -> tool -> call loop with `act()` /
    `act_async()` / `from_yaml()`.
  - Added `tools=` to `LLMEngine.chat` / `chat_async` and the
    `tool_calls` field to the normalized result dict.
  - Added `LLMProvider.supports_tools()` and
    `build_tool_turn_messages()`; wired for Ollama.
  - Added `ToolSchemaError` and `ToolLoopError`.


### v4.3.1 - 21st June 2026

- `.claude/rules/committing.md`: added a **Branching** section — the
  default workflow commits directly to the default branch (`main`); do
  not create a feature branch or worktree unless the user asks or a
  skill requires isolation. Overrides the harness "branch first"
  default per instruction priority.


### v4.3.0 - 21st June 2026

- Added `agents7` CLI console script (`[project.scripts]`) with two subcommands, thin glue over the YAML loaders, `run_evaluation`, and `export_eval_report`:
  - `agents7 run`: execute a YAML-configured agent against an inline-JSON or file input and print the structured output (`--full` for text, thinking, and metrics).
  - `agents7 eval`: score an agent against a labeled golden set, print accuracy, and optionally write a Markdown report (`--report`).
- Added `eval/data.py` (re-exported from `agents7.eval`):
  - `build_agent()`: build a structured-output agent from a YAML config and `agent_type`.
  - `load_eval_rows()`: load a golden set from `.jsonl`, `.json`, or `.csv`.
  - `normalise_rows()`: normalise nested or flat rows to the canonical eval shape.
  - `model_output_fields()`, `resolve_agent_class()`, `load_config_dict()`, `AGENT_TYPE_MAP`.
- Updated `README.md`: documented the CLI and the reusable eval-data helpers.


### v4.2.0 - 14th June 2026

- Added `call_log.py`:
  - `summarize_call_log()`: read a JSONL call log (path or
    iterable of `CallRecord` / dict / JSON line) into aggregate
    statistics.
  - `CallLogSummary`: overall stats plus per-agent / per-model
    breakdowns and error counts; `format()` renders a text
    report.
  - `CallStats`: per-group counts with `success_rate` and
    `avg_duration` properties.


### v4.1.4 - 14th June 2026

- Updated `src/agents7/agents/config_loader.py`:
  - Added `_build_persona()` and `_build_scene()`, wrapping
    `Persona.from_dict` / `Scene.from_dict` so a malformed
    `persona` / `scene` section (missing `full_name`, unknown
    field, non-mapping value) raises `AgentConfigurationError`
    naming the file instead of a raw `TypeError`. Applied across
    all three Chatter-family construction sites.
- Added tests in `tests/test_chatter.py` for the clearer errors
  (empty persona dict, unknown persona field).


### v4.1.3 - 14th June 2026

- Added tests:
  - `tests/test_context_guard.py`: `forbidden_terms_for_language`
    case/whitespace normalization (mixed-case, surrounding
    whitespace).
  - `tests/test_chatter.py`: `Chatter.from_yaml` persona/scene edge
    cases — empty persona dict rejected, persona with `full_name`
    loads, `extra_instructions` stripping, scene-without-persona
    raising.


### v4.1.2 - 14th June 2026

- Refactored `src/agents7/agents/`:
  - `config_loader.py`: extracted `_maybe_add_memory()`, removing the
    duplicated memory-config block from the three Chatter-family
    loaders.
  - `pipeline.py`: extracted `Pipeline._call_mapper_safe()`, removing
    the repeated mapper try/except from `_process_step_result`,
    `_evaluate_batch_step`, `act_stream`, and `act_stream_async`.
  - `parsing.py`: narrowed `ResponseParser.run_validators()` to catch
    only `ValueError` / `ValidationError`; unexpected errors now
    propagate instead of being masked as rejections.


### v4.1.1 - 14th June 2026

- Refactored `config_loader.py`:
  - Extracted `_CHATTER_OPTIONAL_FIELDS` and
    `_GUARDED_OPTIONAL_FIELDS` constants, removing the
    duplicated optional-field maps across the chatter and the
    two guarded loaders.


### v4.1.0 - 14th June 2026

- Updated `JsonCallLogger` (`call_log.py`):
  - Holds a single append file handle (opened lazily on first
    write, flushed per record) instead of reopening the file on
    every call.
  - Added `close()` and context-manager support (`__enter__` /
    `__exit__`) to release the handle.


### v4.0.5 - 14th June 2026

- Removed dead Docker scaffolding:
  - `docker-compose.yml`: orphaned `app` service built from a
    Dockerfile that no longer exists (deleted in v3.19.0) and
    targeting the old Streamlit port 8501. The live app now ships
    from `agents7_ui/`.
  - `.dockerignore`: only used by the removed compose build.


### v4.0.4 - 13th June 2026

- Removed `benchmarks/output_qwen3_8b.txt`: a failed-run capture
  (model `qwen3:8b` was never pulled, so it logged only a 404
  traceback with stale Windows/Dropbox paths and no results). The
  sibling captures hold the real benchmark data.


### v4.0.3 - 13th June 2026

- Rotated changelog: archived 1 entries to `changelog/2026.md`.


### v4.0.2 - 13th June 2026

- Rotated changelog: archived 1 entries to `changelog/2026.md`.


### v4.0.1 - 10th June 2026

- Rotated changelog: archived 9 entries to `changelog/2026.md`.


### v4.0.0 - 7th June 2026

- Removed `PersonalityChatter` and `SceneChatter`. Persona and
  scene are composition plugins on the base `Chatter`, so the
  enforcement-via-subclass idiom was redundant.
  - Use `Chatter(persona=...)` / `Chatter(persona=..., scene=...)`
    or `Chatter.from_yaml(...)` with `persona:` / `scene:`
    sections instead.
  - `Chatter.from_yaml` now parses optional `persona`, `scene`,
    and `extra_instructions` sections and auto-generates the
    system prompt (no manual `system_prompt` needed when a
    persona is present).
  - Deleted `agents/scene.py`; `agents/personality.py` now only
    re-exports `Persona` / `Scene` / `PhysicalTraits` for
    backward-compatible imports.
  - Dropped `PersonalityChatter` / `SceneChatter` from the public
    API (`agents7.agents`).
- Updated docs (`docs/chatter.md`, `docs/design.md`,
  `README.md`, `examples/`) to the persona/scene plugin idiom.


### v3.26.0 - 7th June 2026

- Updated `GuardedChatter` fix fallback: when retries are
  exhausted and the guard supplied a `fix`, the chatter now
  regenerates once using the fix as guidance (in full persona
  voice) and guards the result, instead of returning the
  guard's one-line fix verbatim. Falls back to the verbatim fix
  if the regeneration is still rejected.
  - Added `_regenerate_from_fix` / `_regenerate_from_fix_async`
    and `_add_fix_guidance`.
  - Adds one chatter call + guard check only on the
    exhausted-retry path, and only when a `fix` exists.
- Updated `docs/chatter.md`: documented the fix-guided
  regeneration step.


### v3.25.0 - 7th June 2026

- Added language-aware forbidden vocabulary to the guard:
  - `ContextGuard.forbidden_terms`: list injected into the
    prompt (via a `$forbidden_terms` placeholder), defaulting
    to the previous bilingual ES/EN list so the default prompt
    is byte-identical.
  - `forbidden_terms_for_language()` in `agents/context_guard.py`:
    starter lists for Spanish, English, French, German,
    Portuguese, Italian; unknown/empty falls back to the
    default.
  - YAML loaders derive `forbidden_terms` from
    `persona.language` unless `guard.forbidden_terms` is set;
    `ContextGuard.from_yaml` accepts the field.
  - `GuardedChatter._add_feedback` now lists the guard's
    `forbidden_terms` instead of a hardcoded Spanish set.
- Updated `docs/chatter.md`: documented language-aware
  forbidden vocabulary.


### v3.24.1 - 7th June 2026

- Updated `Chatter.__post_init__`: a non-empty `scene` with no
  `persona` now raises `AgentConfigurationError` instead of
  silently dropping the scene (the scene frames the persona's
  role-play and is unusable on its own). An empty `scene`
  without a persona is still allowed.
- Updated `docs/chatter.md`: noted the scene-requires-persona
  rule.


### v3.24.0 - 7th June 2026

- Added fail-closed guarding to `GuardedChatter` (and
  `GuardedSceneChatter`):
  - `guard_fail_open` (default `True`): when `False`, an
    unparseable guard verdict no longer passes through —
    the response is replaced with a fallback deflection.
  - `guard_fallback_response`: the in-character deflection
    used when fail-closed and the guard gives no verdict
    (empty uses a generic default).
  - `_is_accepted` now honours `guard_fail_open`; the
    parse-failure `GuardEvent` reports `accepted` truthfully.
  - YAML loaders accept both new fields.
- Updated `docs/chatter.md`: documented fail-open vs
  fail-closed guarding.


### v3.23.0 - 7th June 2026

- Added `guard_buffer_stream` field to `GuardedChatter` (and
  `GuardedSceneChatter`): when `True`, streaming buffers the
  full response, runs the guard, and yields only the guarded
  text, so unvalidated content never reaches the caller.
  Defaults to `False` (live streaming, guard runs after).
  - `GuardedChatterStreamResult` / `GuardedChatterAsyncStreamResult`
    override iteration to drain-then-guard in buffered mode.
  - YAML loaders accept `guard_buffer_stream`.
- Updated docs:
  - `docs/chatter.md`: documented `guard_buffer_stream` and the
    streaming guard trade-off.
  - `README.md`: corrected the agent hierarchy diagram
    (`GuardedChatter` extends `Chatter`, not `PersonalityChatter`).


### v3.22.0 - 7th June 2026

- Added `Scene.to_summary()` in `agents/persona.py`: renders a
  compact scene summary (setting, situation, rules) for the guard
  prompt.
- Updated `GuardedSceneChatter`:
  - Added `__post_init__` that folds the scene context into the
    guard's `persona_summary`, so direct Python construction now
    matches the YAML loader. Idempotent; an explicit
    `persona_summary` overrides the persona description but the
    scene context is still appended.
- Updated `_derive_persona_summary` (`agents/config_loader.py`)
  to reuse `Scene.to_summary()`.
- Updated `docs/chatter.md`: scene-aware guarding now applies to
  both construction paths.


### v3.21.43 - 7th June 2026

- Updated `docs/chatter.md`:
  - Replaced the stale ASCII agent hierarchy with a Mermaid
    `classDiagram` showing inheritance plus persona/scene/guard
    composition.
  - Corrected `GuardedChatter` to extend `Chatter` (not
    `PersonalityChatter`).
  - Added a `GuardedSceneChatter` section, including the
    YAML-only scene enrichment of the guard's `persona_summary`.


### v3.21.42 - 7th June 2026

- Rotated changelog: archived 2 entries to `changelog/2026.md`.
