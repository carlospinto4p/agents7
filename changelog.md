
## Changelog - Agents7


### v4.11.0 - 18th July 2026

- Added `src/agents7/eval/suite.py` — the composed evaluation route:
  - `run_eval_suite()`: loads and normalises a golden set, builds the
    agent, scores every row, aggregates metrics, summarises the calls
    made, and optionally writes a Markdown report — in one call.
  - `EvalRun`: the run's full output (agent, model, fields, results,
    metrics, call-log summary, report path) with a `format()` summary.
  - `resolve_eval_fields()`: normalises rows and resolves the fields to
    score, honouring an explicit override.
- Updated `src/agents7/eval/export.py`: `export_eval_report()` accepts a
  `call_log` summary and renders a "Call Log" section.
- Updated `src/agents7/cli/__init__.py`:
  - `agents7 eval` is now a thin wrapper over `run_eval_suite()`.
  - Added `--call-log`: log every call made during the run and
    summarise it in the output and the report.
- Fixed `src/agents7/agents/batch.py`: batch runs never emitted call-log
  records, so `call_logger` silently covered only `act()` /
  `act_async()`. `BatchProcessor` now reports each chunk request via an
  `on_call` hook — one record per LLM call, not per input row.
- Updated `README.md`: added an "Evaluation Workflow" section
  documenting the route end to end, and linked the CLI and call-logging
  sections to it.


### v4.10.2 - 18th July 2026

- Filled in `roadmap.md`, previously an empty scaffold stub:
  - Current phase "measurement first": compose the `agents7.eval` /
    CLI / call-log pieces into one workflow, then adopt it across the
    catalog, then resume feature work.
  - Standing invariants: agents are frozen configs not autonomous
    loops, rule-based pipelines before LLMs, provider-agnostic by
    default, measure before claiming.
  - Refuted directions: LLM classifiers where the signal space is
    enumerable, and thinking left enabled for structured output.


### v4.10.1 - 13th July 2026

- Rotated changelog: archived 8 entries to `changelog/2026.md`.


### v4.10.0 - 11th July 2026

- Added A/B testing harness in `agents7.eval`:
  - `run_ab_test()`: run two agents on the same test cases and return comparative metrics.
  - `run_ab_test_async()`: async variant.
  - `ABTestResult`: dataclass with per-agent metrics, accuracy/per-field/timing deltas, pass-fail agreement counts, and the winner.
- Updated `README.md`: documented the A/B testing API under Evaluation Metrics.


### v4.9.1 - 11th July 2026

- Fixed `llm/ollama.py`:
  - `OllamaProvider._think_kwargs()` now always sends the `think` flag (both `True` and `False`) instead of omitting it when `False`. An explicit `think=False` now genuinely disables Ollama's default thinking on thinking-capable models (qwen3, gemma3n), which previously kept thinking and could overrun structured-output calls and drop fields. Non-thinking models accept `think=False` without error.
- Added regression tests in `tests/test_tool_provider.py`:
  - `test_think_kwargs_always_passes_flag()`
  - `test_build_call_kwargs_includes_think_false()`


### v4.9.0 - 11th July 2026

- Added step-event streaming to `ToolAgent` in `agents/tool_agent.py`:
  - `act_stream()` and `act_stream_async()`: run the tool loop, yielding a `ToolStepEvent` per tool call, tool result, and final answer; expose the parsed `AgentResponse` via `agent_response` after iteration.
  - `ToolStepEvent`: frozen event carrying `kind`, `iteration`, and per-kind fields (`name`, `arguments`, `result`, `text`, `thinking`).
  - `ToolStepKind`: `StrEnum` of `TOOL_CALL`, `TOOL_RESULT`, `ANSWER`.
  - `ToolStreamResult` / `ToolAsyncStreamResult`: sync/async iterables of events.
- Exported the new symbols from `agents7.agents`.
- Updated `README.md`: documented streaming the tool loop.
- Added unit tests in `tests/test_tool_agent.py` for the streaming path.


### v4.8.2 - 11th July 2026

- Updated `integration_tests/`:
  - `test_tool_agent.py`: switched the end-to-end ToolAgent test to `qwen2.5:3b` (fast, tool-capable, no thinking) and added a `skipif` guard so it skips when the model is absent. Validates the request → tool_calls → tool result → final answer round-trip against a live Ollama model — no wire-format drift found.
  - `integration_helpers.py`: added `is_model_available()`, shared across the model-gated integration tests.
  - `test_multi_model.py`: import `is_model_available` from `integration_helpers` instead of a local copy.


### v4.8.1 - 11th July 2026

- Updated `.env.example`: documented every environment variable the framework reads — `OLLAMA_MODEL_NAME`, the new `OPENAI_COMPAT_*` trio, `OPENAI_API_KEY` / `OPENAI_BASE_URL`, `ANTHROPIC_API_KEY`, and `BATCH_MAX_CONCURRENT` (previously only `GITLAB_TOKEN`).


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
