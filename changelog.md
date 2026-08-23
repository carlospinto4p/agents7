
## Changelog - Agents7


### v4.15.1 - 23rd August 2026

- Updated `agents7.models.demo`: `Temperature` and `Sentiment`
  docstrings now say what they demonstrate (`Regressor`'s
  numeric-value-plus-reason shape, `Classifier`'s
  category-plus-reason shape), matching `PersonInfo` and
  `PersonBio`, which already did.


### v4.15.0 - 23rd August 2026

- Added `agents7.tools.demo`:
  - `find_product()`, `check_stock()`, `order_total()`: the demo
    store's tools, moved from the `store_assistant` catalog
    package so a `tool`-type sample can resolve `tools` to a
    dotted path the framework ships, the same way a
    structured-output sample resolves `output_model` to
    `agents7.models.demo`.
  - `Product`, `ProductSearch`, `StockLevel`, `OrderTotal`: the
    tools' structured return models.
- Added `tests/test_tools_demo.py`: 18 tests covering the moved
  tools' business logic (catalog matching, stock lookup, order
  costing).


### v4.14.10 - 23rd August 2026

- `.gitlab-ci.yml`: publish job now uploads to `${CI_PROJECT_ID}`
  instead of the hardcoded shared `cpinto` registry project
  (42232103) — own-registry publish direction, programme
  backlog 2026-08-22.


### v4.14.9 - 16th August 2026

- Rotated changelog: archived 2 entries to `changelog/2026.md`.


### v4.14.8 - 15th August 2026

- Fixed `.pre-commit-config.yaml`: `ruff-fix`/`ruff-format` entries now
  use `uv run ruff ...` (was bare `ruff`, which fails with `Executable
  'ruff' not found` under `language: system`) and dropped the trailing
  `.` from `ruff format .` (was formatting the whole tree instead of
  the staged files). This file is inert — `agents7_framework` shares
  its `.git` with the `agents7` monorepo root, so only the root's own
  `.pre-commit-config.yaml` actually runs — fixed for hygiene/accuracy
  while sweeping other repositories for the same defect.


### v4.14.7 - 28th July 2026

- Rotated changelog: archived 3 entries to `changelog/2026.md`.


### v4.14.6 - 26th July 2026

- Redeployed the hardened `scripts/changelog-add.sh`: guards against
  the CWD-relative wrong-repo footgun (comparing the target
  changelog's version against the wrong repo's manifest when invoked
  from outside its own directory).


### v4.14.5 - 25th July 2026

- `.claude/rules/committing.md`: clarified step 6 — a non-code patch
  bump's `uv.lock` self-referential version drift is expected and
  harmless, not something to chase. Promoted from another project's locally
  accepted wording (`RULE_SKIP_PROJECTS["committing"]`), closing a gap
  where the canonical text lacked that framing and repeatedly got
  rediscovered as apparent drift during fleet-wide sweeps.


### v4.14.4 - 25th July 2026

- Rotated changelog: archived 2 entries to `changelog/2026.md`.


### v4.14.3 - 24th July 2026

- `CLAUDE.md`: replaced the superseded **Core Rules** periodic-review
  trigger with the backlog-depth rule — when `/backlog` shows fewer
  than 5 open items, propose `/scan`, `/improvements`, and `/prune`.
  The old "Every 6-7 versions" cadence also named `/refactor` and
  `/optimize`, neither of which is a real skill. Pushed from the
  programme registry via `sync-config --types section`.


### v4.14.2 - 19th July 2026

- Rotated changelog: archived 10 entries to `changelog/2026.md`.


### v4.14.1 - 18th July 2026

- Fixed `src/agents7/agents/pipeline.py`:
  - `Pipeline.from_yaml()`: resolve relative step `agent:` paths
    against the pipeline config's own directory instead of the
    process working directory, so a pipeline loads the same way
    from any CWD. Absolute paths are unchanged.
- Updated `examples/configs/demo_pipeline.yml`: step paths are now
  written relative to the config file.


### v4.14.0 - 18th July 2026

- Added `src/agents7/eval/trajectory.py`: trajectory scoring for
  tool-calling agents, which answer in free text and so cannot be
  scored label-against-label:
  - `TrajectoryMode`: `in_order` (extra calls tolerated, counted and
    reported) or `exact` (they are not).
  - `ExpectedCall` / `ToolCall`: one expected and one actual call.
    Arguments match on the stated keys only, so a row pins what
    matters instead of every default.
  - `TrajectoryResult`: the matched outcome, exposing `missing` and
    `extra`.
  - `parse_expected_calls()`: parse a golden set's `tools` cell, from
    dicts, bare tool names, or a JSON string (a CSV cell).
  - `match_trajectory()`: match an actual trajectory against the
    expected one, in order.
  - `collect_trajectory()`: run an agent once, recording its calls.
  - `run_trajectory_evaluation()`: score every row of a golden set.
  - `compute_trajectory_metrics()`: call-level totals and per-tool
    recall.
- Updated `src/agents7/eval/`:
  - `data.py`: added `tool` to `AGENT_TYPE_MAP`, making tool agents
    evaluable.
  - `suite.py`: `run_eval_suite()` routes a `ToolAgent` through the
    trajectory scorer, added the `trajectory_mode` parameter (falling
    back to the config's `trajectory_mode`), and omitted the confusion
    matrix for trajectory runs.
  - `metrics.py`: `EvalResult` carries an optional `trajectory`.
  - `export.py`: reports gained a Trajectory section with per-tool
    recall, and per-case detail naming the missing and extra calls.
- Updated `src/agents7/agents/`:
  - `base.py`: moved `with_overrides()` up from `Labeler`, so every
    agent type can be cloned with fields replaced.
  - `tool_agent.py`: added the `call_logger` field, emitting a
    `CallRecord` per model turn through a single choke point shared by
    the blocking, async, and streaming loops.
- Updated `src/agents7/cli/__init__.py`: added `--trajectory-mode` to
  `agents7 eval`.
- Added `tests/test_eval_trajectory.py`: 37 tests.
- Updated `tests/test_eval_suite.py`: 7 tests for the tool route.


### v4.13.0 - 18th July 2026

- Updated `tools/`:
  - `Tool.validate_arguments()`: validates and coerces a model's tool
    arguments through the same Pydantic model the tool's JSON schema
    is derived from. Stringified numbers are coerced; anything
    uncoercible raises `ToolArgumentError` with a readable message
    instead of surfacing an opaque Python error.
  - `Tool.run()`: validates arguments before invoking the callable.
  - `Tool.args_model`: the derived args model, or `None` when the
    tool was given a hand-written schema for an untyped callable.
  - `serialize_tool_result()`: recurses into `dict` / `list` /
    `tuple`, so a bare `list[Product]` of Pydantic models encodes as
    a JSON array of objects rather than of Python `repr` strings.
  - `args_model_from_callable()`: exposed as the shared source of
    both the schema and the validator.
- Added `ToolArgumentError` in `exceptions.py`.


### v4.12.3 - 18th July 2026

- Fixed `eval/suite.py`:
  - `run_eval_suite()` now summarises only the calls *that run*
    made. `JsonCallLogger` appends, so reusing a `call_log` path
    across runs made the summary report every earlier run's calls
    as if they were the current one's.
  - `_summarize_calls()` takes the run's in-memory records instead
    of re-reading the log file. The file keeps its full append
    history; only the summary is scoped to the run.


### v4.12.2 - 18th July 2026

- Updated `CLAUDE.md`: corrected the catalog workflow, which pointed at
  an `agents_catalog/pyproject.toml` and `agents_catalog/changelog.md`
  that do not exist:
  - Stated that the catalog root is a container with no package and no
    venv; each agent is its own versioned package.
  - Fixed the test command to run through an agent's project, since the
    catalog root has no venv to run from.
  - Noted that catalog-root files (`scripts/`, `tests/`, `README.md`)
    carry no version bump, and how to restore an agent's editable
    install after `uv sync` drops it.


### v4.12.1 - 18th July 2026

- Updated `roadmap.md`: marked phase 2 (adopt the composed eval route
  across the catalog) complete and restated its gate as a standing
  invariant — no accuracy or latency claim without an eval run behind
  it. Next up is phase 3, the catalog `ToolAgent` showcase.
- Updated `backlog.md`: pruned the resolved eval-adoption item.


### v4.12.0 - 18th July 2026

- Updated `agents7/eval/suite.py`:
  - `run_eval_suite()`: added the optional `notes` parameter, appending
    caller-supplied note lines to the report after the standard eval-set
    and accuracy notes. Lets an agent carry domain caveats into its
    report without dropping down to `export_eval_report()`.


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
