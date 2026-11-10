
## Changelog - Agents7


### v9.25.22 - 10th November 2026

- Rotated changelog: archived 16 entries to `changelog/2026.md`.


### v9.25.21 - 8th November 2026

- Updated `.claude/rules/00-rule-index.md`: regenerated, it now lists
  `measurement-needs-history` and `metrics-completeness`.


### v9.25.20 - 8th November 2026

- Added `.gitattributes`: text files are stored with Unix line endings.
  Four `.claude/rules/` files and
  `.pre-commit-scripts/check_unstaged_claude.py` were converted from
  Windows line endings; their text is unchanged.


### v9.25.19 - 8th November 2026

- `.claude/rules/measurement-needs-history.md`: shortened from 326 to
  about 80 lines. The nine rules and the functions and constants they
  name are kept; the long incident accounts are gone (they stay in git
  history).


### v9.25.18 - 8th November 2026

- `README.md`: the opening paragraph no longer says every agent answers
  with a Pydantic model. `Chatter` and `ToolAgent` reply in text.


### v9.25.17 - 7th November 2026

- Removed `.gitlab-ci.yml`: it uploaded a wheel to GitLab's package
  registry, and releases are now git tags on GitHub.
- `.env.example`: removed `GITLAB_TOKEN`, which only that upload used.


### v9.25.16 - 7th November 2026

- `docs/`: removed the em dashes from the prose of the pages moved out
  of the README in v9.25.15, by rewording each sentence (179 in all).
  Code blocks and headings are unchanged; dashes inside code comments
  stay.


### v9.25.15 - 7th November 2026

- `README.md` is a short overview (86 lines, was 3,377): what agents7 is,
  the agent types, installing, one example, and links. Its reference
  sections moved unchanged to new pages:
  - `docs/agents.md`: building agents.
  - `docs/providers.md`: LLM providers.
  - `docs/operations.md`: cost, caching and logging.
  - `docs/pipelines.md` and `docs/tools.md`.
  - `docs/evaluation.md`.
  - `docs/chat-agents.md`.
  - `docs/architecture.md` and `docs/development.md`.
- `CLAUDE.md` and `.claude/rules/committing.md`: user-facing changes are
  documented on the `docs/` page; the README stays an overview (owner,
  7th November 2026).


### v9.25.14 - 7th November 2026

- Added `docs/agent-packages.md`: how an agent in its own repository
  depends on an agents7 tag and shows up in agents7-ui. The example agent
  is `github.com/carlospinto4p/agent-sentiment_classifier`.
- Removed `docs/repository-split.md` and `docs/superpowers/`, internal
  planning notes; `README.md` and `CLAUDE.md` point at the new page.


### v9.25.13 - 7th November 2026

- Removed `samples/sentiment_classifier/`. It is now its own package at
  `github.com/carlospinto4p/sentiment_classifier` (step 3 of
  `docs/repository-split.md`), with its own output model and an
  `agents7.agents` entry point. `agents7.models.demo.Sentiment` stays:
  `examples/configs/demo_sentiment_classifier.yml` and the tests use it.
- `docs/repository-split.md`: status updated to steps 1 to 3 done.


### v9.25.12 - 7th November 2026

- `.claude/rules/testing.md`: the SQLite fixture line states the
  session-scoped template pattern itself instead of pointing at a rule
  that is not in this repository.


### v9.25.11 - 7th November 2026

- `CLAUDE.md`: restored what v9.25.10 had dropped or changed from the
  old repository's file: the Shell Commands, Project Configuration,
  Versioning / Release and Testing sections, the "describe changes"
  and `uv pip install -e .` steps, running the integration suite on
  every change, and the auto-fix lint command. Removed a release-tag
  step v9.25.10 had added on its own.
- `.claude/rules/optimization-areas.md`: restored the "sequential LLM
  calls that could be batched" signal, lost with a broken fragment.


### v9.25.10 - 7th November 2026

- Added the project's Claude Code setup, carried over from the old
  combined repository and cleaned for a public repository:
  - `CLAUDE.md`: written for the framework alone.
  - `.claude/rules/`: committing, versioning, testing, the review-area
    notes, `metrics-completeness.md` and `measurement-needs-history.md`,
    with paths at the repository root and the UI-only Svelte notes
    left for the UI's repository.
  - `.claude/skills/validate-guarded-scene/` and its `.opencode/`
    command.
  - `.claude/settings.json`: tool permissions.
- Added the pre-commit setup: `.pre-commit-config.yaml` (ruff pinned to
  v0.15.9, changelog and version guards) and `.pre-commit-scripts/`.
- Added `scripts/changelog-add.sh` and `scripts/pre-commit.sh`.
- `.gitignore`: per-machine Claude Code files.


### v9.25.9 - 7th November 2026

- The framework is now its own repository, cut from the monorepo with
  its history kept: the package sits at the repository root and
  `samples/` beside it.
- Added `LICENSE` (MIT), `ruff.toml`, and a `.gitignore` for the new
  layout.
- `tests/test_sample_golden_sets.py`: finds `samples/` at the
  repository root.
- `README.md`: installs a tagged release instead of a subfolder.
- The sample READMEs point at the repository root instead of
  `agents7_framework/`.


### v9.25.8 - 7th November 2026

- `docs/repository-split.md`: the owner agreed the design and answered
  its four open questions; they are now decisions, followed by the
  order of work.


### v9.25.7 - 7th November 2026

- Added `docs/repository-split.md`: the proposed design for splitting
  the framework, the UI and each agent into separate repositories, with
  the owner's decisions so far and the open questions.


### v9.25.6 - 7th November 2026

- Rotated changelog: archived 5 entries to `changelog/2026.md`.


### v9.25.5 - 7th November 2026

- `README.md`: the core dependencies no longer list `cpinto`, removed
  from `pyproject.toml` long ago.
- `changelog/2026.md`: four entries reworded.


### v9.25.4 - 7th November 2026

- Prepared the repository for its public copy on GitHub:
  - `agents/context_guard.py`: the guard prompt's example of joking
    back at a bot accusation now says "Soy un bot con gato y todo"; its
    test checks for it.
  - `pyproject.toml`: the author's email is `carlospinto4p@gmail.com`.
  - `README.md`: installation is from the GitHub repository, since the
    package is not on PyPI.
  - Added a root `README.md` and an MIT `LICENSE`; the samples link to
    the framework folder instead of the GitLab project.


### v9.25.3 - 7th November 2026

- Reworded examples, docs and tests ahead of publishing this
  repository:
  - `README.md`, `docs/chatter.md`, `docs/design.md`, `roadmap.md`,
    `examples/`: the character examples use the sample persona David
    García López.
  - `src/agents7/`: docstrings and comments only; no code changed.
  - `tests/`: fixtures and assertions use David and neutral made-up
    names; the same behaviour is tested.
  - Deleted seven recorded benchmark outputs (`output_guard_*.txt`).
  - `changelog.md`, `changelog/2026.md`, `.claude/rules/`: the same
    rewording; versions, dates and order are unchanged.


### v9.25.2 - 7th November 2026

- The agent catalog moved out of this repository, with its history.
  The generic samples moved from the catalog's `examples/` to
  `samples/` at the repository root:
  - Moved `tests/test_example_golden_sets.py` to
    `tests/test_sample_golden_sets.py`, now walking `samples/`.
  - Removed the catalog-only refinement rule, skill and command,
    which moved with the catalog.
  - `CLAUDE.md`: the Catalog section became a Samples section that
    points at the new repository.
  - `chat-agent-validation.md`, `measurement-needs-history.md`:
    dropped the catalog paths from their triggers.
- Fixed `.pre-commit-scripts/check_changelog_headers.sh`: it read a
  changelog deleted whole as destroyed headers, which blocked the
  move above. It now leaves deleted files out (`--diff-filter=d`).


### v9.25.1 - 7th October 2026

- Rotated changelog: archived 2 entries to `changelog/2026.md`.


### v9.25.0 - 4th October 2026

- Updated `eval.chat_review.read_payload`: each row of a
  conversation's `context` names its `group` (`run`, `context`,
  `memory` or `variants`), and the `Knew already` row carries its
  facts one by one as `items`. `label` and `value` are unchanged, so
  the generated page renders as before.


### v9.24.1 - 4th October 2026

- Rotated changelog: archived 7 entries to `changelog/2026.md`.


### v9.24.0 - 2nd October 2026

- Added `agents.context.spanish_weekday()`: the day of the week in
  Spanish from the local clock, for a `context_params` `callable`.
  Exported from `agents7.agents`.


### v9.23.1 - 2nd October 2026

- Fixed values supplied through `set_context` not reaching a guarded
  conversation. Found when a read ran every conversation on the
  default nicknames:
  - `BaseAgent.with_overrides`: the copy keeps the supplied context.
    A bare `dataclasses.replace` still drops it.
  - `eval.chat_review.ConversationRecorder`: copies the agent with
    `with_overrides`, so the agent that talks is the one the caller
    configured.
  - `GuardedSceneChatter`: the scene was resolved into the guard when
    the agent was built, so the guard judged every reply against the
    defaults. `guard` now keeps `context_params` and
    `prompt_variants` names as placeholders, and
    `GuardedChatter.current_guard()` fills them per reply.
- A guarded style-history key for an agent whose scene names a context
  param moves once: the guard's identity no longer carries the time
  of day the agent was built at.
- Added 9 tests in `tests/test_context_params.py`,
  `tests/test_chat_review.py` and `tests/test_guarded_scene.py`.
- Updated `README.md`: copying an agent that has context, and
  `current_guard()`.


### v9.23.0 - 2nd October 2026

- Added `when_prompt_matches` to `eval.style.forbidden_pattern`: the
  rule applies only to replies whose prompt matches the given
  regular expression, and is not applicable elsewhere.


### v9.22.0 - 2nd October 2026

- Added script context in `eval.style`:
  - `ConversationScript.context`: values handed to `set_context`
    before the script's first turn, in both runners.
  - `scripts[].context` in YAML, read by `load_style_contract`.
  - `style_history`: a script's context is part of the run's
    fingerprint and configuration key; a script without one hashes
    as before.
- Added 7 tests in `tests/test_style_contract.py` and
  `tests/test_style_history.py`.


### v9.21.1 - 2nd October 2026

- Fixed the integration suite timing out on a cold Ollama:
  - `integration_helpers.warm_model()`: loads a model once per
    session; every fixture that builds an agent calls it.
  - `conftest.py`: the 60s per-test timeout covers the test body
    only, so a model load in setup is not counted.
  - `conftest.py`: the 300s session limit no longer counts time
    spent loading models.


### v9.21.0 - 2nd October 2026

- Added derived context params in `agents.context`:
  - `ContextParam.derived_from`: names another declared param; the
    resolver is then called with that param's current value.
  - `context_params.<name>.from` in YAML, read by
    `parse_context_params`.
  - `BaseAgent.resolved_context` computes a derived param after its
    source, so supplying the source is enough.
- Added `tests/test_context_params.py`: 13 tests for derived params.
- Updated `README.md`: derived values under Per-Conversation Context.
