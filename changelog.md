
## Changelog - Agents7


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


### v9.20.2 - 22nd September 2026

- Rotated changelog: archived 14 entries to `changelog/2026.md`.


### v9.20.1 - 20th September 2026

- Updated `agents.facts` `_EXTRACTOR_PROMPT`: a fact now keeps the
  character's own tense and the words that place it in time. A
  character who said "la de mi piso en Bilbao, hace años" was
  recorded as "Tiene un piso en Bilbao" — the "hace años"
  dropped and a flat it had lived in became one it has. A fact
  store is replayed into every later prompt, so a wrong tense
  compounds.


### v9.20.0 - 20th September 2026

- Updated `eval.chat_review.ConversationRecorder`: when
  `agent_version` is not given it now reads the installed
  distribution the agent's `name` resolves to, and raises
  `AgentConfigurationError` when none answers to that name.
  It previously fell back to the agent's own `version`, which
  for a config-loaded agent is the config *schema* version, so
  a recording made without the argument was filed under `1`.
  A recording's version is what tells two reads apart, so it
  is never guessed.


### v9.19.0 - 20th September 2026

- Added `eval.chat_review.read_payload()`: a whole read as plain
  data — its text, verdicts, measures, comparison rows and
  conversations, plus the note kinds and stamps a renderer needs to
  label them. JSON-serialisable throughout.
  - `render_review` writes a self-contained page and is still the
    answer when a read is something to publish. A UI that lets
    somebody *click a turn* cannot use that page: it has to build
    the transcript itself, and the moment it does, the template
    exists twice and the two drift. Four reads came out with four
    designs before the generator existed.
  - So the data is the contract. `render_review` builds its
    conversations from the same function this does, and a test
    asserts both show the same ones. What is deliberately not
    shared is the visual template, which is each renderer's own
    business.
- Added tests in `tests/test_chat_review.py`: four on
  `read_payload`, including that it is JSON-serialisable and that it
  agrees with the generated page.


### v9.18.0 - 20th September 2026

- Added `eval/chat_feedback.py`, the standardised format for the
  operator's corrections to a chat agent. A style contract says a
  rule stopped holding and a read says whether the agent is good
  company; neither produces the correction of a line and the rule it
  becomes, which is what actually improves the agent. That has been
  written by hand in two shapes — a rewrite and a rule — and both
  said of themselves that they were not machine-readable.
  - `FeedbackEntry`: one correction, carrying a rewrite, a rule, or
    both. `turn` is optional, so a note about a whole conversation
    is expressible; a verdict of `right` marks a line worth
    imitating rather than the absence of criticism; `he_said` sits
    beside the rewrite so it cannot drift from what it corrected;
    `shipped` is empty until the rule lands.
  - `Feedback.rule_changes()`: the rules grouped by the config file
    they land in, unassigned last, with the unshipped separable.
  - `Feedback.eval_rows()`: corrected turns as machine-readable
    targets, and lines marked right as targets in their own right.
  - `render_feedback()` and `write_feedback_report()`: the
    operator's report, grouped by rules to change, corrected lines,
    and what is worth imitating.
  - `merge_entries()`: replaces every entry on a touched turn and
    leaves other turns alone, so an editor can write a file back
    without losing what was there.
  - `load_feedback`, `save_feedback`, `write_eval_rows`.
- Added the `chat-feedback` CLI command: derives all three outputs
  from one file, prints the report when given neither `--report` nor
  `--rows`, and reports how many rules have not shipped.
- Added tests:
  - `tests/test_chat_feedback.py`: 39 covering the entry's
    validation, grouping, eval rows, round trip, merging and the
    report.
  - `tests/test_cli.py`: 4 on `chat-feedback`.
- Updated `README.md`: a "Feedback on a Chat Agent" section, and the
  three-instruments framing in "Scoring a Chat Agent's Style".


### v9.17.0 - 20th September 2026

- Added `Agent.resolved_context`: what the declared `context_params`
  resolve to right now, which is the mapping `_substitute` already
  built and threw away. A `callable` resolver runs inside
  substitution, so nothing afterwards could say which time of day,
  nickname or hint a finished conversation ran with — and at
  temperature a re-run is a different conversation. `_substitute`
  now reads the same property, so the two cannot drift.
- Added to `eval.chat_review`:
  - `Recording.context`: what the agent's context params resolved to
    when the conversation opened. Captured once, because a resolver
    returns a fresh value on every read.
  - `Recording.facts`: the character facts the store held at the
    start. A reply can be shaped by an earlier conversation and a
    transcript does not show it.
  - A context block above every transcript, carrying the date and
    time, the agent and its version, the models that answered, each
    context param, the facts it already knew and the prompt variants
    it drew. It stays visible in Read mode: it is context for the
    chat, the way a messenger names who you are talking to, not part
    of the review apparatus the toggle strips.
  - A recording made before this version has no `context` key. It
    loads with an empty one and the page says `not recorded`, which
    is not the same as a conversation that ran with no context.
- Added tests:
  - `tests/test_context_params.py`: four on `resolved_context`,
    including that it is what the prompt is actually built from.
  - `tests/test_chat_review.py`: six on capturing, round-tripping and
    rendering the context, and on an older recording still loading.


### v9.16.1 - 20th September 2026

- Removed `.pre-commit-config.yaml`: the monorepo root config is the only
  one, and this one was a trap. `pre-commit install` run from this
  directory — the only one holding a venv with pre-commit — resolved it and
  wrote `--config=agents7_framework/.pre-commit-config.yaml` into the hook,
  silencing four of the root's six guards, `check-changelog-headers` and
  `check-version-changelog` among them. Its own two `uv run ruff` hooks
  could not run either: pre-commit works from the repo root, which has no
  `pyproject.toml` for `uv run` to resolve.


### v9.16.0 - 19th September 2026

- Updated `agents/context.py`:
  - `ContextParam.identity`: the declaration — name, default and where a resolver comes from — as it enters a run's fingerprint.
- Updated `agents/base.py`:
  - `system_prompt_identity`: leaves `context_params` as placeholders and declares them after the prompt, the way it already handled `prompt_variants`. It resolved them before, so a `callable` reading the clock put the time of day in the hash and split one configuration's style history into a band per part of the day, and a nickname passed to `set_context` counted as a different agent.
  - `_substitute`: takes `with_context`, to leave those names unfilled when building the identity.
- Added tests in `tests/test_context_params.py`:
  - Six covering run identity, including that two runs hashed at different hours share a fingerprint.
- Updated `README.md`: what a context param contributes to the identity, and why.


### v9.15.2 - 21st September 2026

- Rotated changelog: archived 6 entries to `changelog/2026.md`.


### v9.15.1 - 17th September 2026

- Updated `agents/base.py`: `_set_attr` says it is only for fields read at call time. A `Detector` builds what it sends into `_prompt` in `__post_init__`, so replacing a built guard's `system_prompt` this way changes nothing — use `with_overrides`.


### v9.15.0 - 17th September 2026

- Updated `agents/context_guard.py`: the default guard prompt accepts a cut-off the scene rules call for — when a rule says the character declines or ends the conversation, a short reply that keeps declining is in character, even a bare goodbye, even ignoring what the user asked, even repeating an earlier goodbye. The v9.14.0 rule on goodbyes nobody asked for was reading those refusals as goodbyes, while deleting that rule would have let an early goodbye through.
  - Changes the guard prompt of every guarded scene agent, so guarded style histories start a fresh baseline; unguarded ones, that agent's gate included, are unaffected.
- Added `test_context_guard_prompt_accepts_a_cut_off_the_scene_calls_for` in `tests/test_context_guard.py`.


### v9.14.0 - 17th September 2026

- Updated `agents/context_guard.py`: the default guard prompt rejects a goodbye the character starts before a scene rule allows it, as `OFF_TOPIC`, however dry the user's answers. Staying in the conversation, offering to leave or leaving the next move to the user is not a goodbye, and once the user says they are leaving the rule no longer applies.
  - Without the closing sentence about the user leaving, the guard also rejected the character's replies to the user's own goodbye.
- Added `test_context_guard_prompt_rejects_a_goodbye_nobody_asked_for` in `tests/test_context_guard.py`.


### v9.13.0 - 16th September 2026

- Updated `eval/style.py`:
  - `StyleReport.variants`: the `prompt_variants` texts each scored run was shown, by run label.
  - `run_style_contract()` and `run_style_contract_async()` fill it from each fresh agent.
- Updated `eval/style_history.py`:
  - `StyleRunRecord.run_variants`: the same, persisted once per run; empty on older records.
  - `StyleRunRecord.by_variant()`: adds up one rule's score per text shown.
- Added tests:
  - `tests/test_style_contract.py`: 4 tests.
  - `tests/test_style_history.py`: 6 tests.
- Updated `README.md`: *Prompt Variants* section.


### v9.12.0 - 16th September 2026

- `agents/`:
  - Added `prompt_variants` in `base.py`: alternative texts for a `$variable`, one drawn per agent, because a model shown several alternatives answers with the first:
    - `BaseAgent.draw_variants()`: draws again, repeatably with a seed.
    - `BaseAgent.variant_values`: what was drawn.
    - `BaseAgent.system_prompt_identity`: the prompt with every alternative listed instead of the draw.
  - Updated `chatter.py`: `Chatter.reset()` draws again without a memory backend; with one, the draw is seeded by `conversation_id` so a rebuilt agent keeps it.
  - Updated `config_loader.py`: every loader reads a `prompt_variants` section, via `parse_prompt_variants()`.
- Updated `eval/`:
  - `style_history.py`: fingerprints hash `system_prompt_identity`, so runs that drew differently share a baseline.
  - `chat_review.py`: `Recording.variants` saves what the recorded conversation drew.
- Added `tests/test_prompt_variants.py`: 22 tests.
- Updated `README.md`: *Prompt Variants* section.


### v9.11.1 - 16th September 2026

- Rotated changelog: archived 13 entries to `changelog/2026.md`.


### v9.11.0 - 15th September 2026

- Updated `eval/style_history.py`: the margin from prior invocations is
  read at the upper bound of their spread, not its point estimate:
  - Added `SPREAD_CONFIDENCE` (0.90), exported from `agents7.eval`.
  - `sampling_margin` / `compare_to_history`: four priors that happen to
    agree no longer give a margin of about a point. Replayed over
    a chat agent's gate history, false gate failures on unchanged configs
    went from six to none; median margin went from 9.2 to 16.7 points.
  - `detectable_shift` / `runs_for_shift`: size against the same bar.
- Added tests in `tests/test_style_history.py`, including the
  chi-square quantile against its closed forms.
- Updated `.claude/rules/measurement-needs-history.md` (rule 4) and
  `README.md`.
