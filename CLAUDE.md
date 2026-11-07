# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Core Rules

1. **Self-Improvement**: When the user corrects a mistake, ALWAYS update the
   relevant guidelines (`.claude/rules/` or this file) to prevent it from
   happening again.
2. **Keep CLAUDE.md Minimal**: Do not include library schemas, architecture
   details, or information discoverable from the codebase. Keep only
   essential rules and commands here.
3. **Periodic Reviews**: Triggered by backlog depth — when `/backlog` shows
   fewer than 5 open items, propose `/scan`, `/improvements`, and `/prune`
   together.

## Project Overview

`agents7` is a Python library for LLM-based agents that process data and
return structured outputs via Pydantic models. Supports Ollama and any
OpenAI-compatible API as LLM providers.

It is one of several repositories (`docs/repository-split.md`): the UI and
each agent live in repositories of their own and depend on a tag of this
one. On the owner's machine they sit side by side in one `agents7/`
folder, this one as `framework/`.

## Shell Commands

Never use compound shell commands with `cd && git` or `cd &&` chaining.
Always use separate commands or `cd` first, then run the git/shell command
independently.

## Project Configuration

Store project rules and preferences in project config files (e.g.,
`.claude/settings.json`, `CLAUDE.md`), NOT in memory files. Never save
user instructions as memory files.

## Common Commands

**IMPORTANT**: Always use `uv run` to execute Python commands. Never run
raw `python` commands.

```bash
# Install dependencies
uv sync --all-extras

# Run unit tests
uv run pytest tests/ -v

# Run linter
uv run ruff check src/ tests/

# Run linter with auto-fix
uv run ruff check src/ tests/ --fix
```

## Versioning / Release

After version bumps, always stage and commit `uv.lock` along with the
version change. Never forget the lock file.

## Testing

After any code change, run the full test suite and fix all failures before
committing. Pay special attention to mock patching paths, fixture values,
and assertion text that may need updating after refactors.

## Code Style

- Line length: 78 characters (enforced by ruff)
- Linter: ruff
- Type hints required
- Docstring style: Sphinx/reST (`:param:`, `:return:`, `:raises:`)
- In docstrings, prefer single backticks (`` `name` ``) for inline code
  references
- Use `@staticmethod` for methods that don't use `self`

## Workflow Rules

See `.claude/rules/` for detailed guidelines:
- [Committing](.claude/rules/committing.md)
- [Versioning](.claude/rules/versioning.md)
- [Testing](.claude/rules/testing.md)
- [Metrics completeness](.claude/rules/metrics-completeness.md)
- [Measurement needs history](.claude/rules/measurement-needs-history.md)

## After Every Change

1. Describe changes made
2. Update `README.md` if the change affects the public API
3. Run `uv pip install -e .`, then `git add uv.lock`
4. Run both test suites:
   - Unit: `uv run pytest tests/ -v`
   - Integration: `uv run pytest integration_tests/ -v`
5. Commit, version (`pyproject.toml` + `changelog.md`), push

## Samples (`samples/`)

Each sample is a `config.yml` plus an optional golden set, demonstrating
one framework capability. No package, no version, no tests of their own:
`tests/test_sample_golden_sets.py` checks their golden sets. A
structured-output sample's `output_model` resolves to a dotted path the
framework ships in `agents7.models.demo`; a tool sample (e.g.
`store_assistant`) resolves `tools` the same way, from
`agents7.tools.demo`. A sample-only change is unversioned.
