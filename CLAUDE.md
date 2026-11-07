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

## Common Commands

**IMPORTANT**: Always use `uv run` to execute Python commands. Never run
raw `python` commands.

```bash
# Install dependencies
uv sync --all-extras

# Unit tests
uv run pytest tests/ -v

# Integration tests (need a running LLM provider)
uv run pytest integration_tests/ -v

# Lint
uv run ruff check src/ tests/
```

## After Every Change

1. Update `README.md` if the change affects the public API.
2. Run the unit tests (and the integration tests when a provider path
   changed).
3. Bump the version in `pyproject.toml` and the lock's own `version` line,
   add a `changelog.md` entry (`scripts/changelog-add.sh`), commit, push.
4. A release that the UI or an agent needs gets a tag: `git tag vX.Y.Z`,
   then `git push --tags`.

Workflow details: `.claude/rules/committing.md`, `versioning.md`,
`testing.md`, `metrics-completeness.md`, `measurement-needs-history.md`.

## Code Style

- Line length: 78 characters (enforced by ruff, `ruff.toml`)
- Type hints required
- Docstring style: Sphinx/reST (`:param:`, `:return:`, `:raises:`)
- In docstrings, prefer single backticks (`` `name` ``) for inline code
  references
- Use `@staticmethod` for methods that don't use `self`

## Samples (`samples/`)

Each sample is a `config.yml` plus an optional golden set, demonstrating
one framework capability. No package, no version, no tests of their own:
`tests/test_sample_golden_sets.py` checks their golden sets. A
structured-output sample's `output_model` resolves to a dotted path the
framework ships in `agents7.models.demo`; a tool sample (e.g.
`store_assistant`) resolves `tools` the same way, from
`agents7.tools.demo`. A sample-only change is unversioned.
