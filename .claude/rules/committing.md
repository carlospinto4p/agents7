---
name: committing
description: Git commit conventions, pull-first step, lock-file handling, and post-change workflow
version: 1.7
---

# Committing Guidelines

## Pull First — Before Any Work

**The first step of every session's workflow is `git pull`.** Run it
before reading code, planning, or editing — not just before pushing.

- Use `git -C <repo> pull` as a standalone Bash call (never chained).
- Do this even when the working tree looks clean: the remote may have
  moved ahead from another machine.
- If the pull brings in commits, re-read anything you had already
  looked at — your view of the code is stale.

Pulling only before `git push` is too late: work built on a stale base
has to be redone.

## Concurrent Sessions — Verify Before Every Commit, Not Just After a Broad Add

A second live session in the same repo can commit while this one is
mid-task, and a broad `git add` (`-A`/`.`) there sweeps this session's
uncommitted edits into its unrelated commit.

- **Run `git status --porcelain` immediately before every `git
  commit`.** If files you expected to be uncommitted are missing,
  check `git log` for a commit you didn't make before proceeding.
- **Stage by name, never `-A`/`.`**, so you cannot sweep up another
  session's edits either.
- If your edits are already committed elsewhere, diff that commit
  against your working tree, then drop your redundant version bump.

## Post-Change Workflow

**MANDATORY after every prompt that results in code changes:**

1. **Run tests**: `uv run pytest tests/unit -v`, all passing.
2. **Update tests** the change requires, before committing.
3. **Update version and changelog** per `versioning.md` —
   config/tooling changes count too (see below).
4. **Update the `docs/` page** when user-facing behaviour changed: usage
   examples for new methods or classes, updated examples for changed
   signatures, docs for new configuration options. `README.md` stays a
   short overview that links to those pages (owner, 7th November 2026).
5. **Update CLAUDE.md** when rules change or new important patterns
   emerge.
6. **Sync the lock file**: every version bump updates `uv.lock`'s own
   `version` line in the same commit; where the repo has the
   `check-lock-version` pre-commit hook, it refuses the commit
   otherwise. On a **code-related** bump (source or dependency
   changes) run `uv sync --all-extras`. On a non-code bump (`.claude/`
   config, docs, changelog-only, cross-project rule syncs) run
   `uv run --no-sync python .pre-commit-scripts/check_lock_version.py
   --fix`, which edits only that line, rather than `uv lock`. **If
   `uv.lock` changed by any route, commit it** — never leave it dirty. Dev dependencies are PEP 735 groups, not extras: a
   bare `uv sync` restores everything, editable install included.
7. **Commit** with a message in the format below, **always including
   `uv.lock`** when it changed.
8. **Push** with `git push`.

**Config/tooling changes** (anything under `.claude/` and `CLAUDE.md`)
still take a patch bump and changelog entry with no code change; skip
tests only if no code changed. This is per project: a cross-project
migration touching N repos gets a bump and entry in each.

**Pure tracking changes** (`backlog.md` checkboxes only): commit and
push, no tests, no version bump.

**Lock-file-only changes** (`uv.lock` is the entire diff): commit and
push as a `chore:`, **no version bump and no changelog entry**. The
diff is almost always the lock's own `version` catching up to an
already-released `pyproject.toml`; a bump would recreate the drift.
The exception is a **broken or substantively changed** lock (a
dependency added, removed or upgraded, or a bad resolution repaired),
which takes a patch bump and entry like any code change. The test:
does the diff touch anything beyond the `[[package]] name = "<this
project>"` version line?

## When to Skip Tests

The default is to run them. **Skip only when the staged diff is
exclusively** markdown/docs (`*.md`), a `pyproject.toml` version bump,
`uv.lock`, `.claude/` config files, or `CLAUDE.md`.

If **any** `.py` file is in the diff (including under `tests/`), or
the diff mixes a skippable file with anything else, run tests. When
in doubt, run them — but not a long suite for a one-line markdown
edit. Check `git status --porcelain` before deciding: the signal is
the diff, not your intent.

## Commit Message Format

Use conventional commit style:

```
<type>: <description>

[optional body]
```

## Types

`feat` (new functionality), `fix` (bug fix), `refactor` (no behaviour
change), `docs`, `test`, `chore` (maintenance, dependency updates).

## Best Practices

- Subject line under 72 characters, imperative mood ("Add feature").
- **Default to subject-only.** Skip the body unless it adds something
  the subject, diff and changelog don't.
- Reference issues when applicable.
- **One command per Bash call** — never chain git commands with `&&`,
  `;`, heredocs, or subshells.
- **Never put a git command in a parallel tool batch.** A
  `PreToolUse` hook that rejects one call cancels every other call in
  the same batch, silently discarding the reads and edits bundled with
  it. Send each git operation alone and wait for its result.
- **Use repeated `-m` flags for commit bodies**, not heredocs or
  `$(cat <<EOF …)`, which trip the chained-command hook.

## When to Add a Body

Only when the *why* is non-obvious and not captured elsewhere: a
workaround for a specific bug or upstream issue (link it), a hidden
constraint explaining an unusual choice, or an incident the diff won't
surface. Never a body that restates the diff or describes a new file
(the changelog does that). Separate it from the subject with a blank
line; one short paragraph.

## Examples

```
feat: Add project registry with version tracking
```

```
fix: Handle missing changelog in cross-project sync
```
