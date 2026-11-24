#!/usr/bin/env bash
# SessionStart hook: prints the cloud-session rule, so Claude sees it as
# context. Silent outside a cloud session (Claude Code on the web or in
# the Claude app), where CLAUDE_CODE_REMOTE is "true".
# See https://code.claude.com/docs/en/cloud-environments.md
[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

cat <<'EOF'
# Cloud session: land every prompt's work on `main`

This session runs in the cloud. A new session starts from `main`, so work
left on a branch is invisible to it. At the end of every prompt that
committed something:

1. `git pull --no-rebase origin main`, to take in anything `main` gained.
2. `git push origin HEAD:main`.
3. If the session's instructions name a branch to push to, push it too.

If GitHub refuses the push to `main` (a protection rule or a missing
permission), open a pull request from the branch into `main` and merge it
with a merge commit instead. Tell the user which of the two happened.
EOF
