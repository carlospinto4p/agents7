#!/usr/bin/env bash
# Pre-commit hook: block changelog prepends that destroy an existing
# version header.
#
# The failure this guards against: a "prepend" that overwrites the top
# `### vX.Y.Z` header with the new entry instead of inserting a new
# block above it — the old header is deleted and its bullets are
# orphaned under the new entry.
#
# A header *removal* is legitimate only when the same version is
# re-added elsewhere in the same commit (rotation into
# `changelog/YYYY.md`, or a header reformat/typo-fix). So the check
# fails only when a version's header is removed and *not* re-added
# anywhere in the staged changelog diff.
#
# Pure bash + git/grep — no Python — so it drops into any repo (Rust,
# Kotlin, Svelte, …) without pulling in a language runtime.
#
# The pathspec is depth-agnostic (`*changelog.md` / `*changelog/*.md`):
# it covers a flat repo's root `changelog.md` plus its `changelog/`
# archive AND every subproject `changelog.md` in a monorepo (e.g.
# `agents7_ui/changelog.md`), so a single root-level hook guards them
# all.

set -u

# `--diff-filter=d` leaves out changelogs deleted whole: a project moved
# out of a monorepo takes its changelog with it, and a deleted file
# cannot be the overwrite-prepend bug this hook guards against.
diff="$(git diff --cached --diff-filter=d -- '*changelog.md' '*changelog/*.md')"
if [ -z "$diff" ]; then
  exit 0
fi

# Extract the version strings from changelog header diff lines for a
# given sign. A header diff line is `<sign>### vX.Y.Z …`; file-header
# lines (`--- a/…` / `+++ b/…`) never match — their char after the
# sign is `-`/`+`, not `#`. The sign is wrapped in `[...]` so `+`/`-`
# are treated literally by the regex.
extract() {
  printf '%s\n' "$diff" \
    | grep -E "^[$1]### +v[0-9]+\.[0-9]+\.[0-9]+" \
    | grep -oE "v[0-9]+\.[0-9]+\.[0-9]+" \
    | sort -u
}

removed="$(extract '-')"
added="$(extract '+')"

# Versions whose header was removed but never re-added.
destroyed=""
while IFS= read -r v; do
  [ -n "$v" ] || continue
  if ! printf '%s\n' "$added" | grep -qxF -- "$v"; then
    destroyed+="${destroyed:+ }$v"
  fi
done <<<"$removed"

if [ -n "$destroyed" ]; then
  echo "ERROR: this commit deletes changelog version header(s) without re-adding them:"
  for v in $destroyed; do
    echo "  $v"
  done
  cat <<'MSG'

This is the overwrite-prepend bug: a new entry replaced an existing
'### vX.Y.Z' header instead of being inserted above it, orphaning that
entry's bullets. Insert the new entry in the blank gap above the current
top header — never edit or replace the header line itself.
(If you genuinely renamed a header's version, re-stage so both the old
and new versions appear in the diff.)
MSG
  exit 1
fi

exit 0
