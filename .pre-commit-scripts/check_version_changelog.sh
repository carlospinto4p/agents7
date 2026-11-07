#!/usr/bin/env bash
# Pre-commit hook: block a version bump that ships with no changelog entry.
#
# When a commit changes a package's own version in a manifest
# (pyproject.toml `[project]`, Cargo.toml `[package]`, package.json top
# level) but the staged changelog diff adds no matching `### v<new>`
# header, the entry was forgotten — the failure that left manifold's
# changelog seven versions behind on 2026-06-17.
#
# Scope, kept conservative to avoid false positives:
#   - only the three manifests above are inspected;
#   - Cargo.toml / pyproject.toml *dependency* versions are ignored —
#     the read is section-aware ([package] / [project] only);
#   - a newly added manifest (no version at HEAD) is ignored, so a
#     freshly scaffolded project's first commit is never blocked;
#   - manifests under reservations/ are ignored — defensive
#     package-name placeholders (crates.io / npm / PyPI brand holds)
#     carry their own versions, independent of the app release;
#   - only X.Y.Z versions are tracked.
#
# Pure bash + git/grep/awk — no language runtime — so it drops into any
# repo (Rust, Svelte, Python, …).

set -u

# Package version for a manifest at a git ref ("" = staged index).
# Section-aware, so dependency versions never match.
pkg_version() {
  local path="$1" ref="$2" content
  content="$(git show "${ref}:${path}" 2>/dev/null)" || return 0
  case "$path" in
    *package.json)
      printf '%s\n' "$content" \
        | grep -oE '"version"[[:space:]]*:[[:space:]]*"[0-9]+\.[0-9]+\.[0-9]+"' \
        | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 ;;
    *Cargo.toml)
      printf '%s\n' "$content" | awk '
        /^\[package\]/ {p=1; next}
        /^\[/          {p=0; next}
        p && /^[[:space:]]*version[[:space:]]*=/ {
          if (match($0, /[0-9]+\.[0-9]+\.[0-9]+/)) {
            print substr($0, RSTART, RLENGTH); exit } }' ;;
    *pyproject.toml)
      printf '%s\n' "$content" | awk '
        /^\[project\]/      {p=1; next}
        /^\[tool\.poetry\]/ {p=1; next}
        /^\[/               {p=0; next}
        p && /^[[:space:]]*version[[:space:]]*=/ {
          if (match($0, /[0-9]+\.[0-9]+\.[0-9]+/)) {
            print substr($0, RSTART, RLENGTH); exit } }' ;;
  esac
}

# Versions bumped in this commit (old != new, and old non-empty so a
# brand-new manifest does not count).
bumped=""
while IFS= read -r f; do
  [ -n "$f" ] || continue
  new="$(pkg_version "$f" '')"
  [ -n "$new" ] || continue
  old="$(pkg_version "$f" HEAD)"
  [ -n "$old" ] || continue
  [ "$old" = "$new" ] && continue
  bumped="${bumped:+$bumped }$new"
# `reservations/` holds defensive package-name placeholders (crates.io / npm /
# PyPI brand holds) with their OWN versions, independent of the app release —
# bumping one must NOT demand an app changelog entry. Exclude that subtree.
done < <(git diff --cached --name-only -- '*pyproject.toml' '*Cargo.toml' '*package.json' ':(exclude)reservations/**')

bumped="$(printf '%s\n' $bumped | sort -u | grep -v '^$' || true)"
[ -z "$bumped" ] && exit 0

# Versions whose `### vX.Y.Z` header was added to a staged changelog.
cl_added="$(git diff --cached -- '*changelog.md' '*changelog/*.md' \
  | grep -E '^\+### +v[0-9]+\.[0-9]+\.[0-9]+' \
  | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | sort -u)"

missing=""
while IFS= read -r v; do
  [ -n "$v" ] || continue
  if ! printf '%s\n' "$cl_added" | grep -qxF -- "$v"; then
    missing="${missing:+$missing }$v"
  fi
done <<<"$bumped"

if [ -n "$missing" ]; then
  echo "ERROR: version bumped without a matching changelog entry:"
  for v in $missing; do
    echo "  v$v  — add '### v$v - <DDth Month YYYY>' to changelog.md"
  done
  cat <<'MSG'

A manifest version was bumped but no '### vX.Y.Z' header for it was
added to the changelog in this commit. Add the changelog entry in the
same commit so released versions never go undocumented.
(If you are intentionally splitting the bump and the changelog across
commits, stage the changelog entry here too.)
MSG
  exit 1
fi

exit 0
