#!/usr/bin/env bash
# Prepend a new version entry to changelog.md SAFELY — the mechanical fix for
# the recurring "overwrite-prepend" mistake (a hand-edit that replaces the top
# `### vX.Y.Z` header instead of inserting above it, orphaning that version's
# bullets under the new entry; see .pre-commit-scripts/check_changelog_headers.sh).
#
# It NEVER edits an existing header: the new block is spliced in immediately
# BEFORE the first `### vX.Y.Z` line, so every prior header is only shifted down.
# That makes the overwrite bug structurally impossible — which a careless Edit
# can't promise.
#
# Usage:
#   scripts/changelog-add.sh <version> [date] <<'EOF'
#   - **Headline.** Body of the entry (one or more bullets / lines).
#   EOF
#
#   <version>  e.g. v0.1.107 (a leading 'v' is added if you omit it)
#   [date]     optional; defaults to today as "Dth Month YYYY" (e.g. "20th June
#              2026"), matching the existing entries. Forced to the C locale so
#              the month name is English regardless of the shell locale.
#   body       read from stdin (heredoc or a pipe). Required, non-empty.
#
#   -f, --file PATH   changelog to edit (default: ./changelog.md)
#
# Wrong-repo guard: when -f/--file is omitted, the default './changelog.md'
# is resolved against the CWD, and the script then refuses to run unless
# that directory's manifest (pyproject.toml / Cargo.toml / build.gradle.kts
# / package.json) is already AT the <version> you're adding — the normal
# workflow bumps the manifest before adding the changelog entry. This turns
# "wrong repo's changelog" from a silent write into a refusal. Pass -f/--file
# explicitly (this always skips the check) when editing a changelog outside
# the CWD's own tree, e.g. from another repo's shell.
#
# The body is inserted verbatim — write it exactly as it should appear
# (leading "- " bullets, wrapped lines, etc.). The version you pass should match
# the manifest bump in the same commit; the `check-version-changelog` guard
# enforces that at commit time.
#
# NOTE: this always writes a "### vX.Y.Z - date" header — it is built for
# version-numbered changelogs. A changelog that has moved to plain-date
# headers with no version number at all will still get a versioned header
# spliced in at the now-correct top position, which won't match the
# surrounding entries' format. For a changelog like that, hand-edit
# instead, anchoring the edit on the title line.
set -euo pipefail

file="changelog.md"
file_explicit=0
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file) file="$2"; file_explicit=1; shift 2 ;;
    -h|--help) sed -n '2,44p' "$0"; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) args+=("$1"); shift ;;
  esac
done

if [[ ${#args[@]} -lt 1 || ${#args[@]} -gt 2 ]]; then
  echo "usage: scripts/changelog-add.sh <version> [date] < body" >&2
  exit 2
fi

version="${args[0]}"
[[ "$version" == v* ]] || version="v$version"
if [[ ! "$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Bad version '$version' — want vMAJOR.MINOR.PATCH (e.g. v0.1.107)." >&2
  exit 2
fi

if [[ ${#args[@]} -eq 2 ]]; then
  date_str="${args[1]}"
else
  day="$(LC_ALL=C date +%-d)"
  case "$day" in
    1|21|31) suf=st ;;
    2|22) suf=nd ;;
    3|23) suf=rd ;;
    *) suf=th ;;
  esac
  date_str="${day}${suf} $(LC_ALL=C date +'%B %Y')"
fi

[[ -f "$file" ]] || { echo "No changelog at '$file'." >&2; exit 1; }

# Guard against the "wrong repo" footgun: when -f/--file was NOT passed
# explicitly, the default './changelog.md' is CWD-relative, so invoking the
# script by its own absolute path from a different repo's shell silently
# targets that other repo's changelog (this has happened: an entry for one
# project landed in another project's changelog). Cross-check: the
# changelog's directory must
# hold a manifest whose version already equals <version> — the normal
# workflow bumps the manifest before adding the changelog entry, so this
# holds whenever the script targets the intended repo.
if [[ "$file_explicit" -eq 0 ]]; then
  dir="$(cd "$(dirname "$file")" && pwd)"
  manifest=""
  manifest_ver=""
  for candidate in \
    "pyproject.toml:^version\s*=\s*\"([0-9]+\.[0-9]+\.[0-9]+)\"" \
    "src-tauri/Cargo.toml:^version\s*=\s*\"([0-9]+\.[0-9]+\.[0-9]+)\"" \
    "Cargo.toml:^version\s*=\s*\"([0-9]+\.[0-9]+\.[0-9]+)\"" \
    "app/build.gradle.kts:versionName\s*=\s*\"([0-9]+\.[0-9]+\.[0-9]+)\"" \
    "package.json:\"version\"\s*:\s*\"([0-9]+\.[0-9]+\.[0-9]+)\""
  do
    rel="${candidate%%:*}"
    pattern="${candidate#*:}"
    path="$dir/$rel"
    [[ -f "$path" ]] || continue
    v="$(grep -m1 -oP "$pattern" "$path" | grep -oP '[0-9]+\.[0-9]+\.[0-9]+' || true)"
    [[ -n "$v" ]] || continue
    manifest="$rel"
    manifest_ver="$v"
    break
  done

  version_num="${version#v}"
  if [[ -z "$manifest" ]]; then
    echo "Refusing: no manifest (pyproject.toml/Cargo.toml/app/build.gradle.kts/package.json) with a parsable version found next to '$file'. Pass -f/--file explicitly to confirm the target repo." >&2
    exit 1
  fi
  if [[ "$manifest_ver" != "$version_num" ]]; then
    echo "Refusing: '$dir/$manifest' is at version $manifest_ver, but you're adding a $version entry to '$file'. Wrong repo's changelog? Pass -f/--file to target the right one explicitly." >&2
    exit 1
  fi
fi

# Body from stdin; command substitution trims trailing newlines so we control
# the spacing exactly.
body="$(cat)"
[[ -n "$body" ]] || { echo "Empty body on stdin — nothing to add." >&2; exit 2; }

# Refuse to add a header that already exists (a duplicate would be a separate
# kind of mess). Match the version token at a header line start.
if grep -qE "^### ${version} " "$file"; then
  echo "$version already has a changelog header — refusing to duplicate." >&2
  exit 1
fi

# The splice point: the first existing entry header — ANY `### ` line, not
# just a version-shaped one. Everything before it (title + blank gap) is kept
# as-is; the new block goes in just above it, so no existing header is ever
# touched. If there is no header yet, append after the whole file.
#
# Matching only "### v..." (or bare "### 0.1.2") missed changelogs that don't
# head entries with a version at all — e.g. one that moved to plain
# "### <date>" headers with no version number for its recent entries.
# There the version-only regex skipped straight past those to the first
# *older* version-numbered header further down, silently splicing the new
# entry there instead of at the true top. Anchoring on any `### ` line fixes
# this generically: the topmost one is always the true top of the changelog,
# whatever it's headed with.
hdr_line="$(grep -nE '^### ' "$file" | head -1 | cut -d: -f1 || true)"

# The gap under the new entry copies the one the file already leaves above its
# second entry header: two blank lines in most projects, one in some. A
# fixed count broke one side or the other (v4.152.3 hard-coded one). With no
# second entry to measure, use two, as versioning.md asks.
gap=2
hdr2_line="$(grep -nE '^### ' "$file" | sed -n 2p | cut -d: -f1 || true)"
if [[ -n "$hdr2_line" ]]; then
  gap=0
  while (( hdr2_line - gap - 1 > 0 )) &&
    [[ -z "$(sed -n "$((hdr2_line - gap - 1))p" "$file")" ]]; do
    gap=$((gap + 1))
  done
  (( gap > 0 )) || gap=2
fi

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
if [[ -n "$hdr_line" ]]; then
  head -n "$((hdr_line - 1))" "$file" >"$tmp"
  printf '### %s - %s\n\n%s\n' "$version" "$date_str" "$body" >>"$tmp"
  printf '%*s' "$gap" '' | tr ' ' '\n' >>"$tmp"
  tail -n "+$hdr_line" "$file" >>"$tmp"
else
  # No prior entries: keep the title block, then the new entry.
  cat "$file" >"$tmp"
  printf '\n### %s - %s\n\n%s\n' "$version" "$date_str" "$body" >>"$tmp"
fi
mv "$tmp" "$file"
trap - EXIT

echo "Added '### $version - $date_str' to $file (above the previous top entry)."
