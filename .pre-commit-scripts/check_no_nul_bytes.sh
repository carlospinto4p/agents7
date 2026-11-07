#!/usr/bin/env bash
# Pre-commit hook: reject committing a text file that contains a NUL
# byte (0x00).
#
# The failure this guards against: a stray NUL slipping into a source
# file — e.g. a JavaScript/TypeScript template-literal separator meant
# to be the `\u0000` *escape* but written as a raw NUL. A raw NUL is
# invisible in a normal diff, makes the file classify as "binary" (git
# and most editors refuse it as text), and otherwise compiles/runs
# fine — so it ships unnoticed. This guard makes the whole class fail
# loudly at commit time.
#
# pre-commit passes only files it classifies as text (the hook's
# `types: [text]`), so genuine binaries (images, fonts, compiled
# artifacts) are never flagged — a NUL in those is expected.
#
# Pure bash + coreutils (wc/tr) — no Python, no grep -P NUL-pattern
# quirks — so it drops into any repo (Rust, Svelte, Kotlin, …) without
# a language runtime, matching the other guards here.

set -u

status=0
for f in "$@"; do
  [ -f "$f" ] || continue
  orig=$(wc -c <"$f")
  stripped=$(tr -d '\000' <"$f" | wc -c)
  if [ "$orig" -ne "$stripped" ]; then
    echo "ERROR: NUL byte(s) found in text file: $f ($((orig - stripped)) byte(s))" >&2
    status=1
  fi
done

if [ "$status" -ne 0 ]; then
  cat <<'MSG' >&2

A NUL byte (0x00) in a text file is almost always a mistake — commonly a
`\u0000` escape typed as a raw NUL. Replace the raw NUL with the proper
escape (or remove it). If a file is genuinely binary, it should not be
classified as text — adjust .gitattributes / the hook's file filter.
MSG
fi

exit "$status"
