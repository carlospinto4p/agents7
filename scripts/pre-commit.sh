#!/bin/bash
# Project-local pre-commit checks: the unit tests. A Claude Code hook
# can run this before `git commit`. Exit non-zero to block the commit.
set -e

if [[ -d "tests/unit" ]]; then
    TEST_PATH="tests/unit/"
elif [[ -d "tests" ]]; then
    TEST_PATH="tests/"
else
    echo "pre-commit.sh: no tests directory found; skipping." >&2
    exit 0
fi

uv run pytest "$TEST_PATH" --tb=short -q
