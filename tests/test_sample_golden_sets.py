"""Integrity tests for every sample's golden set in `samples/`.

Walks the samples generically, so a new sample with a golden set is
covered without a test of its own.

No LLM calls — these validate the data files against each sample's
declared `output_model`. From the repository root::

    uv run pytest tests/test_sample_golden_sets.py -v
"""

import json
from enum import Enum
from pathlib import Path

import pytest
import yaml

from agents7.agents.utils import import_from_path

SAMPLES_ROOT = Path(__file__).resolve().parents[1] / "samples"

GOLDEN_SETS = sorted(SAMPLES_ROOT.glob("*/data/golden_set.jsonl"))
IDS = [p.parent.parent.name for p in GOLDEN_SETS]


def _rows(path: Path) -> list[dict]:
    """Parse a JSONL golden set into a list of row dicts."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _enum_fields(config_file: Path) -> dict[str, type[Enum]]:
    """Map each enum-typed `output_model` field to its enum class."""
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    dotted = cfg.get("output_model")
    if not dotted:
        return {}
    model = import_from_path(dotted)
    fields = {}
    for name, info in model.model_fields.items():
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            fields[name] = annotation
    return fields


# ── Golden Set Integrity Tests ──────────────────────────────────


def test_samples_have_golden_sets():
    """The samples tree exposes at least one golden set to check."""
    assert GOLDEN_SETS, f"no golden sets found under {SAMPLES_ROOT}"


@pytest.mark.parametrize("path", GOLDEN_SETS, ids=IDS)
def test_golden_set_is_non_empty_jsonl(path: Path):
    """Every line parses as JSON and the file has at least one row."""
    assert _rows(path), f"golden set is empty: {path}"


@pytest.mark.parametrize("path", GOLDEN_SETS, ids=IDS)
def test_every_row_has_text(path: Path):
    """Every row carries the non-empty `text` input field."""
    for row in _rows(path):
        assert row.get("text"), f"row missing text: {row}"


@pytest.mark.parametrize("path", GOLDEN_SETS, ids=IDS)
def test_enum_labels_are_valid_members(path: Path):
    """Labels naming an enum field hold a valid member of that enum.

    Generalises the per-package category check: any row key matching
    an enum-typed field on the agent's `output_model` must hold a
    name declared by that enum.
    """
    config_file = path.parent.parent / "config.yml"
    enum_fields = _enum_fields(config_file)
    if not enum_fields:
        pytest.skip(f"{config_file.parent.name} has no enum-typed fields")
    for row in _rows(path):
        for field, enum_cls in enum_fields.items():
            if field not in row:
                continue
            valid = {m.name for m in enum_cls}
            assert row[field] in valid, (
                f"{path.name}: bad {field}={row[field]!r}, "
                f"expected one of {sorted(valid)}"
            )
