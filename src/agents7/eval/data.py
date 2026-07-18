"""Eval-data loading, normalisation, and agent construction.

Shared glue for turning a YAML-configured agent and a labeled
golden set into the canonical shape `run_evaluation` expects.
Used by the `agents7` CLI and reusable by any caller that needs
to evaluate an agent headlessly.

A golden-set row may be **nested** (already carrying an
``"input"`` key, e.g. ``{"input": {...}, "category": "..."}``)
or **flat** (input and expected columns side by side, e.g.
``{"text": "...", "category": "..."}``). Both are normalised to
``{"input": <input>, <field>: <expected>, ...}`` before scoring.
"""

import csv
import json
from pathlib import Path

from agents7.agents.base import BaseAgent
from agents7.agents.utils import import_from_path

# agent_type -> dotted class path. Every evaluable agent type,
# scored one of two ways: the structured-output agents compare
# output fields against a labeled row, while `tool` compares the
# trajectory it took (see `agents7.eval.trajectory`). Chatters
# stay absent — they have no golden set to score against.
AGENT_TYPE_MAP: dict[str, str] = {
    "labeler": "agents7.agents.Labeler",
    "detector": "agents7.agents.Detector",
    "classifier": "agents7.agents.Classifier",
    "regressor": "agents7.agents.Regressor",
    "tool": "agents7.agents.ToolAgent",
}

# Columns that annotate a golden-set row but are neither input
# nor an expected output field.
_METADATA_COLUMNS: frozenset[str] = frozenset(
    {"timestamp", "note", "source", "id", "split", "confidence"}
)


def load_config_dict(path: str | Path) -> dict:
    """Parse a YAML agent config into a plain dict.

    Inheritance (``base_config``) is **not** resolved — this is
    a shallow read for metadata such as ``agent_type`` and
    ``output_model``. The agent itself is built by ``from_yaml``,
    which does resolve inheritance.

    :param path: Path to the YAML config file.
    :return: Parsed configuration dictionary (empty if the file
        is blank).
    """
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_eval_rows(path: str | Path) -> list[dict]:
    """Load eval rows from a ``.csv``, ``.jsonl``, or ``.json``
    file.

    - ``.csv``: each row becomes a dict of column → string value.
    - ``.jsonl``: one JSON object per non-blank line.
    - ``.json``: a top-level JSON array of objects.

    :param path: Path to the eval-set file.
    :return: List of raw row dicts.
    :raises ValueError: If a ``.json`` file does not contain a
        list of objects.
    """
    path = Path(path)
    if path.suffix == ".csv":
        with open(path, encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                f"`{path}` must contain a JSON array of objects."
            )
        return [dict(row) for row in data]
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def model_output_fields(config: dict) -> set[str]:
    """Resolve the output model's field names from a config.

    Returns an empty set when no ``output_model`` is declared or
    the model cannot be imported (e.g. the agent package is not
    installed in the current environment).

    :param config: Parsed agent configuration dictionary.
    :return: Set of output-model field names.
    """
    dotted = config.get("output_model")
    if not dotted:
        return set()
    try:
        model_cls = import_from_path(dotted)
        schema = model_cls.model_json_schema()
        return set(schema.get("properties", {}).keys())
    except Exception:
        return set()


def normalise_rows(
    rows: list[dict],
    model_fields: set[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Normalise raw golden-set rows to the canonical eval shape.

    Each output row is ``{"input": <input>, <field>: <expected>,
    ...}``. Supports both nested rows (already carrying an
    ``"input"`` key) and flat rows (input + expected columns side
    by side). When a flat row has a single input column, its value
    is unwrapped to a scalar; multiple input columns are kept as a
    dict.

    :param rows: Raw rows from `load_eval_rows`.
    :param model_fields: Output-model field names. When provided,
        the expected fields are intersected with these so input
        columns are not mistaken for expected outputs.
    :return: ``(eval_rows, expected_fields)`` where
        *expected_fields* is the ordered list of fields to score
        (``category`` first when present).
    :raises ValueError: If a flat eval set has no determinable
        input column(s).
    """
    if not rows:
        return [], []

    model_fields = model_fields or set()
    nested = any("input" in row for row in rows)
    columns: set[str] = set().union(*(row.keys() for row in rows))

    if nested:
        expected = {
            k for k in columns if k != "input" and k not in _METADATA_COLUMNS
        }
        if model_fields:
            expected &= model_fields
        eval_rows = [
            {
                "input": row.get("input", {}),
                **{k: row[k] for k in expected if k in row},
            }
            for row in rows
        ]
    else:
        if model_fields:
            expected = model_fields & columns
        else:
            expected = columns - _METADATA_COLUMNS
        input_cols = columns - expected - _METADATA_COLUMNS
        if not input_cols:
            raise ValueError(
                "Could not determine input column(s) for flat eval "
                "set — declare an output_model or use a nested eval "
                "set."
            )
        eval_rows = []
        for row in rows:
            if len(input_cols) == 1:
                input_data = row[next(iter(input_cols))]
            else:
                input_data = {c: row[c] for c in input_cols if row.get(c)}
            eval_rows.append(
                {
                    "input": input_data,
                    **{k: row[k] for k in expected if row.get(k)},
                }
            )

    fields = sorted(expected, key=lambda f: (f != "category", f))
    return eval_rows, fields


def resolve_agent_class(agent_type: str | None) -> type:
    """Resolve an ``agent_type`` string to an agent class.

    :param agent_type: One of ``labeler``, ``detector``,
        ``classifier``, ``regressor``, ``tool``
        (case-insensitive). When ``None`` or empty, defaults to
        ``labeler``.
    :return: The resolved agent class.
    :raises ValueError: If *agent_type* is not a known type.
    """
    key = (agent_type or "labeler").lower()
    class_path = AGENT_TYPE_MAP.get(key)
    if class_path is None:
        raise ValueError(
            f"Unknown agent_type '{agent_type}' — must be one of: "
            f"{sorted(AGENT_TYPE_MAP)}"
        )
    return import_from_path(class_path)


def build_agent(
    config_path: str | Path,
    agent_type: str | None = None,
) -> BaseAgent:
    """Build an evaluable agent from a YAML config.

    Thin wrapper over the agent class's ``from_yaml``. When
    *agent_type* is not given, it is read from the config's
    ``agent_type`` field (defaulting to ``labeler``).

    :param config_path: Path to the YAML agent config.
    :param agent_type: Optional explicit agent type, overriding
        the config's ``agent_type`` field.
    :return: A configured agent instance.
    :raises ValueError: If the resolved agent type is unknown.
    """
    if agent_type is None:
        agent_type = load_config_dict(config_path).get("agent_type")
    agent_cls = resolve_agent_class(agent_type)
    return agent_cls.from_yaml(str(config_path))
