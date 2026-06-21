"""Unit tests for agents7.eval.data (eval-data helpers)."""

import json

import pytest

from agents7.agents.classifier import Classifier
from agents7.agents.labeler import Labeler
from agents7.eval.data import (
    AGENT_TYPE_MAP,
    build_agent,
    load_config_dict,
    load_eval_rows,
    model_output_fields,
    normalise_rows,
    resolve_agent_class,
)

# ── load_eval_rows ─────────────────────────────────────────────


def test_load_eval_rows_jsonl(tmp_path):
    """JSONL files load one object per non-blank line."""
    path = tmp_path / "g.jsonl"
    path.write_text(
        '{"text": "a", "category": "X"}\n\n{"text": "b", "category": "Y"}\n',
        encoding="utf-8",
    )
    rows = load_eval_rows(path)
    assert rows == [
        {"text": "a", "category": "X"},
        {"text": "b", "category": "Y"},
    ]


def test_load_eval_rows_json_array(tmp_path):
    """JSON files load a top-level array of objects."""
    path = tmp_path / "g.json"
    path.write_text(
        json.dumps([{"text": "a"}, {"text": "b"}]), encoding="utf-8"
    )
    assert load_eval_rows(path) == [{"text": "a"}, {"text": "b"}]


def test_load_eval_rows_json_non_list_raises(tmp_path):
    """A JSON object (not array) is rejected."""
    path = tmp_path / "g.json"
    path.write_text('{"text": "a"}', encoding="utf-8")
    with pytest.raises(ValueError, match="array of objects"):
        load_eval_rows(path)


def test_load_eval_rows_csv(tmp_path):
    """CSV files load each row as a dict of string values."""
    path = tmp_path / "g.csv"
    path.write_text("nick,category\nfoo,MALE\nbar,FEMALE\n", encoding="utf-8")
    assert load_eval_rows(path) == [
        {"nick": "foo", "category": "MALE"},
        {"nick": "bar", "category": "FEMALE"},
    ]


# ── normalise_rows ─────────────────────────────────────────────


def test_normalise_rows_empty():
    """Empty input yields empty outputs."""
    assert normalise_rows([]) == ([], [])


def test_normalise_rows_nested():
    """Nested rows keep their input and extract expected fields."""
    rows = [
        {"input": {"text": "a"}, "category": "X"},
        {"input": {"text": "b"}, "category": "Y"},
    ]
    eval_rows, fields = normalise_rows(rows, {"category"})
    assert fields == ["category"]
    assert eval_rows == [
        {"input": {"text": "a"}, "category": "X"},
        {"input": {"text": "b"}, "category": "Y"},
    ]


def test_normalise_rows_flat_single_input_col():
    """A flat row with one input column unwraps to a scalar."""
    rows = [{"nick": "foo", "category": "MALE"}]
    eval_rows, fields = normalise_rows(rows, {"category"})
    assert fields == ["category"]
    assert eval_rows == [{"input": "foo", "category": "MALE"}]


def test_normalise_rows_flat_multi_input_col():
    """Multiple input columns are kept as an input dict."""
    rows = [{"first": "a", "last": "b", "category": "X"}]
    eval_rows, fields = normalise_rows(rows, {"category"})
    assert eval_rows[0]["input"] == {"first": "a", "last": "b"}
    assert fields == ["category"]


def test_normalise_rows_excludes_metadata():
    """Metadata columns are neither input nor expected."""
    rows = [{"text": "a", "category": "X", "note": "n", "source": "s"}]
    eval_rows, fields = normalise_rows(rows, {"category"})
    assert eval_rows[0]["input"] == "a"
    assert fields == ["category"]


def test_normalise_rows_category_sorted_first():
    """`category` is ordered ahead of other expected fields."""
    rows = [{"text": "a", "reason": "R", "category": "X"}]
    _, fields = normalise_rows(rows, {"category", "reason"})
    assert fields == ["category", "reason"]


def test_normalise_rows_flat_no_input_raises():
    """A flat set whose every column is expected has no input."""
    rows = [{"category": "X"}]
    with pytest.raises(ValueError, match="input column"):
        normalise_rows(rows, {"category"})


# ── model_output_fields ────────────────────────────────────────


def test_model_output_fields_resolves_schema():
    """Fields are read from the output model's JSON schema."""
    fields = model_output_fields({"output_model": "helpers.DummyOutput"})
    assert fields == {"value"}


def test_model_output_fields_missing_model():
    """No output_model yields an empty set."""
    assert model_output_fields({}) == set()


def test_model_output_fields_unimportable():
    """An unimportable model yields an empty set, not an error."""
    assert model_output_fields({"output_model": "nope.NotAModel"}) == set()


# ── resolve_agent_class / build_agent ──────────────────────────


def test_resolve_agent_class_known():
    """Each known type resolves to its class."""
    assert resolve_agent_class("classifier") is Classifier
    assert resolve_agent_class("labeler") is Labeler


def test_resolve_agent_class_defaults_to_labeler():
    """None / empty defaults to Labeler."""
    assert resolve_agent_class(None) is Labeler
    assert resolve_agent_class("") is Labeler


def test_resolve_agent_class_case_insensitive():
    """Type strings are matched case-insensitively."""
    assert resolve_agent_class("Classifier") is Classifier


def test_resolve_agent_class_unknown_raises():
    """An unknown type lists the valid options."""
    with pytest.raises(ValueError, match="Unknown agent_type"):
        resolve_agent_class("chatter")


def test_agent_type_map_covers_known_types():
    """The map exposes exactly the evaluable agent types."""
    assert set(AGENT_TYPE_MAP) == {
        "labeler",
        "detector",
        "classifier",
        "regressor",
    }


def test_build_agent_explicit_type():
    """An explicit agent_type overrides the config field."""
    agent = build_agent(
        "examples/configs/demo_sentiment_classifier.yml",
        agent_type="classifier",
    )
    assert isinstance(agent, Classifier)


def test_build_agent_defaults_to_labeler():
    """With no type given or configured, a Labeler is built."""
    agent = build_agent("examples/configs/demo_labeler.yml")
    assert isinstance(agent, Labeler)


def test_load_config_dict_reads_yaml():
    """The shallow config reader returns the parsed mapping."""
    cfg = load_config_dict("examples/configs/demo_labeler.yml")
    assert cfg["name"]
    assert "engine" in cfg
