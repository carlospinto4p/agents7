"""Unit tests for export_dataframe (Polars DataFrame export)."""

import pytest

pl = pytest.importorskip("polars")

from pydantic import BaseModel  # noqa: E402

from agents7.export import export_dataframe  # noqa: E402
from agents7.models.base import AgentResponse, LLMMetrics  # noqa: E402


# --- Helpers ---


class DummyModel(BaseModel):
    name: str
    score: int


def _make_response(name, score):
    return AgentResponse(
        response="raw",
        model=DummyModel(name=name, score=score),
        metrics=LLMMetrics(tokens=10, prompt_tokens=5, total_duration=1.0),
    )


def _make_none_response():
    return AgentResponse(
        response="bad",
        model=None,
        metrics=LLMMetrics(tokens=5, prompt_tokens=3, total_duration=0.5),
    )


# --- Basic export ---


def test_basic_export():
    """Exports responses to a Polars DataFrame."""
    responses = [
        _make_response("Alice", 90),
        _make_response("Bob", 80),
    ]
    df = export_dataframe(responses)
    assert isinstance(df, pl.DataFrame)
    assert df.shape == (2, 3)
    assert df.columns == ["response", "name", "score"]
    assert df["name"].to_list() == ["Alice", "Bob"]
    assert df["score"].to_list() == ["90", "80"]


# --- Explicit fields ---


def test_explicit_fields():
    """Only specified fields are included."""
    responses = [_make_response("Alice", 90)]
    df = export_dataframe(responses, fields=["name"])
    assert df.columns == ["response", "name"]
    assert "score" not in df.columns


# --- None model ---


def test_none_model():
    """None model produces null values."""
    responses = [_make_none_response()]
    df = export_dataframe(responses, fields=["name", "score"])
    assert df.shape == (1, 3)
    assert df["name"][0] is None
    assert df["score"][0] is None


# --- Empty responses ---


def test_empty_responses():
    """Empty list produces empty DataFrame."""
    df = export_dataframe([], fields=["name"])
    assert df.shape == (0, 2)
    assert df.columns == ["response", "name"]


# --- Mixed responses ---


def test_mixed_none_and_valid():
    """Mixed None and valid models produce correct output."""
    responses = [
        _make_response("Alice", 90),
        _make_none_response(),
        _make_response("Charlie", 70),
    ]
    df = export_dataframe(responses)
    assert df.shape == (3, 3)
    assert df["name"][0] == "Alice"
    assert df["name"][1] is None
    assert df["name"][2] == "Charlie"


# --- Response column ---


def test_response_column_present():
    """Response column contains raw LLM text."""
    responses = [_make_response("Alice", 90)]
    df = export_dataframe(responses)
    assert df["response"][0] == "raw"


# --- No fields detected ---


def test_no_fields_all_none():
    """All-None models with no fields gives response-only."""
    responses = [_make_none_response()]
    df = export_dataframe(responses)
    assert df.columns == ["response"]
    assert df.shape == (1, 1)
