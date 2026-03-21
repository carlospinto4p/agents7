"""Unit tests for the suggestion queue module."""

import pytest

from agents7.exceptions import SuggestionError
from agents7.suggestions import Suggestion, SuggestionQueue


# --- Suggestion dataclass ---


def test_suggestion_to_dict():
    """Suggestion serializes to dict via to_dict()."""
    s = Suggestion(
        agent="test_agent",
        input_data={"text": "hello"},
        expected={"category": "A"},
        note="a note",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    d = s.to_dict()
    assert d["agent"] == "test_agent"
    assert d["input_data"] == {"text": "hello"}
    assert d["expected"] == {"category": "A"}
    assert d["note"] == "a note"


def test_suggestion_is_frozen():
    """Cannot modify fields after creation."""
    s = Suggestion(
        agent="a",
        input_data={"x": 1},
        expected={},
        note="",
        timestamp="",
    )
    with pytest.raises(AttributeError):
        s.agent = "b"


# --- SuggestionQueue initialization ---


def test_default_path():
    """Default path is under ~/.agents_catalog/."""
    q = SuggestionQueue("my_agent")
    assert q.path.name == "my_agent.jsonl"
    assert "suggestions" in str(q.path)


def test_custom_path(tmp_path):
    """Custom path is used when provided."""
    p = tmp_path / "custom.jsonl"
    q = SuggestionQueue("my_agent", path=p)
    assert q.path == p


# --- add() ---


def test_add_creates_file(tmp_path):
    """First add() creates the JSONL file."""
    p = tmp_path / "q.jsonl"
    q = SuggestionQueue("a", path=p)
    q.add({"text": "x"}, {"category": "A"})
    assert p.exists()


def test_add_appends(tmp_path):
    """Multiple add() calls append lines."""
    p = tmp_path / "q.jsonl"
    q = SuggestionQueue("a", path=p)
    q.add({"text": "x"}, {"category": "A"})
    q.add({"text": "y"}, {"category": "B"})
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_add_auto_timestamp(tmp_path):
    """Timestamp is auto-generated in ISO format."""
    p = tmp_path / "q.jsonl"
    q = SuggestionQueue("a", path=p)
    q.add({"text": "x"}, {"category": "A"})
    items = q.list()
    assert "T" in items[0].timestamp
    assert "+" in items[0].timestamp or "Z" in items[0].timestamp


def test_add_empty_input_data_raises(tmp_path):
    """Empty input_data raises SuggestionError."""
    p = tmp_path / "q.jsonl"
    q = SuggestionQueue("a", path=p)
    with pytest.raises(SuggestionError):
        q.add({}, {"category": "A"})


def test_add_creates_parent_dirs(tmp_path):
    """Creates nested directories if needed."""
    p = tmp_path / "deep" / "nested" / "q.jsonl"
    q = SuggestionQueue("a", path=p)
    q.add({"text": "x"}, {"category": "A"})
    assert p.exists()


def test_add_with_note(tmp_path):
    """Note field is stored correctly."""
    p = tmp_path / "q.jsonl"
    q = SuggestionQueue("a", path=p)
    q.add({"text": "x"}, {"category": "A"}, note="reason")
    items = q.list()
    assert items[0].note == "reason"


# --- list() ---


def test_list_returns_suggestions(tmp_path):
    """Returns all suggestions in order."""
    p = tmp_path / "q.jsonl"
    q = SuggestionQueue("a", path=p)
    q.add({"text": "x"}, {"category": "A"})
    q.add({"text": "y"}, {"category": "B"})
    items = q.list()
    assert len(items) == 2
    assert items[0].input_data == {"text": "x"}
    assert items[1].input_data == {"text": "y"}


def test_list_empty_when_no_file(tmp_path):
    """Returns empty list when file doesn't exist."""
    p = tmp_path / "missing.jsonl"
    q = SuggestionQueue("a", path=p)
    assert q.list() == []


# --- count() ---


def test_count_matches_list_length(tmp_path):
    """count() matches len(list())."""
    p = tmp_path / "q.jsonl"
    q = SuggestionQueue("a", path=p)
    q.add({"text": "x"}, {"category": "A"})
    q.add({"text": "y"}, {"category": "B"})
    assert q.count() == 2
    assert q.count() == len(q.list())


def test_count_zero_when_no_file(tmp_path):
    """Returns 0 when file doesn't exist."""
    p = tmp_path / "missing.jsonl"
    q = SuggestionQueue("a", path=p)
    assert q.count() == 0


# --- clear() ---


def test_clear_removes_file(tmp_path):
    """File is deleted after clear()."""
    p = tmp_path / "q.jsonl"
    q = SuggestionQueue("a", path=p)
    q.add({"text": "x"}, {"category": "A"})
    assert p.exists()
    q.clear()
    assert not p.exists()


def test_clear_no_error_when_no_file(tmp_path):
    """No exception when file doesn't exist."""
    p = tmp_path / "missing.jsonl"
    q = SuggestionQueue("a", path=p)
    q.clear()


# --- remove() ---


def test_remove_by_index(tmp_path):
    """Removes the correct suggestion."""
    p = tmp_path / "q.jsonl"
    q = SuggestionQueue("a", path=p)
    q.add({"text": "x"}, {"category": "A"})
    q.add({"text": "y"}, {"category": "B"})
    q.add({"text": "z"}, {"category": "C"})
    q.remove(1)
    items = q.list()
    assert len(items) == 2
    assert items[0].input_data == {"text": "x"}
    assert items[1].input_data == {"text": "z"}


def test_remove_returns_removed(tmp_path):
    """Returns the removed Suggestion."""
    p = tmp_path / "q.jsonl"
    q = SuggestionQueue("a", path=p)
    q.add({"text": "x"}, {"category": "A"})
    removed = q.remove(0)
    assert isinstance(removed, Suggestion)
    assert removed.input_data == {"text": "x"}


def test_remove_last_deletes_file(tmp_path):
    """Removing the last suggestion deletes the file."""
    p = tmp_path / "q.jsonl"
    q = SuggestionQueue("a", path=p)
    q.add({"text": "x"}, {"category": "A"})
    q.remove(0)
    assert not p.exists()


def test_remove_invalid_index_raises(tmp_path):
    """Out-of-range index raises SuggestionError."""
    p = tmp_path / "q.jsonl"
    q = SuggestionQueue("a", path=p)
    q.add({"text": "x"}, {"category": "A"})
    with pytest.raises(SuggestionError):
        q.remove(5)
    with pytest.raises(SuggestionError):
        q.remove(-1)
