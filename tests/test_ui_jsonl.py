"""Unit tests for JSONL parsing (agents7.ui._pages.parse_jsonl)."""

from agents7.ui._pages import parse_jsonl


# --- Valid JSONL ---


def test_valid_jsonl_single_line():
    text = '{"input": "hello", "category": "POSITIVE"}'
    result = parse_jsonl(text)
    assert result == [{"input": "hello", "category": "POSITIVE"}]


def test_valid_jsonl_multiple_lines():
    text = (
        '{"input": "a", "label": "X"}\n'
        '{"input": "b", "label": "Y"}\n'
        '{"input": "c", "label": "Z"}\n'
    )
    result = parse_jsonl(text)
    assert len(result) == 3
    assert result[0] == {"input": "a", "label": "X"}
    assert result[2] == {"input": "c", "label": "Z"}


# --- Empty / blank input ---


def test_empty_string():
    assert parse_jsonl("") == []


def test_only_whitespace():
    assert parse_jsonl("   \n  \n  ") == []


def test_blank_lines_skipped():
    text = '{"a": 1}\n\n\n{"b": 2}\n'
    result = parse_jsonl(text)
    assert len(result) == 2


# --- Malformed lines ---


def test_malformed_line_wrapped_as_text():
    text = "not json at all"
    result = parse_jsonl(text)
    assert result == [{"text": "not json at all"}]


def test_mixed_valid_and_malformed():
    text = '{"input": "ok"}\nbroken line\n{"input": "also ok"}'
    result = parse_jsonl(text)
    assert len(result) == 3
    assert result[0] == {"input": "ok"}
    assert result[1] == {"text": "broken line"}
    assert result[2] == {"input": "also ok"}


# --- Non-object JSON values ---


def test_json_string_wrapped():
    text = '"just a string"'
    result = parse_jsonl(text)
    assert result == [{"text": "just a string"}]


def test_json_number_wrapped():
    text = "42"
    result = parse_jsonl(text)
    assert result == [{"text": "42"}]


def test_json_array_wrapped():
    text = "[1, 2, 3]"
    result = parse_jsonl(text)
    assert result == [{"text": "[1, 2, 3]"}]


# --- Edge cases ---


def test_lines_with_leading_trailing_whitespace():
    text = '  {"key": "val"}  \n  {"key": "val2"}  '
    result = parse_jsonl(text)
    assert len(result) == 2
    assert result[0] == {"key": "val"}
    assert result[1] == {"key": "val2"}


def test_nested_objects():
    text = '{"input": "x", "meta": {"id": 1, "tags": ["a"]}}'
    result = parse_jsonl(text)
    assert len(result) == 1
    assert result[0]["meta"] == {"id": 1, "tags": ["a"]}
