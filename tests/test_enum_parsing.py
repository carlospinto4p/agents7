"""Unit tests for enum conversion edge cases (no LLM required)."""

import json
from enum import StrEnum

import pytest
from pydantic import BaseModel

from agents7.agents.parsing import ResponseParser
from agents7.exceptions import ParseError

from helpers import make_labeler


# --- Test models ---


class Color(StrEnum):
    RED = "Red color."
    BLUE = "Blue color."


class Size(StrEnum):
    SMALL = "Small size."
    LARGE = "Large size."


class MultiEnumOutput(BaseModel):
    color: Color
    size: Size


class OptionalEnumOutput(BaseModel):
    label: str
    color: Color | None = None


class PlainOutput(BaseModel):
    label: str
    score: float


# --- _get_enum_mappings ---


def test_multiple_enum_fields():
    """MultiEnumOutput → mappings for both color and size."""
    agent = make_labeler(output_model=MultiEnumOutput)
    mappings = agent._parser._get_enum_mappings()

    assert "color" in mappings
    assert "size" in mappings
    assert mappings["color"]["RED"] == "Red color."
    assert mappings["size"]["LARGE"] == "Large size."


def test_optional_enum_field():
    """Optional enum field is detected via __origin__/__args__."""
    agent = make_labeler(output_model=OptionalEnumOutput)
    mappings = agent._parser._get_enum_mappings()

    assert "color" in mappings
    assert mappings["color"]["BLUE"] == "Blue color."


def test_mixed_enum_and_plain_fields():
    """Only enum fields appear in mappings; plain fields ignored."""
    agent = make_labeler(output_model=OptionalEnumOutput)
    mappings = agent._parser._get_enum_mappings()

    assert "label" not in mappings
    assert "color" in mappings


# --- _convert_enum_names_to_values (parametrized) ---


@pytest.fixture()
def multi_parser():
    """ResponseParser with MultiEnumOutput model."""
    agent = make_labeler(output_model=MultiEnumOutput)
    return agent._parser


@pytest.mark.parametrize(
    "input_data, expected",
    [
        pytest.param(
            {"color": "RED", "size": "SMALL"},
            {"color": "Red color.", "size": "Small size."},
            id="converts_known_names",
        ),
        pytest.param(
            {"color": "UNKNOWN", "size": "SMALL"},
            {"color": "UNKNOWN", "size": "Small size."},
            id="unknown_name_unchanged",
        ),
        pytest.param(
            {"color": "Red color.", "size": "Small size."},
            {"color": "Red color.", "size": "Small size."},
            id="value_already_correct",
        ),
        pytest.param(
            {},
            {},
            id="empty_json_object",
        ),
    ],
)
def test_convert_enum_names(multi_parser, input_data, expected):
    """Parametrized: enum name conversion edge cases."""
    raw = json.dumps(input_data)
    converted = multi_parser._convert_enum_names_to_values(raw)
    assert json.loads(converted) == expected


def test_multiple_enum_fields_converted(multi_parser):
    """Both color and size names converted in one pass."""
    raw = json.dumps({"color": "BLUE", "size": "LARGE"})
    converted = multi_parser._convert_enum_names_to_values(raw)
    data = json.loads(converted)
    assert data["color"] == "Blue color."
    assert data["size"] == "Large size."


# --- _apply_enum_mappings ---


_MAPPINGS = {
    "color": {"RED": "Red color.", "BLUE": "Blue color."},
}


def test_nested_list_of_dicts():
    """List of dicts with mixed known/unknown names."""
    data = [
        {"color": "RED", "other": "x"},
        {"color": "UNKNOWN", "other": "y"},
    ]
    result = ResponseParser._apply_enum_mappings(data, _MAPPINGS)
    assert result[0]["color"] == "Red color."
    assert result[1]["color"] == "UNKNOWN"


def test_deeply_nested_list():
    """List of lists — inner non-dict items returned as-is."""
    data = [["RED", "BLUE"], "plain"]
    result = ResponseParser._apply_enum_mappings(data, _MAPPINGS)
    assert result == [["RED", "BLUE"], "plain"]


def test_empty_list():
    """Empty list → empty list."""
    result = ResponseParser._apply_enum_mappings([], _MAPPINGS)
    assert result == []


def test_empty_mappings():
    """Non-empty data + empty mappings → data unchanged."""
    data = {"color": "RED", "label": "test"}
    result = ResponseParser._apply_enum_mappings(data, {})
    assert result == {"color": "RED", "label": "test"}


def test_integer_value_in_data():
    """Numeric values are not enum names → unchanged."""
    data = {"color": 42, "label": "test"}
    result = ResponseParser._apply_enum_mappings(data, _MAPPINGS)
    assert result["color"] == 42


# --- parse() fallback path ---


def test_parse_fallback_converts_enum_names(multi_parser):
    """parse() falls back to enum-name conversion when direct parse fails."""
    raw = json.dumps({"color": "RED", "size": "SMALL"})
    result = multi_parser.parse(raw)
    assert result.color == Color.RED
    assert result.size == Size.SMALL


def test_parse_direct_path_succeeds(multi_parser):
    """parse() succeeds without fallback when values are already correct."""
    raw = json.dumps({"color": "Red color.", "size": "Small size."})
    result = multi_parser.parse(raw)
    assert result.color == Color.RED
    assert result.size == Size.SMALL


def test_parse_raises_when_fallback_also_fails(multi_parser):
    """parse() raises ParseError when both direct and fallback validation fail."""
    raw = json.dumps({"color": "INVALID", "size": "ALSO_INVALID"})
    with pytest.raises(ParseError):
        multi_parser.parse(raw)
