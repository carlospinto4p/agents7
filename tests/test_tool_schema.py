"""Unit tests for agents7.tools.schema.schema_from_callable."""

import pytest

from agents7.exceptions import ToolSchemaError
from agents7.tools.schema import schema_from_callable


def test_schema_basic_types():
    def get_weather(city: str, days: int = 1) -> str:
        return ""

    schema = schema_from_callable(get_weather)
    assert schema["type"] == "object"
    assert schema["properties"]["city"]["type"] == "string"
    assert schema["properties"]["days"]["type"] == "integer"
    # `city` is required (no default), `days` is not.
    assert schema["required"] == ["city"]


def test_schema_missing_hint_raises():
    def bad(city) -> str:
        return ""

    with pytest.raises(ToolSchemaError, match="missing a type hint"):
        schema_from_callable(bad)


def test_schema_var_args_raises():
    def bad(*args: int) -> str:
        return ""

    with pytest.raises(ToolSchemaError, match="unsupported"):
        schema_from_callable(bad)
