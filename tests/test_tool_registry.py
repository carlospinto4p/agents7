"""Unit tests for agents7.tools Tool and ToolRegistry."""

import pytest

from agents7.exceptions import ToolSchemaError
from agents7.tools.registry import ToolRegistry
from agents7.tools.tool import Tool


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def boom(x: int) -> int:
    """Always fails."""
    raise RuntimeError("kaboom")


def test_tool_defaults_from_callable():
    tool = Tool(add)
    assert tool.name == "add"
    assert tool.description == "Add two numbers."
    assert tool.json_schema["properties"]["a"]["type"] == "integer"


def test_tool_run_coerces_to_str():
    assert Tool(add).run(a=2, b=3) == "5"


def test_registry_schemas_shape():
    reg = ToolRegistry([add])
    schema = reg.schemas()[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "add"
    assert "parameters" in schema["function"]


def test_registry_execute_happy():
    assert ToolRegistry([add]).execute("add", {"a": 4, "b": 1}) == "5"


def test_registry_execute_tool_error_returned_as_string():
    out = ToolRegistry([boom]).execute("boom", {"x": 1})
    assert out.startswith("Error:")
    assert "kaboom" in out


def test_registry_execute_unknown_tool_raises():
    with pytest.raises(ToolSchemaError, match="Unknown tool"):
        ToolRegistry([add]).execute("nope", {})
