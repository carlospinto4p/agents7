"""Unit tests for agents7.tools Tool and ToolRegistry."""

import json

import pytest
from pydantic import BaseModel

from agents7.exceptions import ToolArgumentError, ToolSchemaError
from agents7.tools.registry import ToolRegistry
from agents7.tools.tool import Tool, serialize_tool_result


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class Product(BaseModel):
    """A Pydantic element of a container return."""

    sku: str
    price_cents: int


def find_products(query: str) -> list[Product]:
    """Return a bare list of Pydantic models."""
    return [Product(sku=f"{query}-01", price_cents=1250)]


def boom(x: int) -> int:
    """Always fails."""
    raise RuntimeError("kaboom")


def lookup(city: str) -> dict:
    """Return a structured weather record."""
    return {"city": city, "temp_c": 21, "unit": "°C"}


class Weather(BaseModel):
    """A Pydantic tool return."""

    city: str
    temp_c: int


def test_tool_defaults_from_callable():
    tool = Tool(add)
    assert tool.name == "add"
    assert tool.description == "Add two numbers."
    assert tool.json_schema["properties"]["a"]["type"] == "integer"


def test_tool_run_coerces_to_str():
    assert Tool(add).run(a=2, b=3) == "5"


def test_tool_run_serializes_dict_as_json():
    out = Tool(lookup).run(city="Lisbon")
    assert json.loads(out) == {
        "city": "Lisbon",
        "temp_c": 21,
        "unit": "°C",
    }


def test_tool_run_serializes_pydantic_as_json():
    out = Tool(lambda: Weather(city="Porto", temp_c=18)).run()
    assert json.loads(out) == {"city": "Porto", "temp_c": 18}


def test_serialize_tool_result_passthrough_and_fallback():
    assert serialize_tool_result("hi") == "hi"
    assert serialize_tool_result(5) == "5"
    assert serialize_tool_result(None) == "None"
    assert json.loads(serialize_tool_result([1, 2])) == [1, 2]
    assert json.loads(serialize_tool_result((1, 2))) == [1, 2]


def test_serialize_recurses_into_list_of_models():
    out = Tool(find_products).run(query="mug")
    assert json.loads(out) == [{"sku": "mug-01", "price_cents": 1250}]


def test_serialize_recurses_into_dict_of_models():
    value = {"items": [Product(sku="A", price_cents=1)]}
    assert json.loads(serialize_tool_result(value)) == {
        "items": [{"sku": "A", "price_cents": 1}]
    }


def test_run_coerces_stringified_numbers():
    # llama3.1:8b sends {"a": "4", "b": "1"} for two int params.
    assert Tool(add).run(a="4", b="1") == "5"


def test_run_rejects_schema_echo_envelope_with_usable_message():
    # qwen2.5:3b wraps the value in an envelope echoing the schema.
    with pytest.raises(ToolArgumentError) as exc:
        Tool(find_products).run(query={"type": "string", "value": "blue mug"})
    assert "find_products" in str(exc.value)
    assert "query" in str(exc.value)


def test_run_reports_missing_argument():
    with pytest.raises(ToolArgumentError, match="b"):
        Tool(add).run(a=1)


def test_tool_without_hints_needs_explicit_schema():
    with pytest.raises(ToolSchemaError):
        Tool(lambda x: x, name="untyped")


def test_tool_with_explicit_schema_skips_validation():
    tool = Tool(
        lambda x: x,
        name="untyped",
        json_schema={"type": "object", "properties": {}},
    )
    assert tool.args_model is None
    assert tool.run(x="raw") == "raw"


def test_registry_execute_bad_arguments_returned_as_string():
    out = ToolRegistry([add]).execute("add", {"a": "x", "b": 1})
    assert out.startswith("Error: Invalid arguments for `add`")


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
