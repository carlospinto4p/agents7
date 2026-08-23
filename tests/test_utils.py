"""Unit tests for agents7.agents.utils.

Covers run_with_semaphore(), resolve_strenum(), and
model_class_to_string().
"""

import asyncio
from datetime import date
from enum import StrEnum

import pytest
from pydantic import BaseModel, Field


from agents7.agents.utils import (
    format_example_value,
    get_examples_str,
    import_from_path,
    model_class_to_string,
    resolve_strenum,
    run_with_semaphore,
)
from agents7.exceptions import AgentConfigurationError


# ── import_from_path ──────────────────────────────────────────────


def test_import_from_path_valid():
    """Valid dotted path imports the named class."""
    cls = import_from_path("agents7.exceptions.AgentConfigurationError")
    assert cls is AgentConfigurationError


def test_import_from_path_unsafe_slash_raises():
    """Path with a slash raises AgentConfigurationError."""
    with pytest.raises(AgentConfigurationError, match="Unsafe module path"):
        import_from_path("os/path.join")


def test_import_from_path_unsafe_space_raises():
    """Path with a space raises AgentConfigurationError."""
    with pytest.raises(AgentConfigurationError, match="Unsafe module path"):
        import_from_path("os.path; import sys")


def test_import_from_path_unsafe_leading_dot_raises():
    """Path starting with a dot raises AgentConfigurationError."""
    with pytest.raises(AgentConfigurationError, match="Unsafe module path"):
        import_from_path(".hidden.Class")


# ── run_with_semaphore ────────────────────────────────────────────


async def _coro(value: int) -> int:
    return value


@pytest.mark.asyncio
async def test_run_with_semaphore_none_returns_result():
    """Without a semaphore, coroutine result is returned directly."""
    result = await run_with_semaphore(_coro(42), sem=None)
    assert result == 42


@pytest.mark.asyncio
async def test_run_with_semaphore_limits_concurrency():
    """Semaphore limits concurrent execution to its count."""
    sem = asyncio.Semaphore(1)
    results = await asyncio.gather(
        run_with_semaphore(_coro(1), sem),
        run_with_semaphore(_coro(2), sem),
        run_with_semaphore(_coro(3), sem),
    )
    assert sorted(results) == [1, 2, 3]


@pytest.mark.asyncio
async def test_run_with_semaphore_propagates_exception():
    """Exception from the coroutine propagates through."""

    async def _fail():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await run_with_semaphore(_fail(), sem=None)


# ── resolve_strenum ───────────────────────────────────────────────


class Color(StrEnum):
    RED = "red"
    BLUE = "blue"


def test_resolve_strenum_plain():
    """Plain StrEnum annotation returns the StrEnum class."""
    assert resolve_strenum(Color) is Color


def test_resolve_strenum_optional():
    """Optional[StrEnum] (i.e. StrEnum | None) returns the StrEnum."""
    annotation = Color | None
    assert resolve_strenum(annotation) is Color


def test_resolve_strenum_non_enum_returns_none():
    """Non-StrEnum annotation returns None."""
    assert resolve_strenum(str) is None


def test_resolve_strenum_int_returns_none():
    """int annotation returns None."""
    assert resolve_strenum(int) is None


def test_resolve_strenum_optional_str_returns_none():
    """Optional[str] returns None — no StrEnum inside."""
    assert resolve_strenum(str | None) is None


# ── model_class_to_string ─────────────────────────────────────────


class _Simple(BaseModel):
    name: str
    score: float


class _WithDefault(BaseModel):
    label: str = "unknown"


class _WithDescription(BaseModel):
    text: str = Field(description="Input text.")


class _Category(StrEnum):
    A = "Option A."
    B = "Option B."


class _WithEnum(BaseModel):
    category: _Category


def test_model_class_to_string_header():
    """Output starts with 'class <Name>:'."""
    result = model_class_to_string(_Simple)
    assert result.startswith("class _Simple:")


def test_model_class_to_string_plain_fields():
    """Plain fields appear as '    <name>: <type>'."""
    result = model_class_to_string(_Simple)
    assert "    name: str" in result
    assert "    score: float" in result


def test_model_class_to_string_default_value():
    """Fields with defaults include ' = <repr>'."""
    result = model_class_to_string(_WithDefault)
    assert "label: str = 'unknown'" in result


def test_model_class_to_string_description():
    """Fields with descriptions include 'Description:' line."""
    result = model_class_to_string(_WithDescription)
    assert "Description: Input text." in result


def test_model_class_to_string_enum_possible_values():
    """Enum fields list possible values."""
    result = model_class_to_string(_WithEnum)
    assert "Possible values:" in result
    assert "A: Option A." in result
    assert "B: Option B." in result


# ── get_examples_str ──────────────────────────────────────────────


def test_get_examples_str_valid_single():
    """Single valid example is formatted correctly."""
    result = get_examples_str([{"input": "hello", "output": "world"}])
    assert "## Examples:" in result
    assert "- Input: hello" in result
    assert "- Output: world" in result


def test_get_examples_str_valid_multiple():
    """Multiple examples are all included in the output."""
    result = get_examples_str(
        [{"input": "a", "output": "1"}, {"input": "b", "output": "2"}]
    )
    assert "- Input: a" in result
    assert "- Output: 1" in result
    assert "- Input: b" in result
    assert "- Output: 2" in result


def test_get_examples_str_empty_list():
    """Empty list returns just the header."""
    result = get_examples_str([])
    assert result == "## Examples:"


def test_get_examples_str_not_a_list_raises():
    """Non-list input raises AgentConfigurationError."""
    with pytest.raises(AgentConfigurationError, match="must be a list"):
        get_examples_str({"input": "x", "output": "y"})  # type: ignore[arg-type]


def test_get_examples_str_example_not_dict_raises():
    """Example entry that is not a dict raises AgentConfigurationError."""
    with pytest.raises(AgentConfigurationError, match="must be a dictionary"):
        get_examples_str(["not a dict"])


def test_get_examples_str_missing_output_key_raises():
    """Example missing 'output' key raises AgentConfigurationError."""
    with pytest.raises(AgentConfigurationError, match="input.*output"):
        get_examples_str([{"input": "x"}])


def test_get_examples_str_missing_input_key_raises():
    """Example missing 'input' key raises AgentConfigurationError."""
    with pytest.raises(AgentConfigurationError, match="input.*output"):
        get_examples_str([{"output": "x"}])


def test_get_examples_str_extra_key_raises():
    """Example with extra key beyond 'input'/'output' raises error."""
    with pytest.raises(AgentConfigurationError, match="input.*output"):
        get_examples_str([{"input": "x", "output": "y", "extra": "z"}])


def test_get_examples_str_dict_output_is_json():
    """Dict outputs render as JSON, never as a Python repr."""
    result = get_examples_str(
        [{"input": "x", "output": {"name": None, "age": 35}}]
    )
    assert '- Output: {"name": null, "age": 35}' in result
    assert "None" not in result
    assert "'name'" not in result


def test_get_examples_str_dict_input_is_json():
    """Dict inputs render as JSON too."""
    result = get_examples_str([{"input": {"text": "hi"}, "output": "label"}])
    assert '- Input: {"text": "hi"}' in result


def test_get_examples_str_list_output_is_json():
    """List outputs render as JSON."""
    result = get_examples_str([{"input": "x", "output": [1, None]}])
    assert "- Output: [1, null]" in result


def test_get_examples_str_scalar_output_unchanged():
    """Scalar outputs are rendered as-is, without JSON quoting."""
    result = get_examples_str([{"input": "x", "output": "TOO_COLD"}])
    assert "- Output: TOO_COLD" in result


# ── format_example_value ──────────────────────────────────────────


def test_format_example_value_nested_dict():
    """Nested structures serialize as JSON."""
    value = {"a": {"b": [None, True]}}
    assert format_example_value(value) == '{"a": {"b": [null, true]}}'


def test_format_example_value_non_serializable_falls_back_to_str():
    """Values JSON cannot encode fall back to their string form."""
    result = format_example_value({"when": date(2026, 2, 1)})
    assert result == '{"when": "2026-02-01"}'
