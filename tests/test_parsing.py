"""Unit tests for ResponseParser (no LLM required)."""

import pytest
from enum import StrEnum
from pydantic import BaseModel, create_model

from agents7.agents.parsing import ResponseParser
from agents7.exceptions import ParseError


class Output(BaseModel):
    value: str


OutputList = create_model("OutputList", items=(list[Output], ...))


def make_parser(**kwargs) -> ResponseParser:
    defaults = {
        "output_model": Output,
        "output_model_list": OutputList,
        "name": "test-agent",
        "validators": None,
    }
    defaults.update(kwargs)
    return ResponseParser(**defaults)


# ── parse() happy path ─────────────────────────────────────────────────────


def test_parse_valid_json():
    """parse() returns model for valid JSON."""
    result = make_parser().parse('{"value": "hello"}')
    assert result.value == "hello"


def test_parse_valid_batch():
    """parse(for_batch=True) returns list model."""
    result = make_parser().parse(
        '{"items": [{"value": "a"}, {"value": "b"}]}', for_batch=True
    )
    assert len(result.items) == 2
    assert result.items[0].value == "a"


# ── _fix_malformed_json ────────────────────────────────────────────────────


def test_fix_backtick_json_block():
    """Backtick-fenced JSON block is unwrapped."""
    fixed = ResponseParser._fix_malformed_json(
        '```json\n{"value": "ok"}\n```'
    )
    assert fixed == '{"value": "ok"}'


def test_fix_backtick_block_no_language():
    """Backtick block without language tag is unwrapped."""
    fixed = ResponseParser._fix_malformed_json('```\n{"value": "ok"}\n```')
    assert fixed == '{"value": "ok"}'


def test_fix_double_brace_prefix():
    """'{"{ ... ' prefix is corrected to '{ ...'."""
    fixed = ResponseParser._fix_malformed_json('{"{"value": "ok"}')
    assert fixed == '{"value": "ok"}'


def test_fix_array_double_prefix():
    """'["[ ... ' prefix is corrected to '[ ...'."""
    fixed = ResponseParser._fix_malformed_json('["[1, 2, 3]')
    assert fixed == "[1, 2, 3]"


def test_fix_clean_json_unchanged():
    """Well-formed JSON is returned unchanged."""
    raw = '{"value": "clean"}'
    assert ResponseParser._fix_malformed_json(raw) == raw


def test_fix_trailing_comma_passes_through():
    """Trailing commas are not fixed — passed through as-is."""
    raw = '{"value": "ok",}'
    assert ResponseParser._fix_malformed_json(raw) == raw


def test_fix_unclosed_bracket_passes_through():
    """Unclosed brackets are not fixed — passed through as-is."""
    raw = '{"value": "ok"'
    assert ResponseParser._fix_malformed_json(raw) == raw


def test_fix_missing_quotes_passes_through():
    """Unquoted string values are not fixed — passed through as-is."""
    raw = "{value: ok}"
    assert ResponseParser._fix_malformed_json(raw) == raw


def test_fix_whitespace_stripped():
    """Leading and trailing whitespace is stripped."""
    raw = '  {"value": "ok"}  '
    assert ResponseParser._fix_malformed_json(raw) == '{"value": "ok"}'


def test_parse_strips_backtick_block():
    """parse() handles backtick-wrapped JSON end-to-end."""
    result = make_parser().parse('```json\n{"value": "clean"}\n```')
    assert result.value == "clean"


# ── Enum name-to-value fallback ────────────────────────────────────────────


def test_parse_enum_name_fallback():
    """LLM returning enum name is converted to enum value."""

    class Sentiment(StrEnum):
        GOOD = "Good result."
        BAD = "Bad result."

    class SentimentOutput(BaseModel):
        reason: Sentiment

    SentimentList = create_model(
        "SentimentList", items=(list[SentimentOutput], ...)
    )
    parser = ResponseParser(
        output_model=SentimentOutput,
        output_model_list=SentimentList,
        name="test",
    )
    result = parser.parse('{"reason": "GOOD"}')
    assert result.reason == Sentiment.GOOD


# ── parse() failure ────────────────────────────────────────────────────────


def test_parse_invalid_json_raises_parse_error():
    """Completely invalid JSON raises ParseError with agent name."""
    with pytest.raises(ParseError, match="test-agent"):
        make_parser().parse("not json at all {{{{")


def test_parse_wrong_schema_raises_parse_error():
    """JSON that doesn't match schema raises ParseError."""
    with pytest.raises(ParseError):
        make_parser().parse('{"wrong_field": 123}')


# ── run_validators() ───────────────────────────────────────────────────────


def test_run_validators_none_model_passthrough():
    """None model passes through unchanged."""
    assert make_parser().run_validators(None) is None


def test_run_validators_no_validators_passthrough():
    """Model passes through unchanged when no validators."""
    parser = make_parser(validators=None)
    model = Output(value="hello")
    assert parser.run_validators(model) is model


def test_run_validators_transforms_model():
    """Single validator transforms the model."""

    def upper(m: BaseModel) -> BaseModel:
        return Output(value=m.value.upper())

    result = make_parser(validators=[upper]).run_validators(
        Output(value="hi")
    )
    assert result.value == "HI"


def test_run_validators_failure_returns_none():
    """Validator that raises returns None."""

    def reject(m: BaseModel) -> BaseModel:
        raise ValueError("rejected")

    assert (
        make_parser(validators=[reject]).run_validators(Output(value="x"))
        is None
    )


def test_run_validators_unexpected_error_propagates():
    """Non-validation errors (e.g. bugs) are not swallowed."""

    def boom(m: BaseModel) -> BaseModel:
        raise TypeError("unexpected")

    with pytest.raises(TypeError, match="unexpected"):
        make_parser(validators=[boom]).run_validators(Output(value="x"))


def test_run_validators_chain():
    """Validators run in order; each output feeds into next."""

    def upper(m: BaseModel) -> BaseModel:
        return Output(value=m.value.upper())

    def exclaim(m: BaseModel) -> BaseModel:
        return Output(value=m.value + "!")

    result = make_parser(validators=[upper, exclaim]).run_validators(
        Output(value="hi")
    )
    assert result.value == "HI!"


def test_run_validators_short_circuits():
    """First validator failure skips remaining validators."""
    called = []

    def reject(m: BaseModel) -> BaseModel:
        raise ValueError("stop")

    def second(m: BaseModel) -> BaseModel:
        called.append("second")
        return m

    assert (
        make_parser(validators=[reject, second]).run_validators(
            Output(value="x")
        )
        is None
    )
    assert "second" not in called
