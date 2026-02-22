"""Unit tests for the Labeler agent (no LLM required)."""

import asyncio
import json
from enum import StrEnum
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from agents7.agents.labeler import Labeler
from agents7.agents.parsing import ResponseParser
from agents7.exceptions import AgentConfigurationError
from agents7.llm import LLMEngine
from agents7.settings import BatchSettings

from helpers import (
    DUMMY_METRICS,
    DummyOutput,
    make_labeler,
)


# --- Helper models ---


class DummyReason(StrEnum):
    GOOD = "Good result."
    BAD = "Bad result."


class DummyOutputWithEnum(BaseModel):
    label: str
    reason: DummyReason


# --- TestInitialization ---


def test_valid_output_model():
    """Valid BaseModel subclass is stored correctly."""
    agent = make_labeler()
    assert agent.output_model is DummyOutput
    assert agent.name == "test"
    assert agent.version == 1


def test_non_basemodel_raises_type_error():
    """Non-BaseModel output_model raises AgentConfigurationError."""
    with pytest.raises(AgentConfigurationError, match="output_model"):
        make_labeler(output_model=dict)


# --- TestYAMLLoading ---


def test_valid_config(tmp_path):
    """Valid YAML config produces correct Labeler attributes."""
    yaml_content = (
        "name: MyLabeler\n"
        "version: 2\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: |\n"
        "  Analyze this text.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
    )
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content)

    agent = Labeler.from_yaml(str(yaml_file))
    assert agent.name == "MyLabeler"
    assert agent.version == 2
    assert "Analyze this text." in agent.system_prompt


def test_config_with_examples(tmp_path):
    """Config with system_examples populates agent.examples."""
    yaml_content = (
        "name: ExLabeler\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: |\n"
        "  Extract info.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
        "system_examples:\n"
        "    - input: Hello\n"
        "      output:\n"
        "        name: World\n"
    )
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content)

    agent = Labeler.from_yaml(str(yaml_file))
    assert agent.examples is not None
    assert len(agent.examples) == 1
    assert agent.examples[0]["input"] == "Hello"


def test_missing_name_raises_value_error(tmp_path):
    """Missing mandatory key 'name' raises AgentConfigurationError."""
    yaml_content = (
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Test.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
    )
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content)

    with pytest.raises(AgentConfigurationError, match="name"):
        Labeler.from_yaml(str(yaml_file))


def test_missing_engine_model_raises_value_error(tmp_path):
    """Missing engine key 'model' raises AgentConfigurationError."""
    yaml_content = (
        "name: Test\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "system_prompt: Test.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
    )
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content)

    with pytest.raises(AgentConfigurationError, match="model"):
        Labeler.from_yaml(str(yaml_file))


def test_invalid_examples_format_raises_value_error(tmp_path):
    """Non-list system_examples raises AgentConfigurationError."""
    yaml_content = (
        "name: Test\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Test.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
        "system_examples: not a list\n"
    )
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content)

    with pytest.raises(AgentConfigurationError, match="system_examples"):
        Labeler.from_yaml(str(yaml_file))


# --- TestPromptGeneration ---


def test_prompt_contains_system_prompt_and_schema():
    """Prompt contains system_prompt text and Output Model Schema."""
    agent = make_labeler(system_prompt="Analyze this data.")
    assert "Analyze this data." in agent.prompt
    assert "Output Model Schema" in agent.prompt


def test_prompt_includes_examples():
    """Prompt includes examples section when examples provided."""
    examples = [
        {"input": "foo", "output": "bar"},
    ]
    agent = make_labeler(examples=examples)
    assert "Examples" in agent.prompt
    assert "foo" in agent.prompt
    assert "bar" in agent.prompt


def test_prompt_no_examples_section():
    """Prompt has no Examples section when examples is None."""
    agent = make_labeler(examples=None)
    assert "Examples" not in agent.prompt


def test_prompt_for_batch_contains_batch_instruction():
    """Batch prompt contains batch processing instructions."""
    agent = make_labeler()
    batch = agent.prompt_for_batch
    assert "multiple inputs" in batch
    assert "array of outputs" in batch


def test_prompt_for_batch_includes_examples():
    """Batch prompt includes examples when provided."""
    examples = [
        {"input": "x", "output": "y"},
    ]
    agent = make_labeler(examples=examples)
    batch = agent.prompt_for_batch
    assert "Examples" in batch
    assert "x" in batch


# --- TestOutputModelList ---


def test_output_model_list_structure():
    """output_model_list is a BaseModel with items field."""
    agent = make_labeler()
    list_model = agent.output_model_list
    assert issubclass(list_model, BaseModel)
    assert list_model.__name__ == "DummyOutputList"
    assert "items" in list_model.model_fields


def test_output_model_list_validates():
    """Instantiating output_model_list with valid data works."""
    agent = make_labeler()
    list_model = agent.output_model_list
    instance = list_model(
        items=[DummyOutput(value="a"), DummyOutput(value="b")]
    )
    assert len(instance.items) == 2
    assert instance.items[0].value == "a"


# --- TestFixMalformedJson ---


def test_strips_markdown_code_block():
    """Strips ```json ... ``` markdown blocks."""
    raw = '```json\n{"value": "ok"}\n```'
    assert ResponseParser._fix_malformed_json(raw) == '{"value": "ok"}'


def test_fixes_double_brace():
    """Fixes {\"{ double-brace malformation."""
    raw = '{"{"value": "ok"}'
    result = ResponseParser._fix_malformed_json(raw)
    assert result == '{"value": "ok"}'


def test_fixes_double_bracket():
    """Fixes [\"[ double-bracket malformation."""
    raw = '["[{"value": "ok"}]'
    result = ResponseParser._fix_malformed_json(raw)
    assert result == '[{"value": "ok"}]'


def test_clean_input_unchanged():
    """Clean JSON input is returned unchanged."""
    clean = '{"value": "ok"}'
    assert ResponseParser._fix_malformed_json(clean) == clean


# --- TestParseModelResponse ---


def test_valid_json_returns_model():
    """Valid JSON string is parsed into DummyOutput instance."""
    agent = make_labeler()
    result = agent._parser.parse('{"value": "hello"}')
    assert isinstance(result, DummyOutput)
    assert result.value == "hello"


def test_invalid_json_returns_none():
    """Invalid JSON returns None."""
    agent = make_labeler()
    result = agent._parser.parse("not json at all")
    assert result is None


def test_batch_parse():
    """Batch mode parses list wrapper correctly."""
    agent = make_labeler()
    payload = json.dumps({"items": [{"value": "a"}, {"value": "b"}]})
    result = agent._parser.parse(payload, for_batch=True)
    assert result is not None
    assert len(result.items) == 2
    assert result.items[0].value == "a"


# --- TestEnumMapping ---


def test_get_enum_mappings_no_enums():
    """Returns empty dict for model without StrEnum fields."""
    agent = make_labeler(output_model=DummyOutput)
    assert agent._parser._get_enum_mappings() == {}


def test_get_enum_mappings_with_enum():
    """Returns correct mapping for DummyOutputWithEnum."""
    agent = make_labeler(output_model=DummyOutputWithEnum)
    mappings = agent._parser._get_enum_mappings()
    assert "reason" in mappings
    assert mappings["reason"]["GOOD"] == "Good result."
    assert mappings["reason"]["BAD"] == "Bad result."


def test_convert_enum_names_to_values():
    """Converts enum name to value in JSON string."""
    agent = make_labeler(output_model=DummyOutputWithEnum)
    raw = json.dumps({"label": "test", "reason": "GOOD"})
    converted = agent._parser._convert_enum_names_to_values(raw)
    data = json.loads(converted)
    assert data["reason"] == "Good result."


def test_convert_enum_names_invalid_json():
    """Invalid JSON is returned unchanged."""
    agent = make_labeler(output_model=DummyOutputWithEnum)
    raw = "not json"
    assert agent._parser._convert_enum_names_to_values(raw) == raw


def test_apply_enum_mappings_list():
    """Handles list of dicts correctly."""
    mappings = {"reason": {"GOOD": "Good result."}}
    data = [
        {"label": "a", "reason": "GOOD"},
        {"label": "b", "reason": "other"},
    ]
    result = ResponseParser._apply_enum_mappings(data, mappings)
    assert result[0]["reason"] == "Good result."
    assert result[1]["reason"] == "other"


def test_apply_enum_mappings_non_dict():
    """Non-dict input is returned unchanged."""
    mappings = {"reason": {"GOOD": "Good result."}}
    assert ResponseParser._apply_enum_mappings("string", mappings) == "string"


# --- TestBatchAsyncConcurrency ---


def _batch_chat_response(items: list[dict]) -> dict:
    """Build a mock chat_async return value for batch mode."""
    return {
        "role": "assistant",
        "response": json.dumps({"items": items}),
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_max_concurrent_none_unlimited(mock_chat):
    """All chunk tasks fire at once when max_concurrent=None."""
    peak = 0
    current = 0
    lock = asyncio.Lock()

    original_return = _batch_chat_response([{"value": "x"}])

    async def _side_effect(**kwargs):
        nonlocal peak, current
        async with lock:
            current += 1
            peak = max(peak, current)
        await asyncio.sleep(0.05)
        async with lock:
            current -= 1
        return dict(original_return)

    mock_chat.side_effect = _side_effect

    agent = make_labeler()
    # 3 items with requests_per_prompt=1 → 3 chunks
    await agent.act_in_batch_async(
        [{"a": 1}, {"a": 2}, {"a": 3}],
        requests_per_prompt=1,
        with_tqdm=False,
        batch_settings=BatchSettings(max_concurrent=None),
    )
    assert peak == 3


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_max_concurrent_limits_parallelism(mock_chat):
    """Semaphore caps concurrent requests to max_concurrent."""
    peak = 0
    current = 0
    lock = asyncio.Lock()

    original_return = _batch_chat_response([{"value": "x"}])

    async def _side_effect(**kwargs):
        nonlocal peak, current
        async with lock:
            current += 1
            peak = max(peak, current)
        await asyncio.sleep(0.05)
        async with lock:
            current -= 1
        return dict(original_return)

    mock_chat.side_effect = _side_effect

    agent = make_labeler()
    await agent.act_in_batch_async(
        [{"a": i} for i in range(5)],
        requests_per_prompt=1,
        with_tqdm=False,
        batch_settings=BatchSettings(max_concurrent=2),
    )
    assert peak <= 2


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_max_concurrent_one_is_sequential(mock_chat):
    """max_concurrent=1 runs one request at a time."""
    peak = 0
    current = 0
    lock = asyncio.Lock()

    original_return = _batch_chat_response([{"value": "x"}])

    async def _side_effect(**kwargs):
        nonlocal peak, current
        async with lock:
            current += 1
            peak = max(peak, current)
        await asyncio.sleep(0.02)
        async with lock:
            current -= 1
        return dict(original_return)

    mock_chat.side_effect = _side_effect

    agent = make_labeler()
    await agent.act_in_batch_async(
        [{"a": i} for i in range(4)],
        requests_per_prompt=1,
        with_tqdm=False,
        batch_settings=BatchSettings(max_concurrent=1),
    )
    assert peak == 1
