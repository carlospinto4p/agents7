"""Unit tests for Pipeline agent chaining (no LLM required)."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from agents7.agents.labeler import Labeler
from agents7.agents.pipeline import (
    Pipeline,
    PipelineStep,
    _default_mapper,
)
from agents7.exceptions import PipelineConfigurationError
from agents7.llm import LLMEngine
from agents7.models import AgentResponse
from agents7.settings import BatchSettings

from helpers import DUMMY_METRICS, ENGINE


# --- Helper models & fixtures ---


class StepOneOutput(BaseModel):
    name: str


class StepTwoOutput(BaseModel):
    greeting: str


def _make_labeler(output_model: type[BaseModel] = StepOneOutput):
    return Labeler(
        name="test",
        version=1,
        engine=ENGINE,
        system_prompt="Test prompt.",
        output_model=output_model,
    )


def _make_response(model: BaseModel | None, idx: int = None) -> AgentResponse:
    return AgentResponse(
        response="raw",
        model=model,
        thinking=None,
        metrics=DUMMY_METRICS,
        idx=idx,
    )


def _step_one_to_step_two(response: AgentResponse) -> dict:
    """Custom mapper: takes name, produces greeting input."""
    return {"text": f"Hello, {response.model.name}!"}


def _failing_mapper(response: AgentResponse) -> dict:
    """Mapper that always raises."""
    raise RuntimeError("mapper exploded")


# --- PipelineStep Tests ---


def test_valid_step_default_mapper():
    """PipelineStep with default mapper is valid."""
    agent = _make_labeler()
    step = PipelineStep(agent=agent)
    assert step.agent is agent
    assert step.mapper is _default_mapper


def test_custom_mapper():
    """PipelineStep accepts a custom mapper."""
    agent = _make_labeler()
    step = PipelineStep(agent=agent, mapper=_step_one_to_step_two)
    assert step.mapper is _step_one_to_step_two


def test_invalid_agent_raises():
    """Non-BaseAgent agent raises PipelineConfigurationError."""
    with pytest.raises(PipelineConfigurationError):
        PipelineStep(agent="not-an-agent")


def test_non_callable_mapper_raises():
    """Non-callable mapper raises PipelineConfigurationError."""
    agent = _make_labeler()
    with pytest.raises(PipelineConfigurationError):
        PipelineStep(agent=agent, mapper="not-callable")


# --- Pipeline __init__ Tests ---


def test_valid_pipeline():
    """Pipeline with valid steps is created."""
    agent = _make_labeler()
    step = PipelineStep(agent=agent)
    pipeline = Pipeline(steps=[step])
    assert len(pipeline.steps) == 1


def test_empty_steps_raises():
    """Empty steps list raises PipelineConfigurationError."""
    with pytest.raises(PipelineConfigurationError):
        Pipeline(steps=[])


def test_non_step_raises():
    """Non-PipelineStep in list raises error."""
    with pytest.raises(PipelineConfigurationError):
        Pipeline(steps=["not-a-step"])


# --- _default_mapper Tests ---


def test_converts_model_to_dict():
    """Default mapper converts model to dict via model_dump."""
    model = StepOneOutput(name="Alice")
    response = _make_response(model)
    result = _default_mapper(response)
    assert result == {"name": "Alice"}


# --- Pipeline.act() sync Tests ---


@patch.object(LLMEngine, "chat")
def test_single_step(mock_chat):
    """Single-step pipeline returns correct response."""
    mock_chat.return_value = {
        "role": "assistant",
        "response": json.dumps({"name": "Alice"}),
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }
    agent = _make_labeler(StepOneOutput)
    pipeline = Pipeline(steps=[PipelineStep(agent=agent)])

    result = pipeline.act({"text": "test"})

    assert result.success is True
    assert result.model.name == "Alice"
    assert len(result.steps) == 1


@patch.object(LLMEngine, "chat")
def test_two_steps_with_mapper(mock_chat):
    """Two-step pipeline with custom mapper chains correctly."""
    mock_chat.side_effect = [
        {
            "role": "assistant",
            "response": json.dumps({"name": "Alice"}),
            "thinking": None,
            "metrics": DUMMY_METRICS,
        },
        {
            "role": "assistant",
            "response": json.dumps({"greeting": "Hello, Alice!"}),
            "thinking": None,
            "metrics": DUMMY_METRICS,
        },
    ]
    agent1 = _make_labeler(StepOneOutput)
    agent2 = _make_labeler(StepTwoOutput)
    pipeline = Pipeline(
        steps=[
            PipelineStep(
                agent=agent1,
                mapper=_step_one_to_step_two,
            ),
            PipelineStep(agent=agent2),
        ]
    )

    result = pipeline.act({"text": "test"})

    assert result.success is True
    assert result.model.greeting == "Hello, Alice!"
    assert len(result.steps) == 2


@patch.object(LLMEngine, "chat")
def test_short_circuit_on_none(mock_chat):
    """Pipeline short-circuits when model is None."""
    mock_chat.return_value = {
        "role": "assistant",
        "response": "invalid json",
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }
    agent1 = _make_labeler(StepOneOutput)
    agent2 = _make_labeler(StepTwoOutput)
    pipeline = Pipeline(
        steps=[
            PipelineStep(agent=agent1),
            PipelineStep(agent=agent2),
        ]
    )

    result = pipeline.act({"text": "test"})

    assert result.success is False
    assert result.model is None
    assert len(result.steps) == 1
    assert mock_chat.call_count == 1


@patch.object(LLMEngine, "chat")
def test_mapper_failure_short_circuits(mock_chat):
    """Pipeline short-circuits when mapper raises."""
    mock_chat.return_value = {
        "role": "assistant",
        "response": json.dumps({"name": "Alice"}),
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }
    agent1 = _make_labeler(StepOneOutput)
    agent2 = _make_labeler(StepTwoOutput)
    pipeline = Pipeline(
        steps=[
            PipelineStep(agent=agent1, mapper=_failing_mapper),
            PipelineStep(agent=agent2),
        ]
    )

    result = pipeline.act({"text": "test"})

    assert result.success is False
    assert result.model is None
    assert len(result.steps) == 1
    assert mock_chat.call_count == 1


# --- Pipeline.act_async() Tests ---


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_async_single_step(mock_chat):
    """Async single-step pipeline works."""
    mock_chat.return_value = {
        "role": "assistant",
        "response": json.dumps({"name": "Bob"}),
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }
    agent = _make_labeler(StepOneOutput)
    pipeline = Pipeline(steps=[PipelineStep(agent=agent)])

    result = await pipeline.act_async({"text": "test"})

    assert result.success is True
    assert result.model.name == "Bob"


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_async_two_steps(mock_chat):
    """Async two-step pipeline chains correctly."""
    mock_chat.side_effect = [
        {
            "role": "assistant",
            "response": json.dumps({"name": "Bob"}),
            "thinking": None,
            "metrics": DUMMY_METRICS,
        },
        {
            "role": "assistant",
            "response": json.dumps({"greeting": "Hello, Bob!"}),
            "thinking": None,
            "metrics": DUMMY_METRICS,
        },
    ]
    agent1 = _make_labeler(StepOneOutput)
    agent2 = _make_labeler(StepTwoOutput)
    pipeline = Pipeline(
        steps=[
            PipelineStep(
                agent=agent1,
                mapper=_step_one_to_step_two,
            ),
            PipelineStep(agent=agent2),
        ]
    )

    result = await pipeline.act_async({"text": "test"})

    assert result.success is True
    assert result.model.greeting == "Hello, Bob!"
    assert len(result.steps) == 2


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_async_short_circuit(mock_chat):
    """Async pipeline short-circuits on None model."""
    mock_chat.return_value = {
        "role": "assistant",
        "response": "bad json",
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }
    agent1 = _make_labeler(StepOneOutput)
    agent2 = _make_labeler(StepTwoOutput)
    pipeline = Pipeline(
        steps=[
            PipelineStep(agent=agent1),
            PipelineStep(agent=agent2),
        ]
    )

    result = await pipeline.act_async({"text": "test"})

    assert result.success is False
    assert result.model is None
    assert len(result.steps) == 1


# --- Pipeline.act_in_batch() Tests ---


@patch.object(LLMEngine, "chat")
def test_batch_all_succeed(mock_chat):
    """Batch pipeline: all items succeed through all steps."""
    mock_chat.side_effect = [
        # Step 1, item 0
        {
            "role": "assistant",
            "response": json.dumps({"name": "Alice"}),
            "thinking": None,
            "metrics": DUMMY_METRICS,
        },
        # Step 1, item 1
        {
            "role": "assistant",
            "response": json.dumps({"name": "Bob"}),
            "thinking": None,
            "metrics": DUMMY_METRICS,
        },
        # Step 2, item 0
        {
            "role": "assistant",
            "response": json.dumps({"greeting": "Hello, Alice!"}),
            "thinking": None,
            "metrics": DUMMY_METRICS,
        },
        # Step 2, item 1
        {
            "role": "assistant",
            "response": json.dumps({"greeting": "Hello, Bob!"}),
            "thinking": None,
            "metrics": DUMMY_METRICS,
        },
    ]
    agent1 = _make_labeler(StepOneOutput)
    agent2 = _make_labeler(StepTwoOutput)
    pipeline = Pipeline(
        steps=[
            PipelineStep(
                agent=agent1,
                mapper=_step_one_to_step_two,
            ),
            PipelineStep(agent=agent2),
        ]
    )

    results = pipeline.act_in_batch([{"text": "a"}, {"text": "b"}])

    assert len(results) == 2
    assert results[0].success is True
    assert results[0].model.greeting == "Hello, Alice!"
    assert results[1].success is True
    assert results[1].model.greeting == "Hello, Bob!"


@patch.object(LLMEngine, "chat")
def test_batch_partial_failure(mock_chat):
    """Batch pipeline: one item fails, others continue."""
    mock_chat.side_effect = [
        # Step 1, item 0 — parse failure
        {
            "role": "assistant",
            "response": "invalid",
            "thinking": None,
            "metrics": DUMMY_METRICS,
        },
        # Step 1, item 1 — success
        {
            "role": "assistant",
            "response": json.dumps({"name": "Bob"}),
            "thinking": None,
            "metrics": DUMMY_METRICS,
        },
        # Step 2, item 1 only (item 0 excluded)
        {
            "role": "assistant",
            "response": json.dumps({"greeting": "Hello, Bob!"}),
            "thinking": None,
            "metrics": DUMMY_METRICS,
        },
    ]
    agent1 = _make_labeler(StepOneOutput)
    agent2 = _make_labeler(StepTwoOutput)
    pipeline = Pipeline(
        steps=[
            PipelineStep(
                agent=agent1,
                mapper=_step_one_to_step_two,
            ),
            PipelineStep(agent=agent2),
        ]
    )

    results = pipeline.act_in_batch([{"text": "a"}, {"text": "b"}])

    assert len(results) == 2
    assert results[0].success is False
    assert results[0].model is None
    assert results[1].success is True
    assert results[1].model.greeting == "Hello, Bob!"


# --- Pipeline.act_in_batch_async() Tests ---


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_batch_async_all_succeed(mock_chat):
    """Async batch: all items succeed through all steps."""
    mock_chat.side_effect = [
        # Step 1, item 0
        {
            "role": "assistant",
            "response": json.dumps({"name": "Alice"}),
            "thinking": None,
            "metrics": DUMMY_METRICS,
        },
        # Step 1, item 1
        {
            "role": "assistant",
            "response": json.dumps({"name": "Bob"}),
            "thinking": None,
            "metrics": DUMMY_METRICS,
        },
        # Step 2, item 0
        {
            "role": "assistant",
            "response": json.dumps({"greeting": "Hello, Alice!"}),
            "thinking": None,
            "metrics": DUMMY_METRICS,
        },
        # Step 2, item 1
        {
            "role": "assistant",
            "response": json.dumps({"greeting": "Hello, Bob!"}),
            "thinking": None,
            "metrics": DUMMY_METRICS,
        },
    ]
    agent1 = _make_labeler(StepOneOutput)
    agent2 = _make_labeler(StepTwoOutput)
    pipeline = Pipeline(
        steps=[
            PipelineStep(
                agent=agent1,
                mapper=_step_one_to_step_two,
            ),
            PipelineStep(agent=agent2),
        ]
    )

    results = await pipeline.act_in_batch_async(
        [{"text": "a"}, {"text": "b"}]
    )

    assert len(results) == 2
    assert results[0].success is True
    assert results[0].model.greeting == "Hello, Alice!"
    assert results[1].success is True
    assert results[1].model.greeting == "Hello, Bob!"


# --- Pipeline.from_yaml() Tests ---


def test_valid_yaml(tmp_path):
    """Valid YAML with default agent_type creates Pipeline."""
    # Create a minimal agent YAML config
    agent_yaml = tmp_path / "agent.yml"
    agent_yaml.write_text(
        "name: TestAgent\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Test.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
    )

    # Create pipeline YAML
    pipeline_yaml = tmp_path / "pipeline.yml"
    pipeline_yaml.write_text(
        f"steps:\n  - agent: {agent_yaml}\n  - agent: {agent_yaml}\n"
    )

    pipeline = Pipeline.from_yaml(str(pipeline_yaml))
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0].mapper is _default_mapper


def test_yaml_with_agent_type(tmp_path):
    """YAML with explicit agent_type resolves correctly."""
    agent_yaml = tmp_path / "agent.yml"
    agent_yaml.write_text(
        "name: TestAgent\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Test.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
    )

    pipeline_yaml = tmp_path / "pipeline.yml"
    pipeline_yaml.write_text(
        f"steps:\n"
        f"  - agent: {agent_yaml}\n"
        f"    agent_type: agents7.agents.labeler.Labeler\n"
    )

    pipeline = Pipeline.from_yaml(str(pipeline_yaml))
    assert len(pipeline.steps) == 1


def test_missing_steps_raises(tmp_path):
    """YAML without steps raises PipelineConfigurationError."""
    pipeline_yaml = tmp_path / "pipeline.yml"
    pipeline_yaml.write_text("name: BadPipeline\n")

    with pytest.raises(PipelineConfigurationError):
        Pipeline.from_yaml(str(pipeline_yaml))


# --- Pipeline.act_in_batch_async() concurrency Tests ---


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_batch_async_concurrent_within_step(mock_chat):
    """Items within a step run concurrently (not sequentially)."""
    peak = 0
    current = 0
    lock = asyncio.Lock()

    async def _side_effect(**kwargs):
        nonlocal peak, current
        async with lock:
            current += 1
            peak = max(peak, current)
        await asyncio.sleep(0.05)
        async with lock:
            current -= 1
        return {
            "role": "assistant",
            "response": json.dumps({"name": "X"}),
            "thinking": None,
            "metrics": DUMMY_METRICS,
        }

    mock_chat.side_effect = _side_effect

    agent = _make_labeler(StepOneOutput)
    pipeline = Pipeline(steps=[PipelineStep(agent=agent)])

    await pipeline.act_in_batch_async(
        [{"text": "a"}, {"text": "b"}, {"text": "c"}],
        batch_settings=BatchSettings(max_concurrent=None),
    )
    # All 3 items should have been in-flight simultaneously
    assert peak == 3


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_batch_async_max_concurrent(mock_chat):
    """Semaphore limits concurrency within each step."""
    peak = 0
    current = 0
    lock = asyncio.Lock()

    async def _side_effect(**kwargs):
        nonlocal peak, current
        async with lock:
            current += 1
            peak = max(peak, current)
        await asyncio.sleep(0.05)
        async with lock:
            current -= 1
        return {
            "role": "assistant",
            "response": json.dumps({"name": "X"}),
            "thinking": None,
            "metrics": DUMMY_METRICS,
        }

    mock_chat.side_effect = _side_effect

    agent = _make_labeler(StepOneOutput)
    pipeline = Pipeline(steps=[PipelineStep(agent=agent)])

    await pipeline.act_in_batch_async(
        [
            {"text": "a"},
            {"text": "b"},
            {"text": "c"},
            {"text": "d"},
            {"text": "e"},
        ],
        batch_settings=BatchSettings(max_concurrent=2),
    )
    assert peak <= 2
