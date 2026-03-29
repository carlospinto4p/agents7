"""Unit tests for agents7.testing utilities."""

import pytest

from agents7.agents.labeler import Labeler
from agents7.testing import assert_accuracy_above, create_mock_engine

from helpers import DummyOutput


# --- create_mock_engine ---


def test_mock_engine_default_response():
    """Default mock returns '{"value": "mock"}'."""
    engine = create_mock_engine()
    assert engine.provider == "mock"
    assert engine.model_name == "mock"

    agent = Labeler(
        name="test",
        version=1,
        engine=engine,
        system_prompt="Test.",
        output_model=DummyOutput,
    )
    response = agent.act({"text": "anything"})
    assert response.model is not None
    assert response.model.value == "mock"


def test_mock_engine_single_response():
    """Single string response returned for all inputs."""
    engine = create_mock_engine('{"value": "hello"}')
    agent = Labeler(
        name="test",
        version=1,
        engine=engine,
        system_prompt="Test.",
        output_model=DummyOutput,
    )
    response = agent.act({"text": "anything"})
    assert response.model.value == "hello"


def test_mock_engine_dict_responses():
    """Dict responses matched by input content."""
    engine = create_mock_engine(
        {
            "good": '{"value": "positive"}',
            "bad": '{"value": "negative"}',
        }
    )
    agent = Labeler(
        name="test",
        version=1,
        engine=engine,
        system_prompt="Test.",
        output_model=DummyOutput,
    )
    r1 = agent.act({"text": "good day"})
    assert r1.model.value == "positive"

    r2 = agent.act({"text": "bad day"})
    assert r2.model.value == "negative"


def test_mock_engine_dict_fallback():
    """Unmatched input falls back to default."""
    engine = create_mock_engine(
        {
            "zzz_unique": '{"value": "found"}',
        }
    )
    agent = Labeler(
        name="test",
        version=1,
        engine=engine,
        system_prompt="Test.",
        output_model=DummyOutput,
    )
    response = agent.act({"text": "something else"})
    assert response.model.value == "mock"


def test_mock_engine_custom_model_name():
    """Custom model_name is stored."""
    engine = create_mock_engine(model_name="gpt-test")
    assert engine.model_name == "gpt-test"
    assert engine.model_id == "mock::gpt-test"


@pytest.mark.asyncio
async def test_mock_engine_async():
    """Mock engine works with async act."""
    engine = create_mock_engine('{"value": "async_ok"}')
    agent = Labeler(
        name="test",
        version=1,
        engine=engine,
        system_prompt="Test.",
        output_model=DummyOutput,
    )
    response = await agent.act_async({"text": "test"})
    assert response.model.value == "async_ok"


# --- assert_accuracy_above ---


def test_assert_accuracy_above_passes():
    """Passes when accuracy meets threshold."""
    engine = create_mock_engine('{"value": "ok"}')
    agent = Labeler(
        name="test",
        version=1,
        engine=engine,
        system_prompt="Test.",
        output_model=DummyOutput,
    )
    eval_data = [
        {"input": {"text": "a"}, "value": "ok"},
        {"input": {"text": "b"}, "value": "ok"},
    ]
    metrics = assert_accuracy_above(
        agent,
        eval_data,
        ["value"],
        1.0,
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["passed"] == 2


def test_assert_accuracy_above_fails():
    """Raises AssertionError when below threshold."""
    engine = create_mock_engine('{"value": "wrong"}')
    agent = Labeler(
        name="test",
        version=1,
        engine=engine,
        system_prompt="Test.",
        output_model=DummyOutput,
    )
    eval_data = [
        {"input": {"text": "a"}, "value": "expected"},
    ]
    with pytest.raises(AssertionError, match="below threshold"):
        assert_accuracy_above(
            agent,
            eval_data,
            ["value"],
            0.9,
        )
