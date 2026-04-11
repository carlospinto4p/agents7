"""Unit tests for Chatter end conditions."""

from unittest.mock import AsyncMock, patch

import pytest

from agents7.agents.chatter import Chatter
from agents7.exceptions import ChatEndedError
from agents7.llm import LLMEngine

from helpers import chat_response_dict, make_chatter


# --- max_turns ---


@patch.object(LLMEngine, "chat")
def test_max_turns_allows_up_to_limit(mock_chat):
    """Chat works up to max_turns."""
    mock_chat.return_value = chat_response_dict()
    agent = make_chatter(max_turns=2)

    agent.chat("First")
    agent.chat("Second")
    assert agent.turns == 2


@patch.object(LLMEngine, "chat")
def test_max_turns_raises_after_limit(mock_chat):
    """ChatEndedError raised after max_turns reached."""
    mock_chat.return_value = chat_response_dict()
    agent = make_chatter(max_turns=2)

    agent.chat("First")
    agent.chat("Second")

    with pytest.raises(ChatEndedError):
        agent.chat("Third")


@patch.object(LLMEngine, "chat")
def test_max_turns_none_means_unlimited(mock_chat):
    """No max_turns means no limit."""
    mock_chat.return_value = chat_response_dict()
    agent = make_chatter(max_turns=None)

    for i in range(10):
        agent.chat(f"Message {i}")

    assert agent.turns == 10


# --- end_condition ---


@patch.object(LLMEngine, "chat")
def test_end_condition_triggers(mock_chat):
    """end_condition returning True ends conversation."""
    mock_chat.return_value = chat_response_dict("goodbye")

    def ends_on_goodbye(history):
        return any("goodbye" in m["content"].lower() for m in history)

    agent = make_chatter(end_condition=ends_on_goodbye)
    agent.chat("Hello")

    with pytest.raises(ChatEndedError):
        agent.chat("Next")


@patch.object(LLMEngine, "chat")
def test_end_condition_false_continues(mock_chat):
    """end_condition returning False allows chat."""
    mock_chat.return_value = chat_response_dict("Hello")
    agent = make_chatter(
        end_condition=lambda h: False,
    )
    agent.chat("Hi")
    agent.chat("Again")
    assert agent.turns == 2


@patch.object(LLMEngine, "chat")
def test_end_condition_none_means_no_check(mock_chat):
    """No end_condition means no custom check."""
    mock_chat.return_value = chat_response_dict()
    agent = make_chatter(end_condition=None)
    agent.chat("Hi")
    assert agent.turns == 1


# --- turns property ---


def test_turns_starts_at_zero():
    """turns is 0 before any chat."""
    agent = make_chatter()
    assert agent.turns == 0


@patch.object(LLMEngine, "chat")
def test_turns_counts_complete_pairs(mock_chat):
    """turns counts assistant messages (complete pairs)."""
    mock_chat.return_value = chat_response_dict()
    agent = make_chatter()
    agent.chat("One")
    agent.chat("Two")
    assert agent.turns == 2


# --- ended property ---


@patch.object(LLMEngine, "chat")
def test_ended_false_before_limit(mock_chat):
    """ended is False when under max_turns."""
    mock_chat.return_value = chat_response_dict()
    agent = make_chatter(max_turns=3)
    agent.chat("Hi")
    assert agent.ended is False


@patch.object(LLMEngine, "chat")
def test_ended_true_at_limit(mock_chat):
    """ended is True when max_turns reached."""
    mock_chat.return_value = chat_response_dict()
    agent = make_chatter(max_turns=1)
    agent.chat("Hi")
    assert agent.ended is True


# --- reset clears turn count ---


@patch.object(LLMEngine, "chat")
def test_reset_allows_new_turns(mock_chat):
    """reset() clears history so max_turns resets."""
    mock_chat.return_value = chat_response_dict()
    agent = make_chatter(max_turns=1)
    agent.chat("Hi")

    assert agent.ended is True
    agent.reset()
    assert agent.ended is False
    assert agent.turns == 0

    agent.chat("Again")
    assert agent.turns == 1


# --- Async ---


@pytest.mark.asyncio
@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
async def test_async_max_turns_raises(mock_chat):
    """chat_async raises ChatEndedError after max_turns."""
    mock_chat.return_value = chat_response_dict()
    agent = make_chatter(max_turns=1)
    await agent.chat_async("First")

    with pytest.raises(ChatEndedError):
        await agent.chat_async("Second")


# --- Streaming ---


@patch.object(LLMEngine, "chat_stream")
def test_stream_raises_when_ended(mock_stream):
    """chat_stream raises ChatEndedError when ended."""
    agent = make_chatter(max_turns=0)

    with pytest.raises(ChatEndedError):
        agent.chat_stream("Hi")


# --- Both conditions ---


@patch.object(LLMEngine, "chat")
def test_both_max_turns_and_end_condition(mock_chat):
    """Either condition can end the conversation."""
    mock_chat.return_value = chat_response_dict("bye")

    agent = make_chatter(
        max_turns=10,
        end_condition=lambda h: any("bye" in m["content"] for m in h),
    )
    agent.chat("Hello")
    # end_condition triggers before max_turns
    with pytest.raises(ChatEndedError):
        agent.chat("Next")


# --- YAML loading ---


def test_from_yaml_with_max_turns(tmp_path):
    """YAML config with max_turns."""
    yaml_content = (
        "name: EndChatter\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Hi.\n"
        "max_turns: 5\n"
    )
    yaml_file = tmp_path / "chatter.yml"
    yaml_file.write_text(yaml_content)

    agent = Chatter.from_yaml(str(yaml_file))
    assert agent.max_turns == 5
