"""Unit tests for Labeler preprocessor."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from agents7.agents.labeler import Labeler
from agents7.llm import LLMEngine

from helpers import DUMMY_METRICS, make_labeler


def _chat_response(value="ok"):
    return {
        "role": "assistant",
        "response": json.dumps({"value": value}),
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }


def _upper_preprocessor(data):
    """Uppercase all string values in the input dict."""
    return {
        k: v.upper() if isinstance(v, str) else v for k, v in data.items()
    }


def _truncate_preprocessor(data):
    """Truncate text field to 10 chars."""
    if "text" in data:
        data = dict(data)
        data["text"] = data["text"][:10]
    return data


# --- No preprocessor (default) ---


@patch.object(LLMEngine, "chat")
def test_no_preprocessor_default(mock_chat):
    """Without preprocessor, input passes through unchanged."""
    mock_chat.return_value = _chat_response("ok")
    agent = make_labeler()
    agent.act({"text": "hello"})

    sent = mock_chat.call_args[1]["messages"][1]["content"]
    assert "hello" in sent


# --- Preprocessor applies to act() ---


@patch.object(LLMEngine, "chat")
def test_act_applies_preprocessor(mock_chat):
    """act() transforms input via preprocessor."""
    mock_chat.return_value = _chat_response("ok")
    agent = make_labeler(preprocessor=_upper_preprocessor)
    agent.act({"text": "hello"})

    sent = mock_chat.call_args[1]["messages"][1]["content"]
    assert "HELLO" in sent
    assert "hello" not in sent


# --- Preprocessor applies to act_async() ---


@patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_act_async_applies_preprocessor(mock_chat):
    """act_async() transforms input via preprocessor."""
    mock_chat.return_value = _chat_response("ok")
    agent = make_labeler(preprocessor=_upper_preprocessor)
    await agent.act_async({"text": "hello"})

    sent = mock_chat.call_args[1]["messages"][1]["content"]
    assert "HELLO" in sent


# --- Preprocessor applies to batch ---


@patch.object(LLMEngine, "chat")
def test_batch_applies_preprocessor(mock_chat):
    """act_in_batch() preprocesses each item."""
    items = [{"value": "ok"}]
    mock_chat.return_value = {
        "role": "assistant",
        "response": json.dumps({"items": items}),
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }
    agent = make_labeler(preprocessor=_upper_preprocessor)
    agent.act_in_batch(
        [{"text": "hello"}, {"text": "world"}],
        iter_func=None,
        requests_per_prompt=2,
    )

    sent = mock_chat.call_args[1]["messages"][1]["content"]
    assert "HELLO" in sent
    assert "WORLD" in sent


# --- Truncation preprocessor ---


@patch.object(LLMEngine, "chat")
def test_truncation_preprocessor(mock_chat):
    """Preprocessor can truncate input."""
    mock_chat.return_value = _chat_response("ok")
    agent = make_labeler(preprocessor=_truncate_preprocessor)
    agent.act({"text": "a very long text that should be cut"})

    sent = mock_chat.call_args[1]["messages"][1]["content"]
    assert "a very lon" in sent
    assert "should be cut" not in sent


# --- Preprocessor returns new dict ---


@patch.object(LLMEngine, "chat")
def test_preprocessor_returns_new_dict(mock_chat):
    """Preprocessor can return a completely new dict."""
    mock_chat.return_value = _chat_response("ok")

    def replace_preprocessor(data):
        return {"replaced": True}

    agent = make_labeler(preprocessor=replace_preprocessor)
    agent.act({"text": "original"})

    sent = mock_chat.call_args[1]["messages"][1]["content"]
    assert "replaced" in sent
    assert "original" not in sent


# --- YAML loading ---


def test_yaml_preprocessor(tmp_path):
    """preprocessor is loaded from YAML config."""
    yaml_content = (
        "name: PreAgent\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Test.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
        "preprocessor: agents7.models.demo.PersonInfo\n"
    )
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content)
    agent = Labeler.from_yaml(str(yaml_file))
    assert agent.preprocessor is not None


def test_yaml_no_preprocessor(tmp_path):
    """Without preprocessor in YAML, defaults to None."""
    yaml_content = (
        "name: NoPre\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Test.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
    )
    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(yaml_content)
    agent = Labeler.from_yaml(str(yaml_file))
    assert agent.preprocessor is None
