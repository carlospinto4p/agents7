"""Shared test helpers used across test files."""

from types import SimpleNamespace

from pydantic import BaseModel

from agents7.agents.labeler import Labeler
from agents7.llm import LLMEngine


# --- Shared constants ---

ENGINE = LLMEngine(provider="ollama", model_name="qwen3")

DUMMY_METRICS = {
    "tokens": 10,
    "prompt_tokens": 5,
    "total_duration": 0.1,
}


# --- Shared models ---


class DummyOutput(BaseModel):
    value: str


# --- Agent factory helpers ---


def make_labeler(**kwargs) -> Labeler:
    """Create a Labeler with sensible test defaults."""
    defaults = {
        "name": "test",
        "version": 1,
        "engine": ENGINE,
        "system_prompt": "Test prompt.",
        "output_model": DummyOutput,
    }
    defaults.update(kwargs)
    return Labeler(**defaults)


# --- Streaming chunk helpers ---


def chat_chunk(
    content: str = "",
    thinking: str = "",
    role: str = "assistant",
    done: bool = False,
    eval_count: int | None = None,
    prompt_eval_count: int | None = None,
    total_duration: int | None = None,
) -> SimpleNamespace:
    """Build a mock ollama chat streaming chunk."""
    msg = SimpleNamespace(
        content=content,
        thinking=thinking,
        role=role,
    )
    return SimpleNamespace(
        message=msg,
        done=done,
        eval_count=eval_count,
        prompt_eval_count=prompt_eval_count,
        total_duration=total_duration,
    )


async def async_iter(items):
    """Turn a list into an async iterator."""
    for item in items:
        yield item
