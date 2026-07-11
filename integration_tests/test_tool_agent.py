"""End-to-end ToolAgent test against a real Ollama model.

Uses ``qwen2.5:3b`` — a small, fast, tool-capable model with no
thinking phase — so the two-turn tool loop (request tool call ->
execute -> feed result back -> final answer) completes well under
the integration suite's per-test timeout.
"""

import pytest
from integration_helpers import is_model_available

from agents7.agents.tool_agent import ToolAgent
from agents7.llm import LLMEngine

_MODEL = "qwen2.5:3b"

pytestmark = pytest.mark.skipif(
    not is_model_available(_MODEL),
    reason=f"{_MODEL} not available",
)


def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"It is 18 degrees and sunny in {city}."


def test_tool_agent_calls_weather_tool():
    agent = ToolAgent(
        name="weather-bot",
        version=1,
        engine=LLMEngine(provider="ollama", model_name=_MODEL),
        system_prompt=(
            "You answer weather questions. Use the get_weather "
            "tool when asked about a city's weather, then report "
            "the result to the user."
        ),
        tools=[get_weather],
    )
    out = agent.act("What's the weather in Paris?")
    assert "18" in out.response or "sunny" in out.response.lower()
    assert out.metrics.tokens > 0
