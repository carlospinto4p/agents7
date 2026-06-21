"""End-to-end ToolAgent test against a real Ollama model."""

from agents7.agents.tool_agent import ToolAgent
from agents7.llm import LLMEngine


def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"It is 18 degrees and sunny in {city}."


def test_tool_agent_calls_weather_tool():
    agent = ToolAgent(
        name="weather-bot",
        version=1,
        engine=LLMEngine(provider="ollama", model_name="qwen3"),
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
