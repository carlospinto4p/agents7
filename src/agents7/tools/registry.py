"""A name-indexed collection of tools an agent can call."""

from collections.abc import Callable

from agents7.exceptions import ToolSchemaError
from agents7.tools.tool import Tool


class ToolRegistry:
    """Maps tool names to `Tool` instances and runs them.

    :param tools: A list of `Tool` instances or bare callables.
        Bare callables are wrapped in `Tool` automatically.
    """

    def __init__(self, tools: list[Tool | Callable]):
        self._tools: dict[str, Tool] = {}
        for entry in tools:
            tool = entry if isinstance(entry, Tool) else Tool(entry)
            self._tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        """Return provider-neutral function schemas for every tool.

        :return: A list of ``{"type": "function", "function":
            {...}}`` dicts suitable for an OpenAI/Ollama-style
            ``tools`` parameter.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.json_schema,
                },
            }
            for tool in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict) -> str:
        """Run the named tool with *arguments*.

        :param name: Tool name requested by the model.
        :param arguments: Keyword arguments for the tool.
        :return: The tool's stringified result, or
            ``"Error: <message>"`` if the tool raised.
        :raises ToolSchemaError: If *name* is not registered.
        """
        if name not in self._tools:
            raise ToolSchemaError(f"Unknown tool `{name}` requested.")
        try:
            return self._tools[name].run(**arguments)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model
            return f"Error: {exc}"

    @property
    def names(self) -> list[str]:
        """Registered tool names."""
        return list(self._tools)
