"""A callable wrapped as an LLM-invocable tool."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

from agents7.tools.schema import schema_from_callable


def serialize_tool_result(value: object) -> str:
    """Serialize a tool's return value for feeding back to the model.

    Plain strings pass through unchanged. Structured returns are
    JSON-encoded so the model receives valid JSON rather than a
    Python ``repr``: Pydantic models via ``model_dump_json``, and
    ``dict`` / ``list`` / ``tuple`` via ``json.dumps``. Anything
    else falls back to ``str``.

    :param value: The value returned by the wrapped callable.
    :return: A string representation suitable for a tool-result
        message.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


@dataclass(frozen=True)
class Tool:
    """Wrap a Python callable as a tool the model can call.

    :param fn: The function to expose.
    :param name: Tool name shown to the model. Defaults to
        ``fn.__name__``.
    :param description: Human description. Defaults to the first
        line of ``fn``'s docstring.
    :param json_schema: Parameter schema. Auto-derived from
        ``fn`` when omitted.
    """

    fn: Callable
    name: str = ""
    description: str = ""
    json_schema: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.fn.__name__)
        if not self.description:
            doc = (self.fn.__doc__ or "").strip().split("\n")[0]
            object.__setattr__(self, "description", doc)
        if not self.json_schema:
            object.__setattr__(
                self, "json_schema", schema_from_callable(self.fn)
            )

    def run(self, **kwargs) -> str:
        """Invoke the wrapped function and serialize its return.

        Structured returns (`dict`, `list`, `tuple`, or a Pydantic
        model) are JSON-encoded; plain strings pass through; any
        other type falls back to ``str``. See
        `serialize_tool_result`.

        :param kwargs: Arguments parsed from the model's tool call.
        :return: The function's return value serialized to a string.
        """
        return serialize_tool_result(self.fn(**kwargs))
