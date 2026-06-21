"""A callable wrapped as an LLM-invocable tool."""

from collections.abc import Callable
from dataclasses import dataclass, field

from agents7.tools.schema import schema_from_callable


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
        """Invoke the wrapped function and stringify its return.

        :param kwargs: Arguments parsed from the model's tool call.
        :return: The function's return value coerced to ``str``.
        """
        return str(self.fn(**kwargs))
