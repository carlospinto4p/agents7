"""A callable wrapped as an LLM-invocable tool."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from agents7.exceptions import ToolArgumentError, ToolSchemaError
from agents7.tools.schema import args_model_from_callable


def _jsonable(value: object) -> object:
    """Convert *value* into JSON-encodable primitives.

    Recurses through ``dict`` / ``list`` / ``tuple`` so Pydantic
    models nested inside containers become objects rather than
    Python ``repr`` strings.

    :param value: Any value returned by a tool.
    :return: The same structure with models dumped to dicts.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def serialize_tool_result(value: object) -> str:
    """Serialize a tool's return value for feeding back to the model.

    Plain strings pass through unchanged. Structured returns are
    JSON-encoded so the model receives valid JSON rather than a
    Python ``repr``: Pydantic models via ``model_dump``, and
    ``dict`` / ``list`` / ``tuple`` via ``json.dumps``. Containers
    are walked recursively, so a ``list[Product]`` encodes as an
    array of objects rather than an array of ``repr`` strings.
    Anything else falls back to ``str``.

    :param value: The value returned by the wrapped callable.
    :return: A string representation suitable for a tool-result
        message.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (BaseModel, dict, list, tuple)):
        return json.dumps(_jsonable(value), ensure_ascii=False, default=str)
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
    args_model: type[BaseModel] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.fn.__name__)
        if not self.description:
            doc = (self.fn.__doc__ or "").strip().split("\n")[0]
            object.__setattr__(self, "description", doc)
        if self.args_model is None:
            try:
                model = args_model_from_callable(self.fn)
            except ToolSchemaError:
                # An untyped callable with a hand-written schema is
                # still usable — it just gets no argument validation.
                if not self.json_schema:
                    raise
                model = None
            object.__setattr__(self, "args_model", model)
        if not self.json_schema and self.args_model is not None:
            object.__setattr__(
                self, "json_schema", self.args_model.model_json_schema()
            )

    def validate_arguments(self, arguments: dict) -> dict:
        """Coerce *arguments* to the types the callable declares.

        Small models routinely send the right values in the wrong
        shape — ``"3"`` for an ``int``, or an envelope echoing the
        schema. Re-validating through the same Pydantic model the
        tool's schema came from coerces what it can and rejects
        the rest with a message the model can act on.

        :param arguments: Raw arguments from the model's tool call.
        :return: Validated keyword arguments for the callable.
        :raises ToolArgumentError: If the arguments cannot be
            coerced to the declared types.
        """
        if self.args_model is None:
            return arguments
        try:
            validated = self.args_model.model_validate(arguments)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in exc.errors()
            )
            raise ToolArgumentError(
                f"Invalid arguments for `{self.name}`: {details}"
            ) from exc
        return {
            name: getattr(validated, name)
            for name in type(validated).model_fields
        }

    def run(self, **kwargs) -> str:
        """Invoke the wrapped function and serialize its return.

        Arguments are validated and coerced against the tool's own
        schema first — see `validate_arguments`. Structured returns
        (`dict`, `list`, `tuple`, or a Pydantic model) are
        JSON-encoded; plain strings pass through; any other type
        falls back to ``str``. See `serialize_tool_result`.

        :param kwargs: Arguments parsed from the model's tool call.
        :return: The function's return value serialized to a string.
        :raises ToolArgumentError: If the arguments fail validation.
        """
        return serialize_tool_result(
            self.fn(**self.validate_arguments(kwargs))
        )
