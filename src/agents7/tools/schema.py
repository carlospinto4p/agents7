"""Derive a JSON schema from a Python callable's signature."""

import inspect
from collections.abc import Callable

from pydantic import BaseModel, create_model

from agents7.exceptions import ToolSchemaError


def args_model_from_callable(fn: Callable) -> type[BaseModel]:
    """Build a Pydantic model for *fn*'s parameters.

    The model is both the source of the tool's JSON schema and
    the validator applied to the arguments a model sends back —
    so what a tool advertises and what it accepts can never
    drift apart.

    :param fn: The function to derive a model from. Every
        parameter must have a type hint; ``*args`` / ``**kwargs``
        are not supported.
    :return: A Pydantic model class whose fields are ``fn``'s
        parameters.
    :raises ToolSchemaError: If a parameter lacks a type hint or
        the signature uses ``*args`` / ``**kwargs``.
    """
    sig = inspect.signature(fn)
    fields: dict[str, tuple] = {}
    for name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ToolSchemaError(
                f"Tool `{fn.__name__}` has an unsupported "
                f"*args/**kwargs parameter `{name}`."
            )
        if param.annotation is inspect.Parameter.empty:
            raise ToolSchemaError(
                f"Parameter `{name}` of tool `{fn.__name__}` is "
                f"missing a type hint."
            )
        default = (
            ... if param.default is inspect.Parameter.empty else param.default
        )
        fields[name] = (param.annotation, default)
    return create_model(f"{fn.__name__}_Args", **fields)


def schema_from_callable(fn: Callable) -> dict:
    """Build a JSON schema for *fn*'s parameters.

    Constructs a Pydantic model from the callable's type hints
    via `args_model_from_callable` and returns its
    ``model_json_schema()``. Parameters without a default are
    marked required.

    :param fn: The function to derive a schema from.
    :return: A JSON-schema dict describing the parameters.
    :raises ToolSchemaError: If a parameter lacks a type hint or
        the signature uses ``*args`` / ``**kwargs``.
    """
    return args_model_from_callable(fn).model_json_schema()
