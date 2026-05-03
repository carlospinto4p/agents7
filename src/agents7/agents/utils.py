import asyncio
import importlib
import re
from enum import Enum, StrEnum

from pydantic import BaseModel

from agents7.exceptions import (
    AgentConfigurationError,
    PropertyConfigurationError,
)

# Only allow standard Python identifier/dot paths (e.g. "pkg.sub.Class").
# Rejects paths with spaces, slashes, or other shell-special chars that
# could indicate injection if config is user-controlled.
_SAFE_MODULE_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


async def run_with_semaphore(coro, sem: asyncio.Semaphore | None):
    """
    Await a coroutine with optional semaphore concurrency limiting.
    :param coro: The coroutine to await.
    :param sem: An `asyncio.Semaphore` to limit concurrency, or
        `None` to run without limiting.
    :return: The result of the coroutine.
    """
    if sem:
        async with sem:
            return await coro
    return await coro


def resolve_strenum(annotation: type) -> type | None:
    """Unwrap Optional/Union and return the StrEnum type.

    Handles plain StrEnum subclasses as well as
    ``Optional[SomeEnum]`` (i.e. ``SomeEnum | None``).

    :param annotation: A type annotation from a Pydantic
        model field.
    :return: The StrEnum subclass, or None if the annotation
        is not a StrEnum.
    """
    if hasattr(annotation, "__args__"):
        for arg in annotation.__args__:
            if isinstance(arg, type) and issubclass(arg, StrEnum):
                return arg
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        return annotation
    return None


def enum_members(enum_cls: type[StrEnum]) -> dict[str, str]:
    """Return ``{name: value}`` for all members of a StrEnum.

    :param enum_cls: A StrEnum subclass.
    :return: Dictionary mapping member names to string values.
    """
    return {m.name: str(m.value) for m in enum_cls}


def extract_enum_members(
    output_model: type[BaseModel],
    field_name: str,
) -> dict[str, str]:
    """
    Extract `StrEnum` members from a Pydantic model field.

    :param output_model: Pydantic model class containing the field.
    :param field_name: Name of the field whose annotation must be
        a `StrEnum`.
    :return: Dictionary mapping member names to their string values.
    :raises PropertyConfigurationError: If the field annotation is
        not a `StrEnum` subclass.
    """
    annotation = output_model.model_fields[field_name].annotation
    enum_cls = resolve_strenum(annotation)
    if enum_cls is None:
        raise PropertyConfigurationError(
            f"`{field_name}` field in the output_model "
            f"must be a `StrEnum`. "
            f"Got `{annotation}` instead."
        )
    return enum_members(enum_cls)


def get_examples_str(examples: list[dict]) -> str:
    """
    Convert examples to a string format for inclusion in the system prompt.
    :param examples: List of dictionaries of examples with 'input' and
        'output' keys.
    :return: Formatted string of examples.
    """
    if not isinstance(examples, list):
        raise AgentConfigurationError(
            "Field `system_examples` must be a list"
        )
    parts = ["## Examples:"]
    for example in examples:
        if not isinstance(example, dict):
            raise AgentConfigurationError(
                "Each example in `system_examples` must be a dictionary"
            )
        if set(example.keys()) != {"input", "output"}:
            raise AgentConfigurationError(
                "Each example in `system_examples` must "
                "contain `input` and `output` keys"
            )
        parts.append(
            f"- Input: {example['input']}\n- Output: {example['output']}"
        )
    return "\n\n".join(parts)


def import_from_path(path: str) -> type:
    """
    Import a class from a given module path.
    :param path: Path to the module and class, e.g.
        "module.submodule.ClassName".
    :return: The class object imported from the specified path.
    :raises AgentConfigurationError: If *path* contains characters
        outside the safe set (alphanumeric, underscores, dots).
        This guards against arbitrary code execution when the path
        originates from user-controlled config.
    """
    if not _SAFE_MODULE_PATH.match(path):
        raise AgentConfigurationError(
            f"Unsafe module path `{path}`: only letters, digits, "
            f"underscores, and dots are allowed."
        )
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def model_class_to_string(model_cls: type[BaseModel]) -> str:
    """
    Convert a Pydantic model class to a string representation.
    This function generates a string representation of the model class,
    including its fields, types, default values, and descriptions.
    :param model_cls: Pydantic model class to convert.
    :return: String representation of the model class.
    """
    lines = [f"class {model_cls.__name__}:"]
    for field_name, field_info in model_cls.model_fields.items():
        annotation = (
            field_info.annotation.__name__
            if hasattr(field_info.annotation, "__name__")
            else str(field_info.annotation)
        )

        default_repr = ""
        if str(field_info.default) != "PydanticUndefined":
            default_repr = f" = {field_info.default!r}"

        description = ""
        if field_info.description:
            description = f"\n\t\tDescription: {field_info.description}"

        if isinstance(field_info.annotation, type) and issubclass(
            field_info.annotation, Enum
        ):
            possible_values = "\n\t\t- ".join(
                [f"{e.name}: {e.value}" for e in field_info.annotation]
            )
            description += f". Possible values:\n\t\t- {possible_values}"

        lines.append(
            f"    {field_name}: {annotation}{default_repr}{description}"
        )

    return "\n".join(lines)
