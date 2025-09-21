import importlib
from enum import Enum

from pydantic import BaseModel


def get_examples_str(examples: list[dict]) -> str:
    """
    Convert examples to a string format for inclusion in the system prompt.
    :param examples: List of dictionaries of examples with 'input' and
        'output' keys.
    :return: Formatted string of examples.
    """
    examples_str = "## Examples:"
    if not isinstance(examples, list):
        raise ValueError("Field `system_examples` must be a list")
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError(
                "Each example in `system_examples` must be a dictionary"
            )
        if set(example.keys()) != {"input", "output"}:
            raise ValueError(
                "Each example in `system_examples` must contain "
                "`input` and `output` keys"
            )
        examples_str += (
            f"\n\n- Input: {example['input']}\n- Output: {example['output']}"
        )
    return examples_str


def import_from_path(path: str) -> type[BaseModel]:
    """
    Import a class from a given module path.
    :param path: Path to the module and class, e.g.
        "module.submodule.ClassName".
    :return: The class object imported from the specified path.
    """
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
