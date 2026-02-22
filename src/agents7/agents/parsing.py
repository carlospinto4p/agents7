"""Response parsing and enum-mapping utilities."""

import json
from enum import StrEnum
from typing import Callable

from pydantic import BaseModel

from agents7.logs import logger


class ResponseParser:
    """Parse and validate LLM responses against Pydantic models.

    Attributes:
        output_model (type[BaseModel]): The model used to
            validate single responses.
        output_model_list (type[BaseModel]): The model used
            to validate batch (list) responses.
        name (str): Agent name, used in log messages.
        validators (list[Callable] | None): Optional
            post-parse validator pipeline.
    """

    __slots__ = (
        "output_model",
        "output_model_list",
        "name",
        "validators",
    )

    def __init__(
        self,
        output_model: type[BaseModel],
        output_model_list: type[BaseModel],
        name: str,
        validators: list[Callable[[BaseModel], BaseModel]] | None = None,
    ):
        self.output_model = output_model
        self.output_model_list = output_model_list
        self.name = name
        self.validators = validators

    def parse(
        self, response: str, for_batch: bool = False
    ) -> BaseModel | None:
        """Parse a raw LLM response into a Pydantic model.

        :param response: The response string from the LLM.
        :param for_batch: If True, validates against
            `output_model_list` instead of `output_model`.
        :return: Validated model instance, or None on
            failure.
        """
        model_cls = self.output_model_list if for_batch else self.output_model
        model = None

        response = self._fix_malformed_json(response)

        try:
            model = model_cls.model_validate_json(response)
        except Exception:
            try:
                converted = self._convert_enum_names_to_values(response)
                model = model_cls.model_validate_json(converted)
            except Exception as e:
                logger.error(
                    f"Agent `{self.name}`: error parsing "
                    f"structured response:\n{e}"
                )
        return model

    def run_validators(self, model: BaseModel | None) -> BaseModel | None:
        """Run the validator pipeline on a parsed model.

        Each validator receives the model and returns it
        (possibly modified). If a validator raises, the
        model is rejected (returns None) and remaining
        validators are skipped.

        :param model: The parsed model instance, or None.
        :return: The validated (possibly transformed) model,
            or None.
        """
        if model is None or not self.validators:
            return model
        for validator in self.validators:
            try:
                model = validator(model)
            except Exception as e:
                logger.error(
                    f"Agent `{self.name}`: validator "
                    f"`{validator.__name__}` failed: {e}"
                )
                return None
        return model

    @staticmethod
    def _fix_malformed_json(response: str) -> str:
        """Fix common JSON malformations from LLM outputs.

        Some models return malformed JSON like
        '{"{"reason":...' instead of '{"reason":...'.
        """
        response = response.strip()

        if response.startswith("```"):
            lines = response.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines).strip()

        if response.startswith('{"{'):
            response = response[2:]

        if response.startswith('["['):
            response = response[2:]

        return response

    def _get_enum_mappings(self) -> dict[str, dict[str, str]]:
        """Get enum name-to-value mappings for all StrEnum
        fields in output_model.

        :return: Dict mapping field names to their
            enum name->value dicts.
        """
        enum_mappings = {}
        for (
            field_name,
            field_info,
        ) in self.output_model.model_fields.items():
            field_type = field_info.annotation
            if hasattr(field_type, "__args__"):
                args = field_type.__args__
                for arg in args:
                    if isinstance(arg, type) and issubclass(arg, StrEnum):
                        field_type = arg
                        break
            if isinstance(field_type, type) and issubclass(
                field_type, StrEnum
            ):
                enum_mappings[field_name] = {
                    member.name: member.value for member in field_type
                }
        return enum_mappings

    def _convert_enum_names_to_values(self, response: str) -> str:
        """Convert enum names to values in a JSON response.

        Some models return enum names (e.g. 'TOO_COLD')
        instead of values (e.g. 'The temperature is too
        cold.').
        """
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return response

        enum_mappings = self._get_enum_mappings()
        data = self._apply_enum_mappings(data, enum_mappings)
        return json.dumps(data)

    @staticmethod
    def _apply_enum_mappings(
        data: dict | list,
        mappings: dict[str, dict[str, str]],
    ) -> dict | list:
        """Apply enum name-to-value mappings to data.

        Handles single objects, lists, and non-dict values.
        """
        if isinstance(data, list):
            return [
                ResponseParser._apply_enum_mappings(item, mappings)
                for item in data
            ]
        if not isinstance(data, dict):
            return data
        for field_name, mapping in mappings.items():
            if field_name in data and data[field_name] in mapping:
                data[field_name] = mapping[data[field_name]]
        return data
