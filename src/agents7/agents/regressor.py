from dataclasses import dataclass

from agents7.agents.detector import Detector
from agents7.models.base import Property


@dataclass(frozen=True)
class Regressor(Detector):
    """
    Agent for extracting numeric values from data.

    This agent extends Detector to specialize in regression tasks where
    the output is a numeric value along with a reason classifying the
    extracted value.

    Attributes:
        name (str): The name of the agent.
        version (int): The version of the agent.
        engine (LLMEngine): The LLM engine used by the agent.
        system_prompt (str): The system prompt for the agent.
        output_model (Property[T]): The model used to validate and structure
            the output. Must be a Property subclass with a numeric type.
        examples (list[dict], optional): List of examples to include in the
            system prompt.

    Note:
        The reason enum will automatically include a NOT_APPLICABLE member
        for cases where no value can be extracted from the input.
    """

    output_model: type[Property]

    @classmethod
    def from_yaml(cls, file_path: str) -> "Regressor":
        """
        Create a Regressor agent from a YAML configuration file.

        Args:
            file_path (str): The path to the YAML configuration file.

        Returns:
            Regressor: An instance of the Regressor agent.
        """
        config = cls._load_yaml_settings(file_path)
        return Regressor(**config)

    @property
    def prompt(self) -> str:
        """Get the full prompt used by the agent."""
        return self._build_prompt(
            "Extract the numeric value from the input. "
            "Then classify the extracted value using one "
            "of the reasons defined above. Use "
            "NOT_APPLICABLE if no value can be extracted."
        )

    @property
    def prompt_for_batch(self) -> str:
        """Get the full prompt for batch processing."""
        return self._build_prompt(
            "Extract the numeric value from each input. "
            "Classify each extracted value using the "
            "reasons defined above. Use NOT_APPLICABLE "
            "if no value can be extracted. Respond with "
            "a JSON array of objects, each matching the "
            "schema."
        )
