from dataclasses import dataclass

from agents7.agents.config_loader import load_yaml_settings
from agents7.agents.labeler import Labeler
from agents7.agents.utils import extract_enum_members
from agents7.exceptions import (
    PropertyConfigurationError,
)
from agents7.models.base import Property


@dataclass(frozen=True)
class Detector(Labeler):
    """
    Agent for detecting properties in data.
    This agent is designed to analyze data and detect properties based on
    predefined rules.
    It is a `Labeler` that uses a specific output model to validate and
    structure the output.

    Attributes:
        name (str): The name of the agent.
        version (int): The version of the agent.
        engine (LLMEngine): The LLM engine used by the agent.
        system_prompt (str): The system prompt for the agent.
        output_model (BaseModel): The model used to validate and
            structure the output.
        examples (list[dict], optional): List of examples to include in the
            system prompt. Each example should be a dictionary with 'input'
            and 'output' keys. Defaults to None.

    Note:
        The reason enum will automatically include a NOT_APPLICABLE member
        for fallback classification, even if not explicitly declared.
    """

    output_model: type[Property]

    def __post_init__(self):
        """
        Post-initialization method to validate the agent.
        This method is called after the agent is initialized.

        Validates that output_model is a subclass of Property.
        """
        super().__post_init__()
        if not issubclass(self.output_model, Property):
            raise PropertyConfigurationError(
                "`output_model` must be a subclass of "
                "`agents7.models.Property` for the `Detector` agents."
            )

    @classmethod
    def from_yaml(cls, file_path: str) -> "Detector":
        """
        Create a Detector agent from a YAML configuration file.
        This method reads the configuration from the specified YAML file
        and initializes a Detector agent with the provided settings.

        Args:
            file_path (str): The path to the YAML configuration file.

        Returns:
            Detector: An instance of the Detector agent.
        """
        config = load_yaml_settings(file_path)
        return Detector(**config)

    @property
    def possible_reasons(self) -> dict[str, str]:
        """Possible reasons from the output model's reason enum."""
        return extract_enum_members(self.output_model, "reason")

    @property
    def prompt(self) -> str:
        """Full prompt used by the agent."""
        return self._build_prompt(
            "You need to stick to the reasons defined "
            "in the output model schema above."
        )

    @property
    def prompt_for_batch(self) -> str:
        """Full prompt for batch processing."""
        return self._build_prompt(
            "You need to stick to the reasons defined "
            "in the output model schema above. Respond "
            "with a JSON array of objects, each one "
            "matching the schema above."
        )
