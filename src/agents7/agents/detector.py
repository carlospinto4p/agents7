from enum import StrEnum

from agents7.agents.labeler import Labeler
from agents7.models.base import Property


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
    """

    output_model: type[Property]

    def __post_init__(self):
        """
        Post-initialization method to validate the agent.
        This method is called after the agent is initialized.
        """
        super().__post_init__()
        if not issubclass(self.output_model, Property):
            raise TypeError(
                "`output_model` must be a subclass of `Property` for the "
                "Detector agents."
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
        config = cls._load_yaml_settings(file_path)

        detector = Detector(**config)
        reasons_str = "\n".join(
            f"- {k}: {v}" for k, v in detector.get_possible_reasons().items()
        )
        detector.system_prompt += (
            "\n\nReturn only valid JSON, no explanations, no extra text."
            f"\n\n## Reasons\n\n{reasons_str}"
        )
        return detector

    def get_possible_reasons(self) -> dict[str, str]:
        """
        Get the possible reasons for the output model.
        This method retrieves the possible reasons defined in the
        `output_model.reason` StrEnum class values.

        Returns:
            dict[str, str]: A dictionary of possible reasons.
        """
        reasons_enum = self.output_model.model_fields["reason"].annotation
        if not issubclass(reasons_enum, StrEnum):
            raise TypeError(
                f"`reason` field in the output_model must be a `StrEnum`. "
                f"Got `{reasons_enum}` instead."
            )
        return {k: str(v) for k, v in reasons_enum.__members__.items()}
