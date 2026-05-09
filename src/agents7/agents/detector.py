from dataclasses import dataclass
from typing import Self

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
        for fallback classification, unless the output model sets
        ``NOT_APPLICABLE = False``.
    """

    output_model: type[Property]

    def _validate_output_model(self) -> None:
        """Validate that output_model is a Property subclass.

        Subclasses can override this to accept a different model
        base class (e.g. Classifier accepts ClassificationProperty).
        """
        if not issubclass(self.output_model, Property):
            raise PropertyConfigurationError(
                "`output_model` must be a subclass of "
                "`agents7.models.Property` for the `Detector` agents."
            )

    def __post_init__(self) -> None:
        """
        Post-initialization method to validate the agent.
        This method is called after the agent is initialized.
        """
        super().__post_init__()
        object.__setattr__(
            self,
            "_possible_reasons",
            extract_enum_members(self.output_model, "reason"),
        )
        object.__setattr__(
            self,
            "_prompt",
            self._build_prompt(
                self._agent_instruction(),
                extra_sections=self._extra_prompt_sections(),
            ),
        )
        object.__setattr__(
            self,
            "_prompt_for_batch",
            self._build_prompt(
                self._batch_instruction(),
                extra_sections=self._extra_prompt_sections(),
            ),
        )

    @classmethod
    def from_yaml(cls, file_path: str) -> Self:
        """Create an agent instance from a YAML configuration file.

        :param file_path: Path to the YAML configuration file.
        :return: An instance of the calling class.
        """
        config = load_yaml_settings(file_path)
        return cls(**config)

    @property
    def possible_reasons(self) -> dict[str, str]:
        """Possible reasons from the output model's reason enum."""
        return self._possible_reasons

    @property
    def _not_applicable_guidance(self) -> str:
        """Guidance section for NOT_APPLICABLE classification."""
        return (
            "## NOT_APPLICABLE Guidance\n\n"
            "Use NOT_APPLICABLE when the input has no "
            "connection to the property being detected. "
            "Do not infer, speculate, or make assumptions "
            "beyond what is explicitly stated or clearly "
            "implied. If the information is not present, "
            "classify as NOT_APPLICABLE rather than "
            "guessing."
        )

    @property
    def _has_not_applicable(self) -> bool:
        """Whether the output model includes NOT_APPLICABLE."""
        return getattr(self.output_model, "NOT_APPLICABLE", True)

    def _agent_instruction(self) -> str:
        """Single-input instruction text for the prompt."""
        return (
            "You need to stick to the reasons defined "
            "in the output model schema above."
        )

    def _batch_instruction(self) -> str:
        """Batch instruction text for the prompt."""
        return (
            "You will receive multiple inputs. Process "
            "each one independently — apply the exact "
            "same analysis as for a single input. Do not "
            "default to a safe answer just because there "
            "are multiple items. Stick to the reasons "
            "defined in the output model schema above. "
            "Respond with a JSON array of objects, each "
            "matching the schema above."
        )

    def _extra_prompt_sections(self) -> list[str]:
        """Extra sections appended to the prompt."""
        extra = []
        if self._has_not_applicable:
            extra.append(self._not_applicable_guidance)
        return extra

    @property
    def prompt(self) -> str:
        """Full prompt used by the agent."""
        return self._prompt

    @property
    def prompt_for_batch(self) -> str:
        """Full prompt for batch processing."""
        return self._prompt_for_batch
