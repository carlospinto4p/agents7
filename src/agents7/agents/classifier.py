from dataclasses import dataclass

from agents7.agents.config_loader import load_yaml_settings
from agents7.agents.detector import Detector
from agents7.agents.utils import extract_enum_members
from agents7.exceptions import PropertyConfigurationError
from agents7.models.base import ClassificationProperty


@dataclass(frozen=True)
class Classifier(Detector):
    """
    Agent for classifying data into predefined categories.

    This agent extends Detector to specialize in classification tasks where
    the output is a category (from a predefined enum) along with a reason
    justifying the classification.

    Attributes:
        name (str): The name of the agent.
        version (int): The version of the agent.
        engine (LLMEngine): The LLM engine used by the agent.
        system_prompt (str): The system prompt for the agent.
        output_model (ClassificationProperty): The model used to
            validate and structure the output. Must be a
            ClassificationProperty subclass.
        examples (list[dict], optional): List of examples to include in the
            system prompt.

    Note:
        Both category and reason enums will automatically include a
        NOT_APPLICABLE member for fallback classification, unless
        the output model sets ``NOT_APPLICABLE = False``.
    """

    output_model: type[ClassificationProperty]

    def __post_init__(self) -> None:
        """
        Post-initialization method to validate the agent.

        Validates that output_model is a subclass of ClassificationProperty.
        """
        # Skip Detector's validation: ClassificationProperty is not a
        # Property subclass, so Detector.__post_init__ would reject it.
        super(Detector, self).__post_init__()
        if not issubclass(self.output_model, ClassificationProperty):
            raise PropertyConfigurationError(
                "`output_model` must be a subclass of "
                "`agents7.models.ClassificationProperty`"
                " for `Classifier` agents."
            )
        object.__setattr__(
            self,
            "_possible_categories",
            extract_enum_members(self.output_model, "category"),
        )
        object.__setattr__(
            self,
            "_possible_reasons",
            extract_enum_members(self.output_model, "reason"),
        )

    @classmethod
    def from_yaml(cls, file_path: str) -> "Classifier":
        """Create a Classifier agent from a YAML configuration file.

        :param file_path: Path to the YAML configuration file.
        :return: A Classifier instance.
        """
        config = load_yaml_settings(file_path)
        return Classifier(**config)

    @property
    def possible_categories(self) -> dict[str, str]:
        """Possible categories from the output model's enum."""
        return self._possible_categories

    @property
    def possible_reasons(self) -> dict[str, str]:
        """Possible reasons from the output model's reason enum."""
        return self._possible_reasons

    @property
    def category_reasons_constraints(self) -> str:
        """Category-reason constraints as a formatted string."""
        if not self.output_model.CATEGORY_REASONS:
            return ""

        lines = ["## Valid Reasons per Category"]
        for category, reasons in self.output_model.CATEGORY_REASONS.items():
            reason_names = [r.name for r in reasons]
            lines.append(f"- {category.name}: {', '.join(reason_names)}")
        if self._has_not_applicable:
            lines.append("- NOT_APPLICABLE: NOT_APPLICABLE")
        return "\n".join(lines)

    def _na_suffix(self, text: str) -> str:
        """Append NOT_APPLICABLE instruction if enabled."""
        if self._has_not_applicable:
            return text
        return ""

    def _agent_instruction(self) -> str:
        """Single-input instruction for classification."""
        return (
            "Classify the input into one of the "
            "categories defined above. Then provide a "
            "reason that justifies your classification."
            + self._na_suffix(
                " Use NOT_APPLICABLE if the input cannot be classified."
            )
        )

    def _batch_instruction(self) -> str:
        """Batch instruction for classification."""
        return (
            "You will receive multiple inputs. Classify "
            "EACH input independently into one of the "
            "categories defined above. Apply the same "
            "rules to each item as you would for a single "
            "input — do not be more conservative just "
            "because there are multiple items.\n\n"
            "IMPORTANT: Choose the reason for EACH item "
            "individually based on what you actually found "
            "in that specific input. Different inputs will "
            "have different reasons — do NOT repeat the "
            "same reason for every item."
            + self._na_suffix(
                " Use NOT_APPLICABLE if an input cannot be classified."
            )
            + " Respond with a JSON array of "
            "objects, each matching the schema."
        )

    def _extra_prompt_sections(self) -> list[str]:
        """Extra sections for classification prompts."""
        extra = [self.category_reasons_constraints]
        if self._has_not_applicable:
            extra.append(self._not_applicable_guidance)
        return extra
