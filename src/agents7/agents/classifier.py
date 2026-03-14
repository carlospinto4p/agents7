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

    def __post_init__(self):
        """
        Post-initialization method to validate the agent.

        Validates that output_model is a subclass of ClassificationProperty.
        """
        # Skip Detector's validation, call Labeler's directly
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
        """
        Create a Classifier agent from a YAML configuration file.

        Args:
            file_path (str): The path to the YAML configuration file.

        Returns:
            Classifier: An instance of the Classifier agent.
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

    @property
    def prompt(self) -> str:
        """Get the full prompt used by the agent."""
        na_suffix = (
            " Use NOT_APPLICABLE if the input cannot be classified."
            if self._has_not_applicable
            else ""
        )
        extra = [self.category_reasons_constraints]
        if self._has_not_applicable:
            extra.append(self._not_applicable_guidance)
        return self._build_prompt(
            "Classify the input into one of the "
            "categories defined above. Then provide a "
            "reason that justifies your classification." + na_suffix,
            extra_sections=extra,
        )

    @property
    def prompt_for_batch(self) -> str:
        """Get the full prompt for batch processing."""
        na_suffix = (
            " Use NOT_APPLICABLE if an input cannot be classified."
            if self._has_not_applicable
            else ""
        )
        extra = [self.category_reasons_constraints]
        if self._has_not_applicable:
            extra.append(self._not_applicable_guidance)
        return self._build_prompt(
            "Classify each input into one of the "
            "categories defined above. Provide a reason "
            "that justifies each classification."
            + na_suffix
            + " Respond with a JSON array of "
            "objects, each matching the schema.",
            extra_sections=extra,
        )
