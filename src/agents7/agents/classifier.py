from dataclasses import dataclass

from agents7.agents.detector import Detector
from agents7.agents.utils import (
    extract_enum_members,
    get_examples_str,
    model_class_to_string,
)
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
        output_model (ClassificationProperty): The model used to validate
            and structure the output. Must be a ClassificationProperty subclass.
        examples (list[dict], optional): List of examples to include in the
            system prompt.

    Note:
        Both category and reason enums will automatically include a
        NOT_APPLICABLE member for fallback classification.
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
                "`agents7.models.ClassificationProperty` for `Classifier` agents."
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
        config = cls._load_yaml_settings(file_path)
        return Classifier(**config)

    @property
    def possible_categories(self) -> dict[str, str]:
        """
        Get the possible categories for the output model.

        Returns:
            dict[str, str]: A dictionary of possible categories,
                <category name>: <description>.
        """
        return extract_enum_members(self.output_model, "category")

    @property
    def possible_reasons(self) -> dict[str, str]:
        """
        Get the possible reasons for the output model.

        Returns:
            dict[str, str]: A dictionary of possible reasons,
                <reason name>: <description>.
        """
        return extract_enum_members(self.output_model, "reason")

    @property
    def category_reasons_constraints(self) -> str:
        """
        Get the category-reason constraints as a formatted string.

        Returns:
            str: Formatted string describing valid reasons for each category.
        """
        if not self.output_model.CATEGORY_REASONS:
            return ""

        lines = ["## Valid Reasons per Category"]
        for category, reasons in self.output_model.CATEGORY_REASONS.items():
            reason_names = [r.name for r in reasons]
            lines.append(f"- {category.name}: {', '.join(reason_names)}")
        lines.append("- NOT_APPLICABLE: NOT_APPLICABLE")
        return "\n".join(lines)

    @property
    def prompt(self) -> str:
        """
        Get the full prompt used by the agent.

        The prompt emphasizes category selection and reason justification.
        """
        full_prompt = (
            f"{self.resolved_system_prompt}\n\n"
            f"## Output Model Schema\n\n"
            f"{model_class_to_string(self.output_model)}\n\n"
            f"Classify the input into one of the categories defined "
            f"above. Then provide a reason that justifies your "
            f"classification. Use NOT_APPLICABLE if the input cannot "
            f"be classified."
        ).strip()

        constraints = self.category_reasons_constraints
        if constraints:
            full_prompt = f"{full_prompt}\n\n{constraints}"

        if self.examples:
            examples_str = get_examples_str(self.examples)
            full_prompt = f"{full_prompt}\n\n{examples_str}"

        return full_prompt

    @property
    def prompt_for_batch(self) -> str:
        """
        Get the full prompt for batch processing.
        """
        full_prompt = (
            f"{self.resolved_system_prompt}\n\n"
            f"## Output Model Schema\n\n"
            f"{model_class_to_string(self.output_model)}\n\n"
            f"Classify each input into one of the categories defined above. "
            f"Provide a reason that justifies each classification. "
            f"Use NOT_APPLICABLE if an input cannot be classified. "
            f"Respond with a JSON array of objects, each matching the schema."
        ).strip()

        constraints = self.category_reasons_constraints
        if constraints:
            full_prompt = f"{full_prompt}\n\n{constraints}"

        if self.examples:
            examples_str = get_examples_str(self.examples)
            full_prompt = f"{full_prompt}\n\n{examples_str}"

        return full_prompt
