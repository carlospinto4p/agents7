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
