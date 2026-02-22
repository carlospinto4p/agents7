from dataclasses import dataclass, field


@dataclass
class FieldDefinition:
    """A single field in a Labeler output model."""

    name: str
    field_type: str = "str"
    optional: bool = False
    description: str = ""


@dataclass
class EnumMember:
    """A single member of a StrEnum (category or reason)."""

    name: str
    description: str = ""


@dataclass
class AgentBuilderState:
    """Full state for the Agent Builder UI workflow.

    Captures every user choice across the 5-step wizard so that
    backend modules (codegen, evaluation, helpers) can operate
    without any Streamlit dependency.
    """

    # Step 1 — Define
    agent_type: str = "classifier"
    agent_name: str = ""
    categories: list[EnumMember] = field(default_factory=list)
    reasons: list[EnumMember] = field(default_factory=list)
    category_reasons: dict[str, list[str]] = field(default_factory=dict)
    value_type: str = "float"
    fields: list[FieldDefinition] = field(default_factory=list)

    # Step 2 — Configure
    model_name: str = "qwen3"
    system_prompt: str = ""
    examples: list[dict] = field(default_factory=list)

    # Discovery
    intent: str = ""
    sample_data: list[dict] = field(default_factory=list)

    # Step 4 — Evaluate
    eval_data: list[dict] = field(default_factory=list)

    @property
    def model_class_name(self) -> str:
        """PascalCase class name derived from agent_name."""
        if not self.agent_name:
            return "Output"
        parts = self.agent_name.replace("-", " ").replace("_", " ").split()
        return "".join(w.capitalize() for w in parts)

    @property
    def category_enum_name(self) -> str:
        return f"{self.model_class_name}Category"

    @property
    def reason_enum_name(self) -> str:
        return f"{self.model_class_name}Reason"
