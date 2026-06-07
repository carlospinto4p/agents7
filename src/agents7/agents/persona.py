"""Persona and scene data classes for role-play agents.

These data classes are leaf-level (no imports from other
agent modules), so they can be imported by `chatter.py`
without creating circular dependencies.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PhysicalTraits:
    """Physical appearance description.

    All fields are optional free-form strings so that
    any level of detail can be provided.
    """

    height: str = ""
    build: str = ""
    hair: str = ""
    eyes: str = ""
    skin: str = ""
    distinguishing_features: str = ""

    def to_text(self) -> str:
        """Render as a human-readable description."""
        parts = []
        if self.height:
            parts.append(f"Height: {self.height}")
        if self.build:
            parts.append(f"Build: {self.build}")
        if self.hair:
            parts.append(f"Hair: {self.hair}")
        if self.eyes:
            parts.append(f"Eyes: {self.eyes}")
        if self.skin:
            parts.append(f"Skin: {self.skin}")
        if self.distinguishing_features:
            parts.append(
                f"Distinguishing features: {self.distinguishing_features}"
            )
        return "\n".join(parts)


@dataclass(frozen=True)
class Persona:
    """Structured character definition for roleplay.

    Provides all the biographical and personality data
    needed to maintain a consistent character across
    conversations.

    Attributes:
        full_name: Character's full name.
        birthdate: Date of birth (free-form string,
            e.g. ``"1990-12-04"``).
        nationality: Country or cultural background.
        occupation: Current job or role.
        physical: Physical appearance traits.
        personality: List of personality adjectives
            (e.g. ``["curious", "empathetic"]``).
        backstory: Brief biography or life story.
        interests: List of hobbies or interests.
        speaking_style: How the character speaks
            (e.g. ``"casual and direct"``).
        language: Language the character speaks in
            (e.g. ``"Spanish"``). When set, the agent
            is instructed to ALWAYS respond in this
            language regardless of the user's language.
        extra: Additional key-value pairs for any
            other character details.
    """

    full_name: str
    birthdate: str = ""
    nationality: str = ""
    occupation: str = ""
    physical: PhysicalTraits = field(
        default_factory=PhysicalTraits,
    )
    personality: list[str] = field(default_factory=list)
    backstory: str = ""
    interests: list[str] = field(default_factory=list)
    speaking_style: str = ""
    language: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    def to_prompt(self) -> str:
        """Render the persona as a system prompt section.

        :return: Multi-line string describing the character.
        """
        lines = [
            f"You are {self.full_name}.",
            "",
            "## Character Profile",
        ]
        if self.birthdate:
            lines.append(f"- Birthdate: {self.birthdate}")
        if self.nationality:
            lines.append(f"- Nationality: {self.nationality}")
        if self.occupation:
            lines.append(f"- Occupation: {self.occupation}")
        if self.language:
            lines.append(f"- Language: {self.language}")

        physical_text = self.physical.to_text()
        if physical_text:
            lines.append("")
            lines.append("## Physical Appearance")
            lines.append(physical_text)

        if self.personality:
            lines.append("")
            lines.append("## Personality")
            lines.append(", ".join(self.personality))

        if self.backstory:
            lines.append("")
            lines.append("## Backstory")
            lines.append(self.backstory)

        if self.interests:
            lines.append("")
            lines.append("## Interests")
            lines.append(", ".join(self.interests))

        if self.speaking_style:
            lines.append("")
            lines.append("## Speaking Style")
            lines.append(self.speaking_style)

        if self.extra:
            lines.append("")
            lines.append("## Additional Details")
            for key, value in self.extra.items():
                lines.append(f"- {key}: {value}")

        lines.append("")
        lines.append("## Instructions")
        instruction = (
            "Stay in character at all times. You are a "
            "real person, not an assistant. Never offer "
            "help, never ask what you can do for someone, "
            "and never use a servile or customer-service "
            "tone. Respond the way your character naturally "
            "would — with their own mood, opinions, and "
            "agenda. Answer questions about yourself using "
            "ONLY the information in your character "
            "profile. If asked about something not covered "
            "in your profile, improvise consistently with "
            "your established personality and backstory. "
            "Never break character or reveal that you "
            "are an AI."
        )
        if self.language:
            instruction += (
                f" ALWAYS respond in {self.language}, "
                f"regardless of the language the user "
                f"writes in."
            )
        lines.append(instruction)

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict) -> "Persona":
        """Create a Persona from a dictionary.

        Accepts nested ``physical`` dict that maps to
        ``PhysicalTraits`` fields.

        :param data: Dictionary with persona fields.
        :return: A Persona instance.
        """
        physical_data = data.pop("physical", {})
        if isinstance(physical_data, dict):
            physical = PhysicalTraits(**physical_data)
        else:
            physical = PhysicalTraits()

        return cls(physical=physical, **data)


@dataclass(frozen=True)
class Scene:
    """Structured scenario definition for role-play.

    Describes the context in which a character operates:
    where they are, what is happening, and what rules
    govern the interaction.

    :param setting: Physical location and environment
        (e.g. ``"A medieval tavern at night"``).
    :param situation: What is currently happening
        (e.g. ``"The tavern is about to close"``).
    :param time_period: Historical or fictional era
        (e.g. ``"Medieval Europe, 14th century"``).
    :param rules: Behavioral constraints that override
        normal responses (e.g. ``["You don't know about
        modern technology"]``).
    :param other_characters: Other characters present in
        the scene (e.g. ``["A drunk sailor at the bar",
        "A hooded stranger in the corner"]``).
    :param goal: The character's current objective
        (e.g. ``"Close the tavern and get everyone out"``).
    :param extra: Additional key-value pairs for any
        other scene details.
    """

    setting: str = ""
    situation: str = ""
    time_period: str = ""
    rules: list[str] = field(default_factory=list)
    other_characters: list[str] = field(
        default_factory=list,
    )
    goal: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    def to_summary(self) -> str:
        """Render a compact scene summary for the guard prompt.

        Includes setting, situation, and rules so the guard
        can judge scene-specific off-topic responses and rule
        violations. Returns an empty string when none of those
        fields are set. Shared by `_derive_persona_summary`
        (YAML path) and `GuardedSceneChatter.__post_init__`
        (Python path) so both produce identical text.

        :return: Multi-line summary string, or ``""``.
        """
        lines: list[str] = []
        if self.setting:
            lines.append(f"Setting: {self.setting.strip()}")
        if self.situation:
            lines.append(f"Situation: {self.situation.strip()}")
        if self.rules:
            lines.append("Scene rules:")
            for rule in self.rules:
                lines.append(f"  - {rule}")
        return "\n".join(lines)

    def to_prompt(self) -> str:
        """Render the scene as a system prompt section.

        :return: Multi-line string describing the scenario.
        """
        lines = ["## Scene"]
        if self.setting:
            lines.append(f"- Setting: {self.setting}")
        if self.situation:
            lines.append(f"- Situation: {self.situation}")
        if self.time_period:
            lines.append(f"- Time period: {self.time_period}")
        if self.goal:
            lines.append(f"- Your goal: {self.goal}")

        if self.other_characters:
            lines.append("")
            lines.append("## Other Characters Present")
            for char in self.other_characters:
                lines.append(f"- {char}")

        if self.rules:
            lines.append("")
            lines.append("## Scene Rules")
            lines.append("You MUST follow these rules at all times:")
            for rule in self.rules:
                lines.append(f"- {rule}")

        if self.extra:
            lines.append("")
            lines.append("## Scene Details")
            for key, value in self.extra.items():
                lines.append(f"- {key}: {value}")

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        """Create a Scene from a dictionary.

        :param data: Dictionary with scene fields.
        :return: A Scene instance.
        """
        return cls(**data)


def _build_scene_system_prompt(
    persona: Persona,
    scene: Scene,
    extra_instructions: str = "",
) -> str:
    """Build the full role-play system prompt.

    Combines persona and scene into a single prompt string.
    Shared by `Chatter._build_plugin_prompt()` and
    `GuardedSceneChatter` when both plugins are set.

    :param persona: The character definition.
    :param scene: The scenario definition.
    :param extra_instructions: Optional extra guidance.
    :return: The complete system prompt string.
    """
    lines = [
        "This is a role-play scenario. You are playing "
        "a character in a specific context. Stay in "
        "character and in context at all times.",
        "",
        scene.to_prompt(),
        "",
        persona.to_prompt(),
    ]
    prompt = "\n".join(lines)
    if extra_instructions:
        prompt = f"{prompt}\n\n## Extra Instructions\n{extra_instructions}"
    return prompt
