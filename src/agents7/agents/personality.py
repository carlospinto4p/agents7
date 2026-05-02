"""Personality-driven conversational agent.

`PersonalityChatter` extends `Chatter` with a structured
`Persona` that defines biographical data, physical traits,
and personality characteristics. The system prompt is
auto-generated from the persona so the agent stays in
character across all conversations.

Usage::

    from agents7.agents.personality import (
        Persona, PersonalityChatter,
    )

    persona = Persona(
        full_name="David García",
        birthdate="1990-12-04",
        nationality="Spanish",
        occupation="journalist",
        physical=PhysicalTraits(
            height="178 cm",
            build="medium",
            hair="dark brown, short",
            eyes="brown",
        ),
        personality=["curious", "empathetic", "direct"],
        backstory="Grew up in Madrid, studied journalism.",
        interests=["politics", "football", "cooking"],
    )
    agent = PersonalityChatter(
        name="david",
        version=1,
        engine=engine,
        persona=persona,
    )
    response = agent.chat("What's your name?")
"""

from dataclasses import dataclass, field

from agents7.agents.chatter import Chatter


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


@dataclass(frozen=True)
class PersonalityChatter(Chatter):
    """Chatter with a predefined character persona.

    Extends `Chatter` by auto-generating the system prompt
    from a structured `Persona`. The persona data ensures
    consistent answers to biographical, physical, and
    personality questions.

    The ``system_prompt`` field is auto-generated from the
    persona — do not set it manually. If you need to add
    extra instructions, use ``extra_instructions``.

    Attributes:
        persona (Persona): The character definition.
        extra_instructions (str): Additional instructions
            appended after the persona prompt. Defaults
            to empty string.
    """

    persona: Persona = field(
        default_factory=lambda: Persona(full_name="Agent"),
        kw_only=True,
    )
    extra_instructions: str = field(
        default="",
        kw_only=True,
    )

    def _build_system_prompt(self) -> str:
        """Build the system prompt from the persona and extra instructions.

        Subclasses override this to change how the prompt is constructed
        without having to duplicate the ``__post_init__`` plumbing.

        :return: The fully assembled system prompt string.
        """
        prompt = self.persona.to_prompt()
        if self.extra_instructions:
            prompt = (
                f"{prompt}\n\n"
                f"## Extra Instructions\n"
                f"{self.extra_instructions}"
            )
        return prompt

    def __post_init__(self) -> None:
        object.__setattr__(self, "system_prompt", self._build_system_prompt())
        super().__post_init__()

    @classmethod
    def from_yaml(
        cls,
        file_path: str,
    ) -> "PersonalityChatter":
        """Create from a YAML configuration file.

        The YAML file extends the Chatter schema with a
        ``persona`` section::

            name: david
            version: 1
            engine:
                provider: ollama
                model: qwen3:4b
            persona:
                full_name: David García
                birthdate: "1990-12-04"
                nationality: Spanish
                occupation: journalist
                physical:
                    height: 178 cm
                    build: medium
                    hair: dark brown, short
                    eyes: brown
                personality:
                    - curious
                    - empathetic
                    - direct
                backstory: |
                    Grew up in Madrid.
                interests:
                    - politics
                    - football
            extra_instructions: |
                Always respond in Spanish.

        :param file_path: Path to the YAML file.
        :return: A PersonalityChatter instance.
        """
        return _load_personality_yaml(file_path)


def _load_personality_yaml(
    file_path: str,
) -> PersonalityChatter:
    """Load a PersonalityChatter from a YAML file.

    Injects a placeholder ``system_prompt`` so the
    Chatter YAML validator passes, then extracts the
    ``persona`` section and builds the agent.

    :param file_path: Path to the YAML file.
    :return: A PersonalityChatter instance.
    """

    from agents7.agents.config_loader import (
        _resolve_inheritance,
        build_engine,
        cast_optional_fields,
        validate_chatter_yaml_config,
    )
    from agents7.agents.memory import (
        MemoryConfig,
        create_memory,
    )

    raw = _resolve_inheritance(file_path)

    # Inject placeholder so validator passes
    raw.setdefault("system_prompt", "(auto-generated)")

    validate_chatter_yaml_config(raw, file_path)

    persona_data = raw.get("persona", {})
    persona = Persona.from_dict(dict(persona_data))

    extra_instructions = raw.get(
        "extra_instructions",
        "",
    )
    if extra_instructions:
        extra_instructions = str(extra_instructions).strip()

    config: dict = {
        "name": raw["name"],
        "version": raw["version"],
        "engine": build_engine(raw["engine"]),
        "system_prompt": "",
        "persona": persona,
        "extra_instructions": extra_instructions,
    }

    cast_optional_fields(
        raw,
        config,
        {
            "temperature": float,
            "think": bool,
            "max_history": int,
            "max_turns": int,
            "typo_rate": float,
            "conversation_id": str,
        },
    )
    if memory_settings := raw.get("memory"):
        memory_config = MemoryConfig(
            backend=memory_settings.get("backend", "json"),
            path=memory_settings.get("path", ""),
        )
        config["memory"] = create_memory(memory_config)

    return PersonalityChatter(**config)
