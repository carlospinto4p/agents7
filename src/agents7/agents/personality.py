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
from agents7.agents.persona import (  # re-exported for backward compat
    Persona,
    PhysicalTraits,
    Scene,
)

__all__ = [
    "Persona",
    "PersonalityChatter",
    "PhysicalTraits",
    "Scene",
]


@dataclass(frozen=True)
class PersonalityChatter(Chatter):
    """Chatter with a predefined character persona.

    Extends `Chatter` by requiring a `Persona`.  The system
    prompt is auto-generated from the persona in
    `Chatter.__post_init__()`, so do not set `system_prompt`
    manually.  Use ``extra_instructions`` (inherited from
    `Chatter`) to append additional guidance.

    Attributes:
        persona (Persona): The character definition.
    """

    persona: Persona = field(
        default_factory=lambda: Persona(full_name="Agent"),
        kw_only=True,
    )

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

    :param file_path: Path to the YAML file.
    :return: A PersonalityChatter instance.
    """

    from agents7.agents.chatter_config_loader import (
        validate_chatter_yaml_config,
    )
    from agents7.agents.config_loader import (
        _resolve_inheritance,
        build_engine,
        cast_optional_fields,
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
