"""Scene-aware conversational agent.

`SceneChatter` extends `PersonalityChatter` with a structured
`Scene` that places the character in a specific context:
a location, situation, time period, and behavioral rules.

The system prompt wraps everything in a role-play frame so
the agent stays both in character and in context.

Usage::

    from agents7.agents.personality import (
        Persona, Scene,
    )
    from agents7.agents.scene import SceneChatter

    persona = Persona(
        full_name="Old Thomas",
        occupation="tavern keeper",
    )
    scene = Scene(
        setting="A medieval tavern at night",
        situation="The tavern is about to close.",
        time_period="Medieval Europe, 14th century",
        rules=[
            "You have no knowledge of anything after "
            "the 14th century.",
            "You are suspicious of strangers.",
        ],
        goal="Close the tavern and get everyone out.",
    )
    agent = SceneChatter(
        name="thomas",
        version=1,
        engine=engine,
        persona=persona,
        scene=scene,
    )
    response = agent.chat("Good evening!")
"""

from dataclasses import dataclass, field
from typing import Any

from agents7.agents.personality import (
    PersonalityChatter,
    Scene,
)


@dataclass(frozen=True)
class SceneChatter(PersonalityChatter):
    """PersonalityChatter placed in a specific scenario.

    Extends `PersonalityChatter` by adding a `Scene` that
    frames the conversation as a role-play with explicit
    context, rules, and goals.  The system prompt is
    auto-generated from both the persona and the scene.

    The ``system_prompt`` field is auto-generated — do not
    set it manually.  Use ``extra_instructions`` for any
    additional guidance beyond the persona and scene.

    :param scene: The scenario definition.
    """

    scene: Scene = field(
        default_factory=Scene,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        prompt = self._build_scene_prompt()
        if self.extra_instructions:
            prompt = (
                f"{prompt}\n\n"
                f"## Extra Instructions\n"
                f"{self.extra_instructions}"
            )
        object.__setattr__(self, "system_prompt", prompt)
        # Skip PersonalityChatter.__post_init__ (which
        # would overwrite system_prompt) and call
        # Chatter.__post_init__ directly for memory loading.
        from agents7.agents.chatter import Chatter

        Chatter.__post_init__(self)

    def _build_scene_prompt(self) -> str:
        """Build the full role-play system prompt.

        Combines a role-play framing preamble, the scene
        description, and the character profile into a
        single coherent prompt.

        :return: The complete system prompt string.
        """
        persona_prompt = self.persona.to_prompt()
        scene_prompt = self.scene.to_prompt()

        lines = [
            "This is a role-play scenario. You are playing "
            "a character in a specific context. Stay in "
            "character and in context at all times.",
            "",
            scene_prompt,
            "",
            persona_prompt,
        ]
        return "\n".join(lines)

    @classmethod
    def from_yaml(
        cls,
        file_path: str,
    ) -> "SceneChatter":
        """Create from a YAML configuration file.

        Extends the PersonalityChatter YAML schema with a
        ``scene`` section::

            name: thomas
            version: 1
            engine:
                provider: ollama
                model: qwen3:4b
            persona:
                full_name: Old Thomas
                occupation: tavern keeper
            scene:
                setting: A medieval tavern at night
                situation: The tavern is about to close.
                time_period: Medieval Europe, 14th century
                rules:
                    - You have no knowledge of anything
                      after the 14th century.
                    - You are suspicious of strangers.
                goal: Close the tavern and get everyone out.

        :param file_path: Path to the YAML file.
        :return: A SceneChatter instance.
        """
        return _load_scene_yaml(file_path)


def _load_scene_yaml(
    file_path: str,
) -> SceneChatter:
    """Load a SceneChatter from a YAML file.

    :param file_path: Path to the YAML file.
    :return: A SceneChatter instance.
    """
    from agents7.agents.config_loader import (
        _load_base_chatter_config,
        build_engine,
        cast_optional_fields,
    )
    from agents7.agents.memory import (
        MemoryConfig,
        create_memory,
    )

    raw, persona, extra_instructions = _load_base_chatter_config(
        file_path,
    )
    scene = Scene.from_dict(dict(raw.get("scene", {})))

    config: dict[str, Any] = {
        "name": raw["name"],
        "version": raw["version"],
        "engine": build_engine(raw["engine"]),
        "system_prompt": "",
        "persona": persona,
        "scene": scene,
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

    return SceneChatter(**config)
