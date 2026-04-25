"""Scene-aware conversational agent with context guard.

`GuardedSceneChatter` combines `SceneChatter`'s scene-based
role-play framing with `GuardedChatter`'s response guard.
Every reply is validated by the guard, retried on rejection,
and optionally replaced by the guard's suggested fix — all
while staying in character within the scene.
"""

from dataclasses import dataclass, field

from agents7.agents.guarded import GuardedChatter
from agents7.agents.personality import Scene


@dataclass(frozen=True)
class GuardedSceneChatter(GuardedChatter):
    """GuardedChatter placed in a specific scenario.

    Extends `GuardedChatter` by adding a `Scene` that frames
    the conversation as a role-play with explicit context,
    rules, and goals. The system prompt is auto-generated
    from both the persona and the scene.

    The ``system_prompt`` field is auto-generated — do not
    set it manually. Use ``extra_instructions`` for any
    additional guidance beyond the persona and scene.

    :param scene: The scenario definition.
    """

    scene: Scene = field(default_factory=Scene, kw_only=True)

    def __post_init__(self) -> None:
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
        prompt = "\n".join(lines)
        if self.extra_instructions:
            prompt = (
                f"{prompt}\n\n"
                f"## Extra Instructions\n"
                f"{self.extra_instructions}"
            )
        object.__setattr__(self, "system_prompt", prompt)
        # Skip GuardedChatter / PersonalityChatter __post_init__
        # (they would overwrite system_prompt) and call
        # Chatter.__post_init__ directly for memory loading.
        from agents7.agents.chatter import Chatter

        Chatter.__post_init__(self)

    @classmethod
    def from_yaml(cls, file_path: str) -> "GuardedSceneChatter":
        """Create from a YAML configuration file.

        Extends the `GuardedChatter` YAML schema with a
        ``scene`` section (same structure as `SceneChatter`).

        :param file_path: Path to the YAML file.
        :return: A `GuardedSceneChatter` instance.
        """
        from agents7.agents.config_loader import (
            load_guarded_scene_yaml_settings,
        )

        return load_guarded_scene_yaml_settings(file_path)
