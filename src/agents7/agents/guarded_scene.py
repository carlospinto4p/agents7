"""Scene-aware conversational agent with context guard.

`GuardedSceneChatter` combines `SceneChatter`'s scene-based
role-play framing with `GuardedChatter`'s response guard.
Every reply is validated by the guard, retried on rejection,
and optionally replaced by the guard's suggested fix — all
while staying in character within the scene.
"""

from dataclasses import dataclass, field

from agents7.agents.guarded import GuardedChatter
from agents7.agents.persona import Scene


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
