"""Scene-aware conversational agent with context guard.

`GuardedSceneChatter` combines scene-based role-play framing
(a `Persona` + `Scene`) with `GuardedChatter`'s response guard.
Every reply is validated by the guard, retried on rejection,
and optionally replaced by the guard's suggested fix — all
while staying in character within the scene.
"""

from dataclasses import dataclass, field, replace

from agents7.agents.context_guard import _PLACEHOLDER
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

    The guard's ``persona_summary`` is enriched with the
    scene's setting, situation, and rules so the guard can
    enforce scene constraints. This happens regardless of how
    the agent is built — both ``from_yaml`` and direct Python
    construction — so the guard always sees the scene context
    its prompt refers to.

    :param scene: The scenario definition.
    """

    scene: Scene = field(default_factory=Scene, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Ensure the guard sees the scene context. The YAML
        # loader already folds scene into the derived summary;
        # this makes direct Python construction match (and is
        # idempotent — the substring check skips re-adding it).
        scene_summary = self.scene.to_summary()
        if scene_summary and scene_summary not in self.guard.persona_summary:
            enriched = f"{self.guard.persona_summary}\n{scene_summary}"
            self._set_attr(
                "guard",
                replace(
                    self.guard,
                    persona_summary=enriched,
                    system_prompt=_PLACEHOLDER,
                ),
            )

    @classmethod
    def from_yaml(cls, file_path: str) -> "GuardedSceneChatter":
        """Create from a YAML configuration file.

        Extends the `GuardedChatter` YAML schema with a
        ``scene`` section.

        :param file_path: Path to the YAML file.
        :return: A `GuardedSceneChatter` instance.
        """
        from agents7.agents.config_loader import (
            load_guarded_scene_yaml_settings,
        )

        return load_guarded_scene_yaml_settings(file_path)
