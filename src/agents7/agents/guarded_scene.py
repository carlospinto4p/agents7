"""Scene-aware conversational agent with context guard.

`GuardedSceneChatter` combines scene-based role-play framing
(a `Persona` + `Scene`) with `GuardedChatter`'s response guard.
Every reply is validated by the guard, retried on rejection,
and optionally replaced by the guard's suggested fix — all
while staying in character within the scene.
"""

from dataclasses import dataclass, field, replace

from agents7.agents.context_guard import _PLACEHOLDER, ContextGuard
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

    A scene that names a `context_params` value is checked against
    the value the agent has when the reply is written, not the one
    it had when it was built — see `current_guard`.

    :param scene: The scenario definition.
    """

    scene: Scene = field(default_factory=Scene, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Ensure the guard sees the scene context. The YAML
        # loader already folds scene into the derived summary;
        # this makes direct Python construction match (and is
        # idempotent — the substring check skips re-adding it).
        #
        # The guard is its own agent and has no persona, so it
        # cannot resolve `$character` itself. A scene written to
        # be character-agnostic would otherwise reach it with the
        # placeholder still in the text, and the guard would be
        # judging replies against a character called `$character`.
        #
        # Only the names the config fixes are filled here. A
        # `context_params` or `prompt_variants` name stays a
        # placeholder and `current_guard` fills it per reply:
        # filled here, the guard kept the defaults for good, and
        # rejected a reply for using the nickname `set_context`
        # had supplied.
        scene_summary = self._fixed(self.scene.to_summary())
        summary = self._fixed(self.guard.persona_summary)
        if scene_summary and scene_summary not in summary:
            summary = f"{summary}\n{scene_summary}"
        if summary != self.guard.persona_summary:
            self._set_attr(
                "guard",
                replace(
                    self.guard,
                    persona_summary=summary,
                    system_prompt=_PLACEHOLDER,
                ),
            )

    def _fixed(self, text: str) -> str:
        """Fill only the placeholders the config fixes."""
        return self._substitute(text, with_variants=False, with_context=False)

    def current_guard(self) -> ContextGuard:
        """The guard, judging by the context the agent has now.

        `guard` keeps every `context_params` and `prompt_variants`
        name as a placeholder, so it is the same for every
        conversation. This fills them with what the agent's own
        prompt carries at this moment, so the two cannot disagree
        about the time of day or about a nickname.

        :return: `guard` when it names no such value, otherwise a
            guard built for the current ones.
        """
        summary = self.substitute(self.guard.persona_summary)
        if summary == self.guard.persona_summary:
            return self.guard
        resolved = getattr(self, "_resolved_guard", None)
        if resolved is None or resolved.persona_summary != summary:
            resolved = replace(
                self.guard,
                persona_summary=summary,
                system_prompt=_PLACEHOLDER,
            )
            self._set_attr("_resolved_guard", resolved)
        return resolved

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
