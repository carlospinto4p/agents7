"""Backward-compatible re-exports of persona/scene dataclasses.

`PersonalityChatter` and `SceneChatter` were removed in v4.0.0:
the base `Chatter` accepts `persona` and `scene` as optional
composition plugins (auto-generating the system prompt), so the
enforcement-via-subclass idiom was redundant. Use::

    from agents7.agents import Chatter, Persona, Scene

    chatter = Chatter(
        name="david", version=1, engine=engine,
        system_prompt="", persona=persona, scene=scene,
    )

or `Chatter.from_yaml(...)` with `persona:` / `scene:` sections.

This module remains only to keep
``from agents7.agents.personality import Persona, Scene,
PhysicalTraits`` working.
"""

from agents7.agents.persona import (
    Persona,
    PhysicalTraits,
    Scene,
)

__all__ = [
    "Persona",
    "PhysicalTraits",
    "Scene",
]
