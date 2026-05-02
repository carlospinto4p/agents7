"""YAML configuration loading for Chatter-family agents."""

from typing import Any

from agents7.agents.config_loader import (
    _resolve_inheritance,
    build_engine,
    cast_optional_fields,
)
from agents7.agents.utils import import_from_path
from agents7.exceptions import AgentConfigurationError


def validate_chatter_yaml_config(
    config: dict,
    file_path: str,
) -> None:
    """Validate mandatory keys in a Chatter YAML config.

    :param config: Parsed YAML configuration dictionary.
    :param file_path: Path to the YAML file (for error
        messages).
    :raises AgentConfigurationError: If a mandatory key is
        missing.
    """
    for key in ("name", "version", "engine", "system_prompt"):
        if key not in config:
            raise AgentConfigurationError(
                f"Missing mandatory key `{key}` in the "
                f"configuration yaml `{file_path}`"
            )
    engine = config["engine"]
    for key in ("provider", "model"):
        if key not in engine:
            raise AgentConfigurationError(
                f"Missing mandatory key `{key}` in the "
                f"engine configuration yaml "
                f"`{file_path}`"
            )


def load_chatter_yaml_settings(file_path: str) -> dict:
    """Load Chatter agent settings from a YAML file.

    :param file_path: The path to the YAML file. Schema:
        ```yaml
        base_config: <path to a base YAML config>
        name: <name of the agent>
        version: <version of the agent>
        engine:
            provider: <llm provider>
            model: <llm model>
        system_prompt: <system prompt / personality>
        temperature: <optional, default 0.7>
        think: <optional, default false>
        max_history: <optional, default 50>
        prompt_params:
            <key>: <value>
        ```
        The fields `base_config`, `prompt_params`,
        `temperature`, `think`, and `max_history` are
        optional.
    :return: Configuration dictionary for Chatter.
    """
    config = _resolve_inheritance(file_path)
    validate_chatter_yaml_config(config, file_path)

    prompt_params = None
    if raw_params := config.get("prompt_params"):
        prompt_params = {str(k): str(v) for k, v in raw_params.items()}

    result = {
        "name": config["name"],
        "version": config["version"],
        "engine": build_engine(config["engine"]),
        "system_prompt": config["system_prompt"].strip(),
        "prompt_params": prompt_params,
    }

    cast_optional_fields(
        config,
        result,
        {
            "temperature": float,
            "think": bool,
            "max_history": int,
            "max_turns": int,
            "typo_rate": float,
            "conversation_id": str,
        },
    )
    if end_condition_path := config.get("end_condition"):
        result["end_condition"] = import_from_path(end_condition_path)
    if memory_settings := config.get("memory"):
        from agents7.agents.memory import (
            MemoryConfig,
            create_memory,
        )

        memory_config = MemoryConfig(
            backend=memory_settings.get("backend", "json"),
            path=memory_settings.get("path", ""),
        )
        result["memory"] = create_memory(memory_config)

    return result


def _derive_persona_summary(
    persona: Any,
    scene: Any = None,
) -> str:
    """Build a persona (and optional scene) summary for the guard.

    When a `Scene` is provided the summary includes setting,
    situation, and rules so the guard can evaluate scene-specific
    off-topic responses and rule violations.

    :param persona: A `Persona` instance.
    :param scene: An optional `Scene` instance.
    :return: Multi-line summary string.
    """
    parts = [persona.full_name]
    if persona.nationality:
        parts.append(persona.nationality)
    if persona.occupation:
        parts.append(persona.occupation)
    if persona.personality:
        parts.append(", ".join(persona.personality[:3]))
    summary = ", ".join(parts) + "."

    if scene is not None:
        scene_lines: list[str] = []
        if scene.setting:
            scene_lines.append(f"Setting: {scene.setting.strip()}")
        if scene.situation:
            scene_lines.append(f"Situation: {scene.situation.strip()}")
        if scene.rules:
            scene_lines.append("Scene rules:")
            for rule in scene.rules:
                scene_lines.append(f"  - {rule}")
        if scene_lines:
            summary += "\n" + "\n".join(scene_lines)

    return summary


def _load_base_chatter_config(
    file_path: str,
) -> tuple[dict, Any, str]:
    """Parse, validate, and extract base chatter fields.

    Common setup for Chatter-family YAML loaders: resolves
    inheritance, validates mandatory keys, builds the Persona,
    and extracts extra_instructions.

    :param file_path: Path to the YAML config file.
    :return: Tuple of (raw config dict, Persona instance,
        extra_instructions string).
    """
    from agents7.agents.personality import Persona

    raw = _resolve_inheritance(file_path)
    raw.setdefault("system_prompt", "(auto-generated)")
    validate_chatter_yaml_config(raw, file_path)
    persona = Persona.from_dict(
        dict(raw.get("persona", {})),
    )
    extra_instructions = raw.get("extra_instructions", "")
    if extra_instructions:
        extra_instructions = str(extra_instructions).strip()
    return raw, persona, extra_instructions


def _build_guard(
    raw: dict,
    persona: Any,
    scene: Any = None,
) -> Any:
    """Build a ContextGuard from the raw YAML config.

    Shared between GuardedChatter and GuardedSceneChatter
    loaders.

    :param raw: Fully resolved YAML config dict.
    :param persona: The agent's Persona instance.
    :param scene: Optional Scene instance (GuardedSceneChatter
        only). When provided, scene context is included in
        the guard's persona summary.
    :return: A configured ContextGuard instance.
    """
    from agents7.agents.context_guard import (
        ContextGuard,
        _PLACEHOLDER,
    )

    guard_raw = raw.get("guard", {})
    guard_engine = build_engine(
        guard_raw.get("engine", raw["engine"]),
    )
    persona_summary = guard_raw.get(
        "persona_summary",
        _derive_persona_summary(persona, scene),
    )
    guard_kwargs: dict[str, Any] = {
        "name": f"{raw['name']}-guard",
        "version": raw["version"],
        "engine": guard_engine,
        "system_prompt": _PLACEHOLDER,
        "persona_summary": persona_summary,
    }
    if "think" in guard_raw:
        guard_kwargs["think"] = bool(guard_raw["think"])
    return ContextGuard(**guard_kwargs)


def load_guarded_yaml_settings(file_path: str) -> Any:
    """Load a `GuardedChatter` from a YAML file.

    Consolidates all YAML parsing, validation, guard
    construction, and optional-field casting in one place.

    :param file_path: Path to the YAML configuration file.
    :return: A `GuardedChatter` instance.
    """
    from agents7.agents.guarded import GuardedChatter
    from agents7.agents.memory import (
        MemoryConfig,
        create_memory,
    )

    raw, persona, extra_instructions = _load_base_chatter_config(
        file_path,
    )
    guard = _build_guard(raw, persona)

    config: dict[str, Any] = {
        "name": raw["name"],
        "version": raw["version"],
        "engine": build_engine(raw["engine"]),
        "system_prompt": "",
        "persona": persona,
        "extra_instructions": extra_instructions,
        "guard": guard,
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
            "max_guard_retries": int,
            "guard_context_turns": int,
        },
    )
    if memory_settings := raw.get("memory"):
        memory_config = MemoryConfig(
            backend=memory_settings.get("backend", "json"),
            path=memory_settings.get("path", ""),
        )
        config["memory"] = create_memory(memory_config)

    return GuardedChatter(**config)


def load_guarded_scene_yaml_settings(file_path: str) -> Any:
    """Load a `GuardedSceneChatter` from a YAML file.

    Mirrors `load_guarded_yaml_settings` but also reads the
    ``scene`` section and returns a `GuardedSceneChatter`.

    :param file_path: Path to the YAML configuration file.
    :return: A `GuardedSceneChatter` instance.
    """
    from agents7.agents.guarded_scene import GuardedSceneChatter
    from agents7.agents.memory import (
        MemoryConfig,
        create_memory,
    )
    from agents7.agents.personality import Scene

    raw, persona, extra_instructions = _load_base_chatter_config(
        file_path,
    )
    scene = Scene.from_dict(dict(raw.get("scene", {})))
    guard = _build_guard(raw, persona, scene=scene)

    config: dict[str, Any] = {
        "name": raw["name"],
        "version": raw["version"],
        "engine": build_engine(raw["engine"]),
        "system_prompt": "",
        "persona": persona,
        "scene": scene,
        "extra_instructions": extra_instructions,
        "guard": guard,
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
            "max_guard_retries": int,
            "guard_context_turns": int,
        },
    )
    if memory_settings := raw.get("memory"):
        memory_config = MemoryConfig(
            backend=memory_settings.get("backend", "json"),
            path=memory_settings.get("path", ""),
        )
        config["memory"] = create_memory(memory_config)

    return GuardedSceneChatter(**config)
