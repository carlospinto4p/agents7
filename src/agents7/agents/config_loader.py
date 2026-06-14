"""YAML configuration loading for agents."""

import os
from typing import Any

from agents7.agents.utils import (
    get_examples_str,
    import_from_path,
)
from agents7.exceptions import AgentConfigurationError
from agents7.llm import LLMEngine
from agents7.llm.cache import CacheConfig
from agents7.llm.retry import RetryConfig


def _resolve_env_var(value: str) -> str:
    """Resolve a value that may be an environment variable reference.

    If *value* is of the form ``${VAR_NAME}``, the environment
    variable ``VAR_NAME`` is returned. Otherwise *value* is
    returned unchanged.

    Use this format in YAML configs to avoid storing API keys
    in plain text::

        engine:
          api_key: ${MY_API_KEY}

    :param value: Raw string value from YAML config.
    :return: Resolved string value.
    :raises AgentConfigurationError: If the referenced
        environment variable is not set.
    """
    if value.startswith("${") and value.endswith("}"):
        var_name = value[2:-1]
        env_value = os.environ.get(var_name)
        if env_value is None:
            raise AgentConfigurationError(
                f"Environment variable `{var_name}` referenced "
                f"in config is not set."
            )
        return env_value
    return value


def cast_optional_fields(
    source: dict,
    target: dict,
    fields: dict[str, type],
) -> None:
    """Copy fields from *source* to *target*, casting each value.

    Only fields present in *source* are copied. This replaces
    repetitive ``if "field" in config`` blocks in YAML loaders.

    :param source: Raw config dictionary (e.g. parsed YAML).
    :param target: Target dictionary to populate.
    :param fields: Mapping of field name → type callable
        (e.g. ``{"temperature": float, "think": bool}``).
    """
    for name, caster in fields.items():
        if name in source:
            target[name] = caster(source[name])


def _maybe_add_memory(source: dict, target: dict) -> None:
    """Attach a memory backend to *target* if configured.

    Reads the optional ``memory`` section from *source* and,
    when present, builds the backend and stores it under
    ``target["memory"]``. Shared by every Chatter-family
    loader.

    :param source: Fully resolved YAML config dict.
    :param target: Result/config dict to populate in place.
    """
    memory_settings = source.get("memory")
    if not memory_settings:
        return
    from agents7.agents.memory import MemoryConfig, create_memory

    target["memory"] = create_memory(
        MemoryConfig(
            backend=memory_settings.get("backend", "json"),
            path=memory_settings.get("path", ""),
        )
    )


def _build_persona(data: Any, file_path: str) -> Any:
    """Build a `Persona` from raw YAML data with a clear error.

    Wraps `Persona.from_dict` so a malformed ``persona`` section
    (e.g. missing the required ``full_name``, an unknown field, or
    a non-mapping value) surfaces as an `AgentConfigurationError`
    naming the file rather than a raw `TypeError`.

    :param data: The raw ``persona`` value from the config.
    :param file_path: Path to the YAML file (for error messages).
    :return: A `Persona` instance.
    :raises AgentConfigurationError: If the persona fields are
        invalid.
    """
    from agents7.agents.persona import Persona

    try:
        return Persona.from_dict(dict(data))
    except (TypeError, ValueError) as e:
        raise AgentConfigurationError(
            f"Invalid `persona` section in `{file_path}`: {e}"
        ) from e


def _build_scene(data: Any, file_path: str) -> Any:
    """Build a `Scene` from raw YAML data with a clear error.

    Wraps `Scene.from_dict` so a malformed ``scene`` section
    (e.g. an unknown field or a non-mapping value) surfaces as an
    `AgentConfigurationError` naming the file rather than a raw
    `TypeError`.

    :param data: The raw ``scene`` value from the config.
    :param file_path: Path to the YAML file (for error messages).
    :return: A `Scene` instance.
    :raises AgentConfigurationError: If the scene fields are
        invalid.
    """
    from agents7.agents.persona import Scene

    try:
        return Scene.from_dict(dict(data))
    except (TypeError, ValueError) as e:
        raise AgentConfigurationError(
            f"Invalid `scene` section in `{file_path}`: {e}"
        ) from e


# Optional fields shared by every Chatter-based loader, keyed by
# the caster applied to the raw YAML value.
_CHATTER_OPTIONAL_FIELDS: dict[str, type] = {
    "temperature": float,
    "think": bool,
    "max_history": int,
    "max_turns": int,
    "typo_rate": float,
    "conversation_id": str,
}

# Guarded loaders accept the Chatter base fields plus the guard
# tuning fields.
_GUARDED_OPTIONAL_FIELDS: dict[str, type] = {
    **_CHATTER_OPTIONAL_FIELDS,
    "max_guard_retries": int,
    "guard_context_turns": int,
    "guard_buffer_stream": bool,
    "guard_fail_open": bool,
    "guard_fallback_response": str,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*.

    For nested dicts, values are merged recursively.
    For all other types, *override* wins.

    :param base: Base configuration dictionary.
    :param override: Override dictionary (takes priority).
    :return: Merged dictionary (new dict, inputs unchanged).
    """
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(file_path: str) -> dict:
    import yaml

    with open(file_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_inheritance(
    file_path: str,
    _seen: set[str] | None = None,
) -> dict:
    """Load a YAML config, resolving ``base_config`` chain.

    ``base_config`` values must be relative paths. Absolute paths
    are rejected to prevent injection of arbitrary system files
    (e.g. ``base_config: /etc/passwd``).

    :param file_path: Path to the YAML file.
    :param _seen: Set of already-visited paths (for
        circular-reference detection).
    :return: Fully merged configuration dictionary.
    :raises AgentConfigurationError: On circular references or
        if ``base_config`` is an absolute path.
    """
    resolved = os.path.abspath(file_path)
    if _seen is None:
        _seen = set()
    if resolved in _seen:
        raise AgentConfigurationError(
            f"Circular base_config reference detected: "
            f"`{file_path}` was already visited."
        )
    _seen.add(resolved)

    config = _load_yaml(file_path)
    base_path = config.pop("base_config", None)
    if base_path is None:
        return config

    if os.path.isabs(base_path):
        raise AgentConfigurationError(
            f"base_config path `{base_path}` must be relative, "
            f"not an absolute path."
        )

    # Resolve relative path from the current file's directory
    base_path = os.path.join(os.path.dirname(resolved), base_path)

    base_config = _resolve_inheritance(base_path, _seen)
    return _deep_merge(base_config, config)


def load_yaml_settings(file_path: str) -> dict:
    """Load agent settings from a YAML file.

    :param file_path: The path to the YAML file. Schema:
        ```yaml
        base_config: <path to a base YAML config>
        name: <name of the agent>
        version: <version of the agent>
        engine:
            provider: <llm provider>
            model: <llm model>
            fallback_model: <optional fallback model name>
        system_prompt: <system prompt for the agent>
        think: <optional, default true>
        prompt_params:
            <key>: <value to substitute $key in system_prompt>
            ...
        system_examples:
            - input:
                <example 1 input>
              output:
                <example 1 output>
            ...
        output_model: <path to the pydantic model class, e.g.
            "agents7.models.MyOutputModel">
        ```
        The fields `base_config`, `prompt_params`,
        `system_examples`, `think`, and `fallback_model`
        are optional.

        When ``base_config`` is specified, the base YAML is
        loaded first and the current file's keys override
        it. Nested dicts (e.g. ``engine``) are merged
        recursively. Inheritance chains are supported
        (A → B → C) but circular references raise an error.
    :return: Loaded configuration, as a dictionary. Keys:
        - name (str): The name of the agent.
        - version (int): The version of the agent.
        - engine (str): The LLM engine used by the agent.
        - system_prompt (str): The system prompt.
        - prompt_params (dict, optional): Dictionary of
            parameters to substitute into system_prompt.
        - output_model (BaseModel): The model used to
            validate and structure the output.
        - examples (list[dict], optional): List of examples.
    """
    config = _resolve_inheritance(file_path)
    validate_yaml_config(config, file_path)

    examples = parse_yaml_examples(config, file_path)

    prompt_params = None
    if raw_params := config.get("prompt_params"):
        prompt_params = {str(k): str(v) for k, v in raw_params.items()}

    validators = None
    if validator_paths := config.get("validators"):
        validators = [import_from_path(path) for path in validator_paths]

    result = {
        "name": config["name"],
        "version": config["version"],
        "engine": build_engine(config["engine"]),
        "system_prompt": config["system_prompt"].strip(),
        "prompt_params": prompt_params,
        "output_model": import_from_path(config["output_model"]),
        "examples": examples,
        "validators": validators,
    }

    cast_optional_fields(
        config,
        result,
        {
            "think": bool,
            "max_parse_retries": int,
        },
    )

    if call_logger_path := config.get("call_logger"):
        result["call_logger"] = import_from_path(call_logger_path)

    if preprocessor_path := config.get("preprocessor"):
        result["preprocessor"] = import_from_path(preprocessor_path)

    return result


def validate_yaml_config(config: dict, file_path: str) -> None:
    """Validate mandatory keys in a YAML configuration.

    :param config: Parsed YAML configuration dictionary.
    :param file_path: Path to the YAML file (for error
        messages).
    :raises AgentConfigurationError: If a mandatory key is
        missing at the top level or inside the engine section.
    """
    for key in (
        "name",
        "version",
        "engine",
        "system_prompt",
        "output_model",
    ):
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


def build_engine(engine_config: dict) -> LLMEngine:
    """Build an `LLMEngine` from the engine section of YAML.

    :param engine_config: Dictionary with keys `provider`,
        `model`, and optional `retry`, `cache`, and
        `fallback_model` fields.
    :return: Configured `LLMEngine` instance.
    """
    retry_config = None
    if retry_settings := engine_config.get("retry"):
        retry_config = RetryConfig(**retry_settings)

    cache_config = None
    cache_settings = engine_config.get("cache")
    if cache_settings is not None:
        cache_config = CacheConfig(**cache_settings)

    extra_kwargs = {}
    if engine_config.get("base_url"):
        extra_kwargs["base_url"] = _resolve_env_var(engine_config["base_url"])
    if engine_config.get("api_key"):
        extra_kwargs["api_key"] = _resolve_env_var(engine_config["api_key"])

    return LLMEngine(
        engine_config["provider"],
        engine_config["model"],
        retry=retry_config,
        cache=cache_config,
        fallback_model=engine_config.get("fallback_model"),
        **extra_kwargs,
    )


def parse_yaml_examples(config: dict, file_path: str) -> list[dict] | None:
    """Extract and validate examples from YAML config.

    :param config: Parsed YAML configuration dictionary.
    :param file_path: Path to the YAML file (for error
        messages).
    :return: List of example dicts, or `None` if absent.
    :raises AgentConfigurationError: If examples are
        malformed.
    """
    examples = config.get("system_examples")
    if examples is None:
        return None
    try:
        get_examples_str(examples)
    except (ValueError, AgentConfigurationError) as e:
        raise AgentConfigurationError(
            f"{e} in the configuration file `{file_path}`"
        ) from e
    return examples


# ── Chatter-family loaders ─────────────────────────────────────────


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

    # Persona/scene-driven chatters auto-generate the system
    # prompt, so a manual `system_prompt` is not required.
    has_persona = "persona" in config
    has_scene = "scene" in config
    if has_persona or has_scene:
        config.setdefault("system_prompt", "(auto-generated)")

    validate_chatter_yaml_config(config, file_path)

    prompt_params = None
    if raw_params := config.get("prompt_params"):
        prompt_params = {str(k): str(v) for k, v in raw_params.items()}

    result = {
        "name": config["name"],
        "version": config["version"],
        "engine": build_engine(config["engine"]),
        "system_prompt": (
            ""
            if (has_persona or has_scene)
            else config["system_prompt"].strip()
        ),
        "prompt_params": prompt_params,
    }

    if has_persona or has_scene:
        if has_persona:
            result["persona"] = _build_persona(config["persona"], file_path)
        if has_scene:
            result["scene"] = _build_scene(config["scene"], file_path)
        if extra := config.get("extra_instructions"):
            result["extra_instructions"] = str(extra).strip()

    cast_optional_fields(config, result, _CHATTER_OPTIONAL_FIELDS)
    if end_condition_path := config.get("end_condition"):
        result["end_condition"] = import_from_path(end_condition_path)
    _maybe_add_memory(config, result)

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
    lines: list[str] = [", ".join(parts) + "."]

    if scene is not None:
        scene_summary = scene.to_summary()
        if scene_summary:
            lines.append(scene_summary)

    return "\n".join(lines)


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
    raw = _resolve_inheritance(file_path)
    raw.setdefault("system_prompt", "(auto-generated)")
    validate_chatter_yaml_config(raw, file_path)
    persona = _build_persona(raw.get("persona", {}), file_path)
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
        forbidden_terms_for_language,
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
    # Explicit forbidden_terms win; otherwise derive from the
    # persona's language (defaulting to the bilingual list).
    if "forbidden_terms" in guard_raw:
        guard_kwargs["forbidden_terms"] = list(guard_raw["forbidden_terms"])
    elif getattr(persona, "language", ""):
        guard_kwargs["forbidden_terms"] = forbidden_terms_for_language(
            persona.language
        )
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
    cast_optional_fields(raw, config, _GUARDED_OPTIONAL_FIELDS)
    _maybe_add_memory(raw, config)

    return GuardedChatter(**config)


def load_guarded_scene_yaml_settings(file_path: str) -> Any:
    """Load a `GuardedSceneChatter` from a YAML file.

    Mirrors `load_guarded_yaml_settings` but also reads the
    ``scene`` section and returns a `GuardedSceneChatter`.

    :param file_path: Path to the YAML configuration file.
    :return: A `GuardedSceneChatter` instance.
    """
    from agents7.agents.guarded_scene import GuardedSceneChatter

    raw, persona, extra_instructions = _load_base_chatter_config(
        file_path,
    )
    scene = _build_scene(raw.get("scene", {}), file_path)
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
    cast_optional_fields(raw, config, _GUARDED_OPTIONAL_FIELDS)
    _maybe_add_memory(raw, config)

    return GuardedSceneChatter(**config)
