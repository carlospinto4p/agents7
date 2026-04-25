"""YAML configuration loading for agents."""

import os
from typing import Any

from cpinto.files import load_yaml

from agents7.agents.utils import (
    get_examples_str,
    import_from_path,
)
from agents7.exceptions import AgentConfigurationError
from agents7.llm import LLMEngine
from agents7.llm.cache import CacheConfig
from agents7.llm.retry import RetryConfig


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


def _resolve_inheritance(
    file_path: str,
    _seen: set[str] | None = None,
) -> dict:
    """Load a YAML config, resolving ``base_config`` chain.

    :param file_path: Path to the YAML file.
    :param _seen: Set of already-visited paths (for
        circular-reference detection).
    :return: Fully merged configuration dictionary.
    :raises AgentConfigurationError: On circular references.
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

    config = load_yaml(file_path)
    base_path = config.pop("base_config", None)
    if base_path is None:
        return config

    # Resolve relative paths from the current file's directory
    if not os.path.isabs(base_path):
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
        extra_kwargs["base_url"] = engine_config["base_url"]
    if engine_config.get("api_key"):
        extra_kwargs["api_key"] = engine_config["api_key"]

    return LLMEngine(
        engine_config["provider"],
        engine_config["model"],
        retry=retry_config,
        cache=cache_config,
        fallback_model=engine_config.get("fallback_model"),
        **extra_kwargs,
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


def _derive_persona_summary(persona: Any) -> str:
    """Build a one-line persona summary for the guard.

    :param persona: A `Persona` instance.
    :return: Comma-separated summary string.
    """
    parts = [persona.full_name]
    if persona.nationality:
        parts.append(persona.nationality)
    if persona.occupation:
        parts.append(persona.occupation)
    if persona.personality:
        parts.append(", ".join(persona.personality[:3]))
    return ", ".join(parts)


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


def _build_guard(raw: dict, persona: Any) -> Any:
    """Build a ContextGuard from the raw YAML config.

    Shared between GuardedChatter and GuardedSceneChatter
    loaders.

    :param raw: Fully resolved YAML config dict.
    :param persona: The agent's Persona instance.
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
        _derive_persona_summary(persona),
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
    guard = _build_guard(raw, persona)

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
