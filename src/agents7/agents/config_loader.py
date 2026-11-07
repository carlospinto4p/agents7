"""YAML configuration loading for agents."""

import os
from typing import Any

from agents7.agents.context import ContextParam
from agents7.agents.utils import (
    get_examples_str,
    import_from_path,
)
from agents7.exceptions import AgentConfigurationError
from agents7.llm import LLMEngine
from agents7.llm.cache import CacheConfig
from agents7.llm.pricing import Pricing
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


def _resolve_store_path(path: str, file_path: str) -> str:
    """Anchor a store path to the config file that declares it.

    A ``memory`` or ``character_facts`` path names a folder that
    belongs to the *agent*, not to whoever runs it. Left relative
    it would follow the process's working directory instead, so
    the same config would build one store when loaded by a web
    backend and another when loaded by a script, and neither
    would see what the other recorded. Resolving it against the
    directory holding the config keeps one store per agent,
    wherever it is loaded from.

    An absolute path is returned unchanged, as is an empty one —
    that means "the backend's own default", not a location.

    :param path: The path as written in the config.
    :param file_path: Path to the YAML file declaring it.
    :return: An absolute path, or *path* unchanged when it is
        empty or already absolute.
    """
    if not path or os.path.isabs(path):
        return path
    config_dir = os.path.dirname(os.path.abspath(file_path))
    # Normalised, so a path that climbs out of the config's own
    # folder — `../../data/calls.db` from a config living under
    # `src/<pkg>/` — reads as where it lands rather than as the
    # route it took. The joined form works either way; it is
    # error messages and backup listings that are unreadable.
    return os.path.normpath(os.path.join(config_dir, path))


def _maybe_add_memory(source: dict, target: dict, file_path: str) -> None:
    """Attach a memory backend to *target* if configured.

    Reads the optional ``memory`` section from *source* and,
    when present, builds the backend and stores it under
    ``target["memory"]``. Shared by every Chatter-family
    loader.

    :param source: Fully resolved YAML config dict.
    :param target: Result/config dict to populate in place.
    :param file_path: Path to the YAML file, which a relative
        ``path`` is resolved against.
    """
    memory_settings = source.get("memory")
    if not memory_settings:
        return
    from agents7.agents.memory import MemoryConfig, create_memory

    target["memory"] = create_memory(
        MemoryConfig(
            backend=memory_settings.get("backend", "json"),
            path=_resolve_store_path(
                memory_settings.get("path", ""), file_path
            ),
        )
    )


def _maybe_add_facts(source: dict, target: dict, file_path: str) -> None:
    """Attach a character fact store to *target* if configured.

    Reads the optional ``character_facts`` section from
    *source*::

        character_facts:
            path: facts/          # optional, default "facts/"
            character_id: david   # optional, default the name
            banned_phrases:       # optional, never stored or
                - sin remedio      # put in the prompt
            extractor:            # optional
                engine:           # optional, default the agent's
                    provider: ollama
                    model: gemma4:12b
                think: false

    A relative ``path`` is resolved against the directory
    holding the config, so the folder sits beside the agent it
    belongs to rather than wherever the process happens to run
    — see `_resolve_store_path`.

    An extractor is always built, so `Chatter.remember` works
    out of the box; it reuses the agent's own engine unless
    the section names another one. Shared by every
    Chatter-family loader.

    :param source: Fully resolved YAML config dict.
    :param target: Result/config dict to populate in place.
    :param file_path: Path to the YAML file, which a relative
        ``path`` is resolved against.
    """
    settings = source.get("character_facts")
    if not settings:
        return
    from agents7.agents.facts import FactExtractor, JsonFileFactStore

    target["facts"] = JsonFileFactStore(
        _resolve_store_path(settings.get("path", "facts/"), file_path)
    )
    if character_id := settings.get("character_id"):
        target["character_id"] = str(character_id)
    if banned := settings.get("banned_phrases"):
        target["banned_fact_phrases"] = tuple(str(p) for p in banned)
    extractor_raw = settings.get("extractor") or {}
    extractor_kwargs: dict[str, Any] = {
        "name": f"{source['name']}-facts",
        "version": source["version"],
        "engine": build_engine(
            extractor_raw.get("engine", source["engine"]),
        ),
    }
    if "think" in extractor_raw:
        extractor_kwargs["think"] = bool(extractor_raw["think"])
    target["fact_extractor"] = FactExtractor(**extractor_kwargs)


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
# `call_logger` is absent from these maps on purpose: it is a
# dotted path to import, not a value to cast. See
# `_maybe_add_call_logger`.
_CHATTER_OPTIONAL_FIELDS: dict[str, type] = {
    "temperature": float,
    "think": bool,
    "max_history": int,
    "max_turns": int,
    "typo_rate": float,
    "conversation_id": str,
    "can_open": bool,
    "opening_prompt": str,
    "opening_instructions": str,
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


# The two composition keys. A `character` file says who the agent
# is; a `chat_goal` file says what it is doing there and why.
# Neither may hold the other's section, so a config that has been
# split cannot quietly go back to mixing the two — the check is in
# `_load_fragment`, and it is the point of the split rather than a
# nicety, because the entanglement is what made a character
# unusable in a second scenario.
_CHARACTER_KEY = "character"
_CHAT_GOAL_KEY = "chat_goal"
_FRAGMENT_KEYS = (_CHARACTER_KEY, _CHAT_GOAL_KEY)

# What each fragment is forbidden to declare, and why.
_FRAGMENT_FORBIDDEN: dict[str, dict[str, str]] = {
    _CHARACTER_KEY: {
        "scene": (
            "the scene is what the character is doing, which "
            "belongs to the `chat_goal` file"
        ),
    },
    _CHAT_GOAL_KEY: {
        "persona": (
            "the persona is who the character is, which belongs "
            "to the `character` file"
        ),
    },
}

# Free-text blocks that accumulate across a composition instead of
# being overwritten by it. Both axes have something to say about
# how the agent writes — the character its voice, the goal its
# moves — so the two are joined rather than one silently replacing
# the other. Order is fixed: character, then chat_goal, then
# whatever the agent config says itself.
_CONCATENATED_KEYS = ("extra_instructions", "opening_instructions")


def _compose(base: dict, override: dict) -> dict:
    """Merge *override* onto *base*, accumulating the text blocks.

    Identical to `_deep_merge` except for the keys in
    `_CONCATENATED_KEYS`, which are joined with a blank line in
    base-then-override order.

    :param base: The lower-precedence mapping.
    :param override: The higher-precedence mapping.
    :return: Merged mapping (new dict, inputs unchanged).
    """
    merged = _deep_merge(base, override)
    for key in _CONCATENATED_KEYS:
        texts = [
            str(source[key]).strip()
            for source in (base, override)
            if str(source.get(key) or "").strip()
        ]
        if texts:
            merged[key] = "\n\n".join(texts)
    return merged


def _load_fragment(key: str, path: str, config_path: str) -> dict:
    """Load and check one composition fragment.

    :param key: Either ``"character"`` or ``"chat_goal"``.
    :param path: The relative path the config declared.
    :param config_path: The config file that declared it, so the
        path resolves beside it rather than against the cwd.
    :return: The fragment as a plain mapping.
    :raises AgentConfigurationError: If the path is absolute or
        missing, the fragment nests another composition or
        inheritance key, or it declares a section belonging to
        the other axis.
    """
    if os.path.isabs(path):
        raise AgentConfigurationError(
            f"`{key}` path `{path}` must be relative, not an absolute path."
        )
    resolved = os.path.join(os.path.dirname(config_path), path)
    if not os.path.isfile(resolved):
        raise AgentConfigurationError(
            f"`{key}` file `{path}`, declared in "
            f"`{config_path}`, does not exist."
        )
    fragment = _load_yaml(resolved)

    nested = [k for k in ("base_config", *_FRAGMENT_KEYS) if k in fragment]
    if nested:
        raise AgentConfigurationError(
            f"`{key}` file `{path}` declares "
            f"`{', '.join(nested)}`. A fragment is one section of "
            f"a config, not a config of its own — compose it from "
            f"the agent config instead."
        )

    for forbidden, reason in _FRAGMENT_FORBIDDEN[key].items():
        if forbidden in fragment:
            raise AgentConfigurationError(
                f"`{key}` file `{path}` declares `{forbidden}`: {reason}."
            )
    return fragment


def _compose_fragments(config: dict, file_path: str) -> dict:
    """Fold any ``character`` / ``chat_goal`` files into *config*.

    The fragments are the base and the agent config is the
    override, so a config can pin a value the character or the
    goal left open. Both composition keys are removed from the
    result.

    :param config: The config as resolved so far.
    :param file_path: Path to the config file, used for the
        relative fragment paths and for error messages.
    :return: The composed configuration mapping.
    """
    layers: list[dict] = []
    for key in _FRAGMENT_KEYS:
        path = config.pop(key, None)
        if path is None:
            continue
        if not isinstance(path, str):
            raise AgentConfigurationError(
                f"`{key}` in `{file_path}` must be a path to a "
                f"YAML file, not {type(path).__name__}."
            )
        layers.append(_load_fragment(key, path, file_path))
    if not layers:
        return config

    composed = layers[0]
    for layer in (*layers[1:], config):
        composed = _compose(composed, layer)
    return composed


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
        return _compose_fragments(config, file_path)

    if os.path.isabs(base_path):
        raise AgentConfigurationError(
            f"base_config path `{base_path}` must be relative, "
            f"not an absolute path."
        )

    # Resolve relative path from the current file's directory
    base_path = os.path.join(os.path.dirname(resolved), base_path)

    base_config = _resolve_inheritance(base_path, _seen)
    return _compose_fragments(
        _deep_merge(base_config, config),
        file_path,
    )


def resolve_config(file_path: str) -> dict:
    """Return an agent config with every file reference resolved.

    Applies the ``base_config`` inheritance chain and then folds
    in any ``character`` / ``chat_goal`` fragments, so the result
    is what the agent is actually built from. Nothing is imported
    and no engine is constructed — this is the configuration as
    text, for anything that needs to read a config rather than
    build the agent (a config-contract test, a catalog listing).

    :param file_path: Path to the agent's YAML config.
    :return: The fully resolved configuration mapping.
    :raises AgentConfigurationError: On a circular, absolute, or
        missing reference, or a fragment holding the other axis's
        section.
    """
    return _resolve_inheritance(file_path)


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
            fallback_engine:
                <an engine section of its own, any provider>
            timeout: <optional seconds per request>
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
        `system_examples`, `think`, `fallback_model`,
        `fallback_engine` and `timeout` are optional.

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

    prompt_params = _parse_prompt_params(config)

    validators = None
    if validator_paths := config.get("validators"):
        validators = [import_from_path(path) for path in validator_paths]

    result = {
        "name": config["name"],
        "version": config["version"],
        "engine": build_engine(config["engine"]),
        "system_prompt": config["system_prompt"].strip(),
        "prompt_params": prompt_params,
        "context_params": parse_context_params(config, file_path),
        "prompt_variants": parse_prompt_variants(config, file_path),
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

    _maybe_add_call_logger(config, result, file_path)

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
        `model`, and optional `retry`, `cache`, `pricing`,
        `timeout`, `fallback_model` and `fallback_engine`
        fields. A config that declares no `retry` gets
        `DEFAULT_RETRY`, not no retry — see `LLMEngine`. To
        disable retrying, declare `retry: {max_retries: 0}`; to
        retry a failure to reach a paid provider at all, declare
        `retry: {retry_connection_errors: true}`.

        `fallback_engine` is an engine section of its own, built
        by this same function, so it may name any provider and
        carries its own `base_url`, `api_key`, `pricing`,
        `timeout` and `retry`. That is what lets a paid primary
        fall back to a local model and still report what the
        run actually cost.
    :return: Configured `LLMEngine` instance.
    :raises AgentConfigurationError: If `pricing` is present but
        is not a mapping of the expected rates, if `timeout` is
        not a positive number, or if `fallback_engine` is not a
        mapping.
    """
    retry_config = None
    if retry_settings := engine_config.get("retry"):
        retry_config = RetryConfig(**retry_settings)

    pricing = _build_pricing(engine_config.get("pricing"))

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
        fallback_engine=_build_fallback_engine(
            engine_config.get("fallback_engine")
        ),
        pricing=pricing,
        timeout=_build_timeout(engine_config.get("timeout")),
        **extra_kwargs,
    )


def _build_fallback_engine(section: object) -> LLMEngine | None:
    """Build the engine a `fallback_engine` section declares.

    Recurses into `build_engine`, so the fallback gets the same
    treatment as any other engine — its own provider,
    credentials, pricing, timeout and retry. Nothing is
    inherited from the primary: a fallback that quietly borrowed
    the primary's rates is the thing this section exists to
    stop.

    :param section: The `fallback_engine` mapping, or ``None``.
    :return: The built `LLMEngine`, or ``None`` when absent.
    :raises AgentConfigurationError: If *section* is present but
        is not a mapping, or declares a `cache`.
    """
    if section is None:
        return None
    if not isinstance(section, dict):
        raise AgentConfigurationError(
            "`engine.fallback_engine` must be an engine section "
            f"(a mapping with `provider` and `model`), got "
            f"{type(section).__name__}."
        )
    if "cache" in section:
        raise AgentConfigurationError(
            "`engine.fallback_engine` must not declare a "
            "`cache`. One cache covers the call whichever model "
            "answers it, and it belongs to the primary engine — "
            "declare it there."
        )
    return build_engine(section)


def _build_timeout(timeout_config: object) -> float | None:
    """Validate the engine's `timeout` section.

    The section is optional; without it the provider's own SDK
    default stands, which is how every config behaved before
    timeouts existed.

    :param timeout_config: The `timeout` value, or ``None``.
    :return: The timeout in seconds, or ``None`` when absent.
    :raises AgentConfigurationError: If the value is not a
        positive number.
    """
    if timeout_config is None:
        return None
    if isinstance(timeout_config, bool) or not isinstance(
        timeout_config, (int, float)
    ):
        raise AgentConfigurationError(
            "`engine.timeout` must be a number of seconds, got "
            f"{timeout_config!r}."
        )
    if timeout_config <= 0:
        raise AgentConfigurationError(
            "`engine.timeout` must be greater than 0 seconds, got "
            f"{timeout_config!r}."
        )
    return float(timeout_config)


def _build_pricing(pricing_config: dict | None) -> Pricing | None:
    """Build a `Pricing` from the engine's `pricing` section.

    The section is optional; without it `cost_cents` stays
    ``None`` on every response, which is how every config
    behaved before pricing existed.

    Expected shape, in USD per million tokens::

        pricing:
            input: 0.44
            output: 1.32
            cached_input: 0.014   # optional, defaults to input

    :param pricing_config: The `pricing` mapping, or ``None``.
    :return: A `Pricing`, or ``None`` when absent.
    :raises AgentConfigurationError: If the section is not a
        mapping, omits a required rate, carries an unknown key,
        or holds a value that is not a non-negative number.
    """
    if pricing_config is None:
        return None
    if not isinstance(pricing_config, dict):
        raise AgentConfigurationError(
            "`engine.pricing` must be a mapping with `input` and "
            "`output` rates in USD per million tokens, got "
            f"{type(pricing_config).__name__}."
        )
    allowed = {"input", "output", "cached_input"}
    if unknown := set(pricing_config) - allowed:
        raise AgentConfigurationError(
            f"`engine.pricing` has unknown key(s): "
            f"{', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(sorted(allowed))}."
        )
    for key in ("input", "output"):
        if key not in pricing_config:
            raise AgentConfigurationError(
                f"`engine.pricing` is missing `{key}`. Both "
                f"`input` and `output` are required, in USD per "
                f"million tokens."
            )
    values = {}
    for key, value in pricing_config.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AgentConfigurationError(
                f"`engine.pricing.{key}` must be a number in USD "
                f"per million tokens, got {value!r}."
            )
        values[key] = float(value)
    try:
        return Pricing(**values)
    except ValueError as exc:
        raise AgentConfigurationError(
            f"`engine.pricing` is invalid: {exc}"
        ) from exc


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


def _parse_prompt_params(config: dict) -> dict[str, str] | None:
    """Read the `prompt_params` section as fixed strings.

    :param config: Parsed YAML configuration dictionary.
    :return: Mapping of placeholder name to value, or None when
        the section is absent or empty.
    """
    if raw_params := config.get("prompt_params"):
        return {str(k): str(v) for k, v in raw_params.items()}
    return None


def parse_prompt_variants(
    config: dict,
    file_path: str,
) -> dict[str, tuple[str, ...]] | None:
    """Read the `prompt_variants` section as lists of texts.

    Each entry names a placeholder and lists the texts one of
    which is drawn per conversation. Whether there are enough of
    them, and whether they repeat, is checked by the agent.

    :param config: Parsed YAML configuration dictionary.
    :param file_path: Path to the YAML file (for error messages).
    :return: Mapping of placeholder name to texts, or None when
        the section is absent or empty.
    :raises AgentConfigurationError: If the section is not a
        mapping or an entry is not a list.
    """
    raw = config.get("prompt_variants")
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise AgentConfigurationError(
            f"`prompt_variants` must be a mapping of name to a list "
            f"of texts in the configuration yaml `{file_path}`"
        )
    variants: dict[str, tuple[str, ...]] = {}
    for name, texts in raw.items():
        if not isinstance(texts, list):
            raise AgentConfigurationError(
                f"prompt_variants['{name}'] must be a list of texts "
                f"in the configuration yaml `{file_path}`"
            )
        variants[str(name)] = tuple(str(t).strip() for t in texts)
    return variants


def parse_context_params(
    config: dict,
    file_path: str,
) -> tuple[ContextParam, ...] | None:
    """Read the `context_params` section into declarations.

    Each entry is either a scalar (taken as the default) or a
    mapping with optional ``default``, ``description``,
    ``callable`` (a dotted path to a zero-argument function
    called before every message) and ``from`` (the name of
    another declared param, whose value the ``callable`` then
    takes as its one argument).

    :param config: Parsed YAML configuration dictionary.
    :param file_path: Path to the YAML file (for error
        messages).
    :return: Tuple of `ContextParam`, or None when the section
        is absent or empty.
    :raises AgentConfigurationError: If an entry is malformed or
        its ``callable`` cannot be imported.
    """
    raw = config.get("context_params")
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise AgentConfigurationError(
            f"`context_params` must be a mapping of name to "
            f"declaration in the configuration yaml `{file_path}`"
        )
    params: list[ContextParam] = []
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            params.append(
                ContextParam(
                    name=str(name),
                    default="" if spec is None else str(spec),
                )
            )
            continue
        resolver = None
        if path := spec.get("callable"):
            try:
                resolver = import_from_path(str(path))
            except (ImportError, AttributeError, ValueError) as e:
                raise AgentConfigurationError(
                    f"context_params['{name}'].callable `{path}` "
                    f"could not be imported, in the configuration "
                    f"yaml `{file_path}`: {e}"
                ) from e
        params.append(
            ContextParam(
                name=str(name),
                default=str(spec.get("default", "")),
                description=str(spec.get("description", "")),
                resolver=resolver,
                derived_from=str(spec.get("from") or ""),
            )
        )
    return tuple(params)


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


def _maybe_add_call_logger(
    config: dict, result: dict, file_path: str
) -> None:
    """Attach a call-log sink to *result* if one is configured.

    Two ways to declare one, and a config may use either:

    - ``call_log``, a ``backend``/``path`` block, which builds
      one of the framework's own sinks. This is what a config
      needs to log to a file at all — a dotted path cannot carry
      a destination, which is why no catalog config logged
      anything before the block existed.
    - ``call_logger``, a dotted path to a callable the caller
      has built themselves.

    Declaring both is a contradiction rather than a merge, so it
    is rejected instead of one silently winning.

    :param config: The raw YAML config.
    :param result: The kwargs dict to add the callable to.
    :param file_path: Path to the YAML file, which a relative
        ``path`` is resolved against.
    :raises AgentConfigurationError: If both keys are declared.
    """
    logger_path = config.get("call_logger")
    declared = "call_log" in config
    if logger_path and declared:
        raise AgentConfigurationError(
            "Declare either `call_log` or `call_logger`, not both "
            f"— `{file_path}` declares both."
        )
    if logger_path:
        result["call_logger"] = import_from_path(logger_path)
        return
    if not declared:
        return
    # A bare `call_log:` and an empty `call_log: {}` both mean
    # "log, with the defaults" — writing the key at all is the
    # request, and reading either as "no log" would leave a
    # config that plainly asks for one silently writing nothing.
    settings = config["call_log"] or {}
    if not isinstance(settings, dict):
        raise AgentConfigurationError(
            f"`call_log` must be a mapping with `backend` and "
            f"`path`, got {type(settings).__name__} in `{file_path}`."
        )
    from agents7.call_log import CallLogConfig, create_call_logger

    backend = settings.get("backend", "json")
    # The backend's own default is a bare filename, so it would
    # follow the working directory and a config loaded by a web
    # backend would log somewhere else than the same config
    # loaded by a script. Anchor it here, as `memory` is.
    raw_path = settings.get("path") or (
        "calls.db" if backend == "sqlite" else "calls.jsonl"
    )
    result["call_logger"] = create_call_logger(
        CallLogConfig(
            backend=backend,
            path=_resolve_store_path(raw_path, file_path),
        )
    )


def load_chatter_yaml_settings(file_path: str) -> dict:
    """Load Chatter agent settings from a YAML file.

    :param file_path: The path to the YAML file. Schema:
        ```yaml
        base_config: <path to a base YAML config>
        character: <path to a YAML file holding `persona` and
                    the character's own instructions>
        chat_goal: <path to a YAML file holding `scene`, the
                    opening behaviour, and the goal's own
                    instructions>
        name: <name of the agent>
        version: <version of the agent>
        engine:
            provider: <llm provider>
            model: <llm model>
        system_prompt: <system prompt / personality>
        temperature: <optional, default 0.7>
        think: <optional, default false>
        max_history: <optional, default 50>
        call_log:
            backend: <json | sqlite, default json>
            path: <optional; a relative one sits beside this
                   config file>
        call_logger: <optional dotted path to a callable
                      taking one CallRecord per LLM call;
                      an alternative to call_log, not an
                      addition to it>
        prompt_params:
            <key>: <value>
        context_params:
            <key>:
                default: <fallback value>
                description: <what it means>
                callable: <dotted path, called each turn>
                from: <another key, passed to the callable>
        prompt_variants:
            <key>:
                - <text, one drawn per conversation>
                - <another text>
        character_facts:
            path: <folder for the fact files; a relative one
                   sits beside this config file>
            character_id: <key for this character>
            banned_phrases: <phrases no fact may carry>
            extractor:
                engine:
                    provider: <llm provider>
                    model: <llm model>
        ```
        The fields `base_config`, `character`, `chat_goal`,
        `prompt_params`, `context_params`, `prompt_variants`,
        `character_facts`, `temperature`, `think`, and
        `max_history` are optional.

        `character` and `chat_goal` are the two axes of a
        role-play agent, each in its own file so either can be
        swapped without touching the other. A `character` file
        holds `persona` and may not hold `scene`; a `chat_goal`
        file holds `scene` and may not hold `persona`. Both may
        carry `extra_instructions` / `opening_instructions`, and
        those accumulate — character first, then chat_goal, then
        whatever this file says — rather than one replacing the
        other. Every other key is a plain deep merge with this
        file winning. Paths are relative to this file.
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

    prompt_params = _parse_prompt_params(config)

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
        "context_params": parse_context_params(config, file_path),
        "prompt_variants": parse_prompt_variants(config, file_path),
    }

    if has_persona or has_scene:
        if has_persona:
            result["persona"] = _build_persona(config["persona"], file_path)
        if has_scene:
            result["scene"] = _build_scene(config["scene"], file_path)
        if extra := config.get("extra_instructions"):
            result["extra_instructions"] = str(extra).strip()

    cast_optional_fields(config, result, _CHATTER_OPTIONAL_FIELDS)
    _maybe_add_call_logger(config, result, file_path)
    if end_condition_path := config.get("end_condition"):
        result["end_condition"] = import_from_path(end_condition_path)
    _maybe_add_memory(config, result, file_path)
    _maybe_add_facts(config, result, file_path)

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
        "prompt_params": _parse_prompt_params(raw),
        "context_params": parse_context_params(raw, file_path),
        "prompt_variants": parse_prompt_variants(raw, file_path),
    }
    cast_optional_fields(raw, config, _GUARDED_OPTIONAL_FIELDS)
    _maybe_add_call_logger(raw, config, file_path)
    _maybe_add_memory(raw, config, file_path)
    _maybe_add_facts(raw, config, file_path)

    return GuardedChatter(**config)


def load_tool_agent_yaml_settings(file_path: str) -> dict:
    """Load `ToolAgent` settings from a YAML file.

    Schema: the Chatter base keys (``name``, ``version``,
    ``engine``, ``system_prompt``) plus an optional ``tools`` list
    of import paths and optional ``max_iterations`` / ``temperature``.

    :param file_path: Path to the YAML configuration file.
    :return: A kwargs dict for `ToolAgent`.
    """
    config = _resolve_inheritance(file_path)
    validate_chatter_yaml_config(config, file_path)
    tools = [import_from_path(p) for p in config.get("tools", [])]
    result = {
        "name": config["name"],
        "version": config["version"],
        "engine": build_engine(config["engine"]),
        "system_prompt": config["system_prompt"].strip(),
        "tools": tools,
        "prompt_params": _parse_prompt_params(config),
        "context_params": parse_context_params(config, file_path),
        "prompt_variants": parse_prompt_variants(config, file_path),
    }
    cast_optional_fields(
        config,
        result,
        {"max_iterations": int, "temperature": float},
    )
    return result


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
        "prompt_params": _parse_prompt_params(raw),
        "context_params": parse_context_params(raw, file_path),
        "prompt_variants": parse_prompt_variants(raw, file_path),
    }
    cast_optional_fields(raw, config, _GUARDED_OPTIONAL_FIELDS)
    _maybe_add_call_logger(raw, config, file_path)
    _maybe_add_memory(raw, config, file_path)
    _maybe_add_facts(raw, config, file_path)

    return GuardedSceneChatter(**config)
