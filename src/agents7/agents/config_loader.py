"""YAML configuration loading for agents."""

from cpinto.files import load_yaml

from agents7.agents.utils import (
    get_examples_str,
    import_from_path,
)
from agents7.exceptions import AgentConfigurationError
from agents7.llm import LLMEngine
from agents7.llm.cache import CacheConfig
from agents7.llm.retry import RetryConfig


def load_yaml_settings(file_path: str) -> dict:
    """Load agent settings from a YAML file.

    :param file_path: The path to the YAML file. Schema:
        ```yaml
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
        The fields `prompt_params`, `system_examples`,
        `think`, and `fallback_model` are optional.
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
    config = load_yaml(file_path)
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

    if "think" in config:
        result["think"] = bool(config["think"])

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

    return LLMEngine(
        engine_config["provider"],
        engine_config["model"],
        retry=retry_config,
        cache=cache_config,
        fallback_model=engine_config.get("fallback_model"),
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
