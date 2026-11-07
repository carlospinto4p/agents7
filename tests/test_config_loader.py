"""Unit tests for agents7.agents.config_loader.

Covers load_yaml_settings(), build_engine(), validate_yaml_config(),
parse_yaml_examples(), and cast_optional_fields().
_deep_merge() and _resolve_inheritance() are covered in
test_config_inheritance.py.
"""

import json

import pytest

from agents7.call_log import JsonCallLogger, SqliteCallLogger
from agents7.agents.config_loader import (
    _resolve_env_var,
    build_engine,
    cast_optional_fields,
    load_yaml_settings,
    parse_yaml_examples,
    validate_yaml_config,
)
from agents7.exceptions import AgentConfigurationError
from agents7.llm.pricing import Pricing
from agents7.llm import LLMEngine
from agents7.llm.retry import DEFAULT_RETRY, RetryConfig


# ── helpers ───────────────────────────────────────────────────────

_MINIMAL_YAML = """\
name: TestAgent
version: 1
engine:
    provider: ollama
    model: qwen3
system_prompt: Test prompt.
output_model: agents7.models.demo.PersonInfo
"""


def _write_yaml(tmp_path, content: str):
    p = tmp_path / "config.yml"
    p.write_text(content)
    return str(p)


# ── validate_yaml_config ──────────────────────────────────────────


@pytest.mark.parametrize(
    "missing_key",
    ["name", "version", "engine", "system_prompt", "output_model"],
)
def test_validate_missing_top_level_key_raises(missing_key):
    """Missing mandatory top-level key raises AgentConfigurationError."""
    config = {
        "name": "A",
        "version": 1,
        "engine": {"provider": "ollama", "model": "qwen3"},
        "system_prompt": "Prompt.",
        "output_model": "agents7.models.demo.PersonInfo",
    }
    del config[missing_key]
    with pytest.raises(AgentConfigurationError, match=missing_key):
        validate_yaml_config(config, "dummy.yml")


@pytest.mark.parametrize("missing_key", ["provider", "model"])
def test_validate_missing_engine_key_raises(missing_key):
    """Missing engine sub-key raises AgentConfigurationError."""
    config = {
        "name": "A",
        "version": 1,
        "engine": {"provider": "ollama", "model": "qwen3"},
        "system_prompt": "Prompt.",
        "output_model": "agents7.models.demo.PersonInfo",
    }
    del config["engine"][missing_key]
    with pytest.raises(AgentConfigurationError, match=missing_key):
        validate_yaml_config(config, "dummy.yml")


# ── build_engine ──────────────────────────────────────────────────


def test_build_engine_minimal():
    """Minimal config creates an LLMEngine with correct provider/model."""
    engine = build_engine({"provider": "ollama", "model": "qwen3"})
    assert isinstance(engine, LLMEngine)
    assert engine.provider == "ollama"
    assert engine.model_name == "qwen3"


def test_build_engine_with_retry():
    """Retry settings produce a RetryConfig on the engine."""
    engine = build_engine(
        {
            "provider": "ollama",
            "model": "qwen3",
            "retry": {"max_retries": 5, "base_delay": 2.0},
        }
    )
    assert isinstance(engine.retry_config, RetryConfig)
    assert engine.retry_config.max_retries == 5
    assert engine.retry_config.base_delay == 2.0


def test_build_engine_with_cache():
    """Cache settings produce a CacheConfig on the dispatcher."""
    engine = build_engine(
        {
            "provider": "ollama",
            "model": "qwen3",
            "cache": {"max_size": 64},
        }
    )
    assert engine._cache is not None


def test_build_engine_no_retry_no_cache():
    """No retry section means the default, and no cache."""
    engine = build_engine({"provider": "ollama", "model": "qwen3"})
    assert engine.retry_config == DEFAULT_RETRY
    assert engine._cache is None


def test_build_engine_retry_can_be_switched_off():
    """`max_retries: 0` is how a config asks for no retrying."""
    engine = build_engine(
        {
            "provider": "ollama",
            "model": "qwen3",
            "retry": {"max_retries": 0},
        }
    )
    assert engine.retry_config.max_retries == 0


def test_build_engine_with_fallback_model():
    """fallback_model is wired into the engine dispatcher."""
    engine = build_engine(
        {
            "provider": "ollama",
            "model": "qwen3",
            "fallback_model": "llama3",
        }
    )
    assert engine._fallback_provider is not None


# ── load_yaml_settings ────────────────────────────────────────────


def test_load_yaml_settings_returns_required_keys(tmp_path):
    """Valid YAML produces a dict with all required keys."""
    path = _write_yaml(tmp_path, _MINIMAL_YAML)
    result = load_yaml_settings(path)
    assert result["name"] == "TestAgent"
    assert result["version"] == 1
    assert isinstance(result["engine"], LLMEngine)
    assert result["system_prompt"] == "Test prompt."
    assert result["prompt_params"] is None
    assert result["examples"] is None
    assert result["validators"] is None


def test_load_yaml_settings_strips_system_prompt(tmp_path):
    """System prompt whitespace is stripped."""
    yaml = _MINIMAL_YAML.replace(
        "system_prompt: Test prompt.",
        "system_prompt: |  \n  Hello world.  \n",
    )
    result = load_yaml_settings(_write_yaml(tmp_path, yaml))
    assert result["system_prompt"] == "Hello world."


def test_load_yaml_settings_with_prompt_params(tmp_path):
    """prompt_params are loaded as string:string dict."""
    yaml = _MINIMAL_YAML + "prompt_params:\n    lang: Spanish\n    n: 3\n"
    result = load_yaml_settings(_write_yaml(tmp_path, yaml))
    assert result["prompt_params"] == {"lang": "Spanish", "n": "3"}


def test_load_yaml_settings_missing_key_raises(tmp_path):
    """YAML missing a mandatory key raises AgentConfigurationError."""
    yaml = _MINIMAL_YAML.replace("name: TestAgent\n", "")
    with pytest.raises(AgentConfigurationError):
        load_yaml_settings(_write_yaml(tmp_path, yaml))


def test_load_yaml_settings_optional_think(tmp_path):
    """Optional think field is cast to bool and stored."""
    yaml = _MINIMAL_YAML + "think: false\n"
    result = load_yaml_settings(_write_yaml(tmp_path, yaml))
    assert result["think"] is False


def test_load_yaml_settings_optional_max_parse_retries(tmp_path):
    """Optional max_parse_retries is cast to int."""
    yaml = _MINIMAL_YAML + "max_parse_retries: 2\n"
    result = load_yaml_settings(_write_yaml(tmp_path, yaml))
    assert result["max_parse_retries"] == 2


def test_load_yaml_settings_empty_prompt_params_treated_as_none(tmp_path):
    """Empty prompt_params dict is falsy and treated as absent."""
    yaml = _MINIMAL_YAML + "prompt_params: {}\n"
    result = load_yaml_settings(_write_yaml(tmp_path, yaml))
    assert result["prompt_params"] is None


def test_load_yaml_settings_malformed_examples_raise(tmp_path):
    """system_examples with a missing key raises AgentConfigurationError."""
    yaml = (
        _MINIMAL_YAML
        + "system_examples:\n"
        + "  - input: hi\n"
        + "    wrong_key: hello\n"
    )
    with pytest.raises(AgentConfigurationError):
        load_yaml_settings(_write_yaml(tmp_path, yaml))


# ── parse_yaml_examples ───────────────────────────────────────────


def test_parse_yaml_examples_absent_returns_none():
    """Returns None when system_examples is not in config."""
    result = parse_yaml_examples({}, "dummy.yml")
    assert result is None


def test_parse_yaml_examples_valid_returns_list():
    """Valid examples list is returned unchanged."""
    examples = [{"input": "hi", "output": "hello"}]
    result = parse_yaml_examples({"system_examples": examples}, "dummy.yml")
    assert result == examples


def test_parse_yaml_examples_not_a_list_raises():
    """Non-list system_examples raises AgentConfigurationError."""
    with pytest.raises(AgentConfigurationError, match="list"):
        parse_yaml_examples({"system_examples": "not a list"}, "dummy.yml")


def test_parse_yaml_examples_example_not_dict_raises():
    """Example that is not a dict raises AgentConfigurationError."""
    with pytest.raises(AgentConfigurationError):
        parse_yaml_examples(
            {"system_examples": ["just a string"]}, "dummy.yml"
        )


def test_parse_yaml_examples_missing_output_key_raises():
    """Example missing the output key raises AgentConfigurationError."""
    with pytest.raises(AgentConfigurationError):
        parse_yaml_examples(
            {"system_examples": [{"input": "hi"}]}, "dummy.yml"
        )


def test_parse_yaml_examples_extra_key_raises():
    """Example with extra keys raises AgentConfigurationError."""
    with pytest.raises(AgentConfigurationError):
        parse_yaml_examples(
            {
                "system_examples": [
                    {"input": "hi", "output": "hello", "extra": "bad"}
                ]
            },
            "dummy.yml",
        )


# ── cast_optional_fields ──────────────────────────────────────────


def test_cast_optional_fields_copies_present_fields():
    """Present fields are cast and copied to target."""
    source = {"temperature": "0.7", "max_history": "50"}
    target: dict = {}
    cast_optional_fields(
        source, target, {"temperature": float, "max_history": int}
    )
    assert target["temperature"] == 0.7
    assert target["max_history"] == 50


def test_cast_optional_fields_skips_absent_fields():
    """Absent fields are not added to target."""
    source = {"temperature": "0.5"}
    target: dict = {}
    cast_optional_fields(
        source, target, {"temperature": float, "think": bool}
    )
    assert "think" not in target
    assert target["temperature"] == 0.5


def test_cast_optional_fields_does_not_modify_source():
    """Source dict is not mutated."""
    source = {"x": "1"}
    original = dict(source)
    cast_optional_fields(source, {}, {"x": int})
    assert source == original


# ── _resolve_env_var ──────────────────────────────────────────────


def test_resolve_env_var_plain_string():
    """Non-reference strings are returned unchanged."""
    assert _resolve_env_var("plain-value") == "plain-value"


def test_resolve_env_var_resolves_set_var(monkeypatch):
    """${VAR} syntax resolves to the env var value."""
    monkeypatch.setenv("TEST_API_KEY", "secret123")
    assert _resolve_env_var("${TEST_API_KEY}") == "secret123"


def test_resolve_env_var_missing_raises(monkeypatch):
    """${VAR} where VAR is unset raises AgentConfigurationError."""
    monkeypatch.delenv("MISSING_VAR", raising=False)
    with pytest.raises(AgentConfigurationError, match="MISSING_VAR"):
        _resolve_env_var("${MISSING_VAR}")


def test_resolve_env_var_partial_reference_unchanged():
    """Strings that look like ${...} but are malformed pass through."""
    assert _resolve_env_var("${no-closing") == "${no-closing"
    assert _resolve_env_var("no-opening}") == "no-opening}"


# ── load_yaml_settings with inheritance ──────────────────────────


def test_load_yaml_settings_with_base_config(tmp_path):
    """load_yaml_settings() resolves a base_config chain end-to-end."""
    base = tmp_path / "base.yml"
    base.write_text(
        "name: BaseAgent\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Base prompt.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
    )
    child = tmp_path / "child.yml"
    child.write_text("base_config: base.yml\nname: ChildAgent\nversion: 2\n")

    result = load_yaml_settings(str(child))

    assert result["name"] == "ChildAgent"
    assert result["version"] == 2
    assert result["system_prompt"] == "Base prompt."
    assert isinstance(result["engine"], LLMEngine)
    assert result["engine"].model_name == "qwen3"


# ── variable substitution in engine section ───────────────────────


def test_build_engine_resolves_env_var_in_api_key(monkeypatch):
    """build_engine resolves ${VAR} in engine.api_key."""
    monkeypatch.setenv("MY_API_KEY_TEST", "sk-secret")
    engine = build_engine(
        {
            "provider": "ollama",
            "model": "qwen3",
            "api_key": "${MY_API_KEY_TEST}",
        }
    )
    assert isinstance(engine, LLMEngine)


def test_build_engine_missing_env_var_raises(monkeypatch):
    """build_engine raises AgentConfigurationError for unset ${VAR}."""
    monkeypatch.delenv("UNSET_VAR_XYZ", raising=False)
    with pytest.raises(AgentConfigurationError, match="UNSET_VAR_XYZ"):
        build_engine(
            {
                "provider": "ollama",
                "model": "qwen3",
                "api_key": "${UNSET_VAR_XYZ}",
            }
        )


# ── build_engine: pricing ──────────────────────────────────────────


def test_build_engine_without_pricing():
    """No `pricing` section leaves cost estimation off."""
    engine = build_engine({"provider": "ollama", "model": "qwen3"})

    assert engine.pricing is None


def test_build_engine_with_pricing():
    """Rates are read as dollars per million tokens."""
    engine = build_engine(
        {
            "provider": "ollama",
            "model": "qwen3",
            "pricing": {
                "input": 0.44,
                "output": 1.32,
                "cached_input": 0.014,
            },
        }
    )

    assert engine.pricing == Pricing(
        input=0.44, output=1.32, cached_input=0.014
    )


def test_build_engine_pricing_cached_input_optional():
    engine = build_engine(
        {
            "provider": "ollama",
            "model": "qwen3",
            "pricing": {"input": 0.44, "output": 1.32},
        }
    )

    assert engine.pricing.cached_input == 0.44


@pytest.mark.parametrize(
    "pricing, message",
    [
        ({"output": 1.0}, "missing `input`"),
        ({"input": 1.0}, "missing `output`"),
        ({"input": 1.0, "output": 1.0, "nope": 1.0}, "unknown key"),
        ({"input": "cheap", "output": 1.0}, "must be a number"),
        ({"input": True, "output": 1.0}, "must be a number"),
        ({"input": -1.0, "output": 1.0}, "invalid"),
    ],
)
def test_build_engine_rejects_bad_pricing(pricing, message):
    """A malformed price is a config error, not a silent zero."""
    with pytest.raises(AgentConfigurationError, match=message):
        build_engine(
            {
                "provider": "ollama",
                "model": "qwen3",
                "pricing": pricing,
            }
        )


def test_build_engine_rejects_non_mapping_pricing():
    with pytest.raises(AgentConfigurationError, match="must be a mapping"):
        build_engine(
            {"provider": "ollama", "model": "qwen3", "pricing": 0.44}
        )


# ── call_log block ────────────────────────────────────────────────


def test_call_log_block_builds_a_sqlite_sink(tmp_path):
    """A config can now name a destination, not just a callable.

    `call_logger` takes a dotted path, so a YAML-declared agent
    could only log to a sink some module had already built and
    exported — which is why no config in the catalog logged
    anything at all.
    """
    path = _write_yaml(
        tmp_path,
        _MINIMAL_YAML + "call_log:\n    backend: sqlite\n",
    )

    logger = load_yaml_settings(path)["call_logger"]

    assert isinstance(logger, SqliteCallLogger)


def test_call_log_path_sits_beside_the_config(tmp_path):
    """A relative path belongs to the agent, not to the caller.

    Left relative it would follow the working directory, so the
    same agent would log to one file under a web backend and
    another under a script — the hazard `_resolve_store_path`
    exists for, applied to the call log as it already was to
    memory.
    """
    path = _write_yaml(
        tmp_path,
        _MINIMAL_YAML + "call_log:\n    path: data/calls.db\n",
    )

    logger = load_yaml_settings(path)["call_logger"]

    assert logger._path == tmp_path / "data" / "calls.db"


@pytest.mark.parametrize("block", ["call_log:\n", "call_log: {}\n"])
def test_call_log_defaults_to_a_json_file_beside_the_config(tmp_path, block):
    """Writing the key at all is the request to log.

    An empty block is falsy, and reading it as "no log" would
    leave a config that plainly asks for one writing nothing —
    a silence with no error behind it.
    """
    path = _write_yaml(tmp_path, _MINIMAL_YAML + block)

    logger = load_yaml_settings(path)["call_logger"]

    assert isinstance(logger, JsonCallLogger)
    assert logger._path == tmp_path / "calls.jsonl"


def test_declaring_both_call_log_and_call_logger_raises(tmp_path):
    """Two destinations is a contradiction, not a merge.

    Only one sink can be attached, so letting one win silently
    would mean a config that reads as logging to a database
    quietly writing somewhere else.
    """
    path = _write_yaml(
        tmp_path,
        _MINIMAL_YAML
        + "call_log:\n    backend: sqlite\ncall_logger: json.dumps\n",
    )

    with pytest.raises(AgentConfigurationError, match="not both"):
        load_yaml_settings(path)


def test_an_unknown_call_log_backend_raises(tmp_path):
    """Named backends only, as memory and the cache already do."""
    path = _write_yaml(
        tmp_path,
        _MINIMAL_YAML + "call_log:\n    backend: postgres\n",
    )

    with pytest.raises(AgentConfigurationError, match="postgres"):
        load_yaml_settings(path)


def test_a_dotted_call_logger_still_works(tmp_path):
    """The existing form is an alternative, not a casualty."""
    path = _write_yaml(tmp_path, _MINIMAL_YAML + "call_logger: json.dumps\n")

    assert load_yaml_settings(path)["call_logger"] is json.dumps


# ── build_engine: fallback_engine ─────────────────────────────────
#
# The YAML half of the cross-provider fallback. `fallback_model`
# is a bare model name on the primary's own provider; a
# `fallback_engine` section is a whole engine, so it can name any
# provider and carries its own credentials, pricing and retry.


def test_build_engine_fallback_engine_crosses_providers():
    """A paid primary can fall back to a local model."""
    engine = build_engine(
        {
            "provider": "openai_compat",
            "model": "deepseek-chat",
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "fallback_engine": {
                "provider": "ollama",
                "model": "gemma3:12b",
            },
        }
    )

    assert engine.provider == "openai_compat"
    assert engine.fallback_engine.provider == "ollama"
    assert engine.fallback_engine.model_name == "gemma3:12b"
    assert engine._fallback_provider is engine.fallback_engine.llm_provider


def test_build_engine_fallback_engine_carries_its_own_pricing():
    """The fallback's rates price a response it answered."""
    engine = build_engine(
        {
            "provider": "openai_compat",
            "model": "deepseek-chat",
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "pricing": {"input": 0.28, "output": 0.42},
            "fallback_engine": {
                "provider": "ollama",
                "model": "gemma3:12b",
                "pricing": {"input": 0.0, "output": 0.0},
            },
        }
    )

    assert engine._dispatcher._pricing == Pricing(input=0.28, output=0.42)
    assert engine._dispatcher._fallback_pricing == Pricing(
        input=0.0, output=0.0
    )


def test_build_engine_fallback_engine_carries_its_own_retry():
    """The nested section's retry is the fallback's own budget."""
    engine = build_engine(
        {
            "provider": "ollama",
            "model": "qwen3",
            "retry": {"max_retries": 1},
            "fallback_engine": {
                "provider": "ollama",
                "model": "gemma3:12b",
                "retry": {"max_retries": 4},
            },
        }
    )

    assert engine._dispatcher._retry_config.max_retries == 1
    assert engine._dispatcher._fallback_retry.max_retries == 4


def test_build_engine_fallback_engine_must_be_a_mapping():
    """A bare model name in that key is a config mistake."""
    with pytest.raises(AgentConfigurationError, match="engine section"):
        build_engine(
            {
                "provider": "ollama",
                "model": "qwen3",
                "fallback_engine": "gemma3:12b",
            }
        )


def test_build_engine_fallback_engine_rejects_a_cache():
    """Rejected rather than ignored — it would never be read."""
    with pytest.raises(AgentConfigurationError, match="`cache`"):
        build_engine(
            {
                "provider": "ollama",
                "model": "qwen3",
                "fallback_engine": {
                    "provider": "ollama",
                    "model": "gemma3:12b",
                    "cache": {"max_size": 64},
                },
            }
        )


def test_build_engine_without_a_fallback_engine():
    """Omitting the section leaves the engine with no fallback."""
    engine = build_engine({"provider": "ollama", "model": "qwen3"})

    assert engine.fallback_engine is None
    assert engine._fallback_provider is None
