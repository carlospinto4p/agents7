"""Unit tests for agents7.agents.config_loader.

Covers load_yaml_settings(), build_engine(), and
validate_yaml_config(). _deep_merge() and _resolve_inheritance()
are covered in test_config_inheritance.py.
"""

import pytest

from agents7.agents.config_loader import (
    _resolve_env_var,
    build_engine,
    load_yaml_settings,
    validate_yaml_config,
)
from agents7.exceptions import AgentConfigurationError
from agents7.llm import LLMEngine
from agents7.llm.retry import RetryConfig


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
    """Engine built without retry/cache has None for both."""
    engine = build_engine({"provider": "ollama", "model": "qwen3"})
    assert engine.retry_config is None
    assert engine._cache is None


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
