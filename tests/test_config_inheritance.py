"""Unit tests for YAML config inheritance (base_config)."""

import pytest

from agents7.agents.config_loader import (
    _deep_merge,
    _resolve_inheritance,
)
from agents7.exceptions import AgentConfigurationError


# --- _deep_merge ---


def test_deep_merge_flat():
    """Override replaces base keys."""
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    assert _deep_merge(base, override) == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested():
    """Nested dicts are merged recursively."""
    base = {"engine": {"provider": "ollama", "model": "qwen3"}}
    override = {"engine": {"model": "llama3"}}
    result = _deep_merge(base, override)
    assert result == {"engine": {"provider": "ollama", "model": "llama3"}}


def test_deep_merge_does_not_mutate():
    """Inputs are not modified."""
    base = {"a": {"x": 1}}
    override = {"a": {"y": 2}}
    _deep_merge(base, override)
    assert base == {"a": {"x": 1}}
    assert override == {"a": {"y": 2}}


# --- _resolve_inheritance ---


def test_no_base_config(tmp_path):
    """Config without base_config loads normally."""
    cfg = tmp_path / "agent.yml"
    cfg.write_text("name: TestAgent\nversion: 1\n")
    result = _resolve_inheritance(str(cfg))
    assert result == {"name": "TestAgent", "version": 1}


def test_single_inheritance(tmp_path):
    """Child inherits and overrides base keys."""
    (tmp_path / "base.yml").write_text(
        "name: Base\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Base prompt.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
    )

    (tmp_path / "child.yml").write_text(
        "base_config: base.yml\nname: Child\nversion: 2\n"
    )

    result = _resolve_inheritance(str(tmp_path / "child.yml"))
    assert result["name"] == "Child"
    assert result["version"] == 2
    assert result["engine"]["model"] == "qwen3"
    assert result["system_prompt"] == "Base prompt."


def test_nested_engine_merge(tmp_path):
    """Child can override engine sub-keys."""
    (tmp_path / "base.yml").write_text(
        "name: Base\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
    )

    (tmp_path / "child.yml").write_text(
        "base_config: base.yml\nengine:\n    model: llama3\n"
    )

    result = _resolve_inheritance(str(tmp_path / "child.yml"))
    assert result["engine"]["provider"] == "ollama"
    assert result["engine"]["model"] == "llama3"


def test_chained_inheritance(tmp_path):
    """A → B → C chain resolves correctly."""
    (tmp_path / "gp.yml").write_text(
        "name: GP\nversion: 1\nlevel: grandparent\n"
    )
    (tmp_path / "parent.yml").write_text(
        "base_config: gp.yml\nname: Parent\nlevel: parent\n"
    )
    (tmp_path / "child.yml").write_text(
        "base_config: parent.yml\nname: Child\n"
    )

    result = _resolve_inheritance(str(tmp_path / "child.yml"))
    assert result["name"] == "Child"
    assert result["version"] == 1
    assert result["level"] == "parent"


def test_circular_reference_raises(tmp_path):
    """Circular base_config raises AgentConfigurationError."""
    (tmp_path / "a.yml").write_text("base_config: b.yml\nname: A\n")
    (tmp_path / "b.yml").write_text("base_config: a.yml\nname: B\n")

    with pytest.raises(AgentConfigurationError, match="Circular"):
        _resolve_inheritance(str(tmp_path / "a.yml"))


def test_self_reference_raises(tmp_path):
    """Self-referencing base_config raises error."""
    (tmp_path / "self.yml").write_text("base_config: self.yml\nname: Self\n")

    with pytest.raises(AgentConfigurationError, match="Circular"):
        _resolve_inheritance(str(tmp_path / "self.yml"))


def test_absolute_base_config_raises(tmp_path):
    """Absolute path in base_config raises AgentConfigurationError."""
    (tmp_path / "base.yml").write_text("name: Base\n")
    child = tmp_path / "child.yml"
    child.write_text(f"base_config: {tmp_path / 'base.yml'}\nname: Child\n")

    with pytest.raises(AgentConfigurationError, match="must be relative"):
        _resolve_inheritance(str(child))


def test_relative_base_path(tmp_path):
    """Relative base_config path resolves from file dir."""
    sub = tmp_path / "sub"
    sub.mkdir()

    base = tmp_path / "base.yml"
    base.write_text("name: Base\nversion: 1\n")

    child = sub / "child.yml"
    child.write_text("base_config: ../base.yml\nname: Child\n")

    result = _resolve_inheritance(str(child))
    assert result["name"] == "Child"
    assert result["version"] == 1


def test_base_config_removed_from_result(tmp_path):
    """base_config key is not in the merged result."""
    (tmp_path / "base.yml").write_text("name: Base\n")
    (tmp_path / "child.yml").write_text("base_config: base.yml\nversion: 1\n")

    result = _resolve_inheritance(str(tmp_path / "child.yml"))
    assert "base_config" not in result
