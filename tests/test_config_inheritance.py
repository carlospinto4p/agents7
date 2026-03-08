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
    base = tmp_path / "base.yml"
    base.write_text(
        "name: Base\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: Base prompt.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
    )

    child = tmp_path / "child.yml"
    child.write_text(f"base_config: {base}\nname: Child\nversion: 2\n")

    result = _resolve_inheritance(str(child))
    assert result["name"] == "Child"
    assert result["version"] == 2
    assert result["engine"]["model"] == "qwen3"
    assert result["system_prompt"] == "Base prompt."


def test_nested_engine_merge(tmp_path):
    """Child can override engine sub-keys."""
    base = tmp_path / "base.yml"
    base.write_text(
        "name: Base\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
    )

    child = tmp_path / "child.yml"
    child.write_text(f"base_config: {base}\nengine:\n    model: llama3\n")

    result = _resolve_inheritance(str(child))
    assert result["engine"]["provider"] == "ollama"
    assert result["engine"]["model"] == "llama3"


def test_chained_inheritance(tmp_path):
    """A → B → C chain resolves correctly."""
    grandparent = tmp_path / "gp.yml"
    grandparent.write_text("name: GP\nversion: 1\nlevel: grandparent\n")

    parent = tmp_path / "parent.yml"
    parent.write_text(
        f"base_config: {grandparent}\nname: Parent\nlevel: parent\n"
    )

    child = tmp_path / "child.yml"
    child.write_text(f"base_config: {parent}\nname: Child\n")

    result = _resolve_inheritance(str(child))
    assert result["name"] == "Child"
    assert result["version"] == 1
    assert result["level"] == "parent"


def test_circular_reference_raises(tmp_path):
    """Circular base_config raises AgentConfigurationError."""
    a = tmp_path / "a.yml"
    b = tmp_path / "b.yml"
    a.write_text(f"base_config: {b}\nname: A\n")
    b.write_text(f"base_config: {a}\nname: B\n")

    with pytest.raises(AgentConfigurationError, match="Circular"):
        _resolve_inheritance(str(a))


def test_self_reference_raises(tmp_path):
    """Self-referencing base_config raises error."""
    cfg = tmp_path / "self.yml"
    cfg.write_text(f"base_config: {cfg}\nname: Self\n")

    with pytest.raises(AgentConfigurationError, match="Circular"):
        _resolve_inheritance(str(cfg))


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
    base = tmp_path / "base.yml"
    base.write_text("name: Base\n")

    child = tmp_path / "child.yml"
    child.write_text(f"base_config: {base}\nversion: 1\n")

    result = _resolve_inheritance(str(child))
    assert "base_config" not in result
