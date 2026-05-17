"""Unit tests for LLMEngine initialization and provider wiring."""

from unittest.mock import Mock, patch

import pytest

from agents7.llm.base import LLMEngine


# ── Initialization ─────────────────────────────────────────────────────────


def test_llm_engine_attributes():
    """LLMEngine stores provider, model_name, and model_id."""
    with patch.object(LLMEngine, "_create_provider", return_value=Mock()):
        engine = LLMEngine(provider="ollama", model_name="qwen3")
    assert engine.provider == "ollama"
    assert engine.model_name == "qwen3"
    assert engine.model_id == "ollama::qwen3"


def test_llm_engine_provider_lowercased():
    """provider is normalised to lowercase during init."""
    with patch.object(LLMEngine, "_create_provider", return_value=Mock()):
        engine = LLMEngine(provider="Anthropic", model_name="claude-3")
    assert engine.provider == "anthropic"


def test_llm_engine_no_fallback_by_default():
    """Without fallback_model, _fallback_provider is None."""
    with patch.object(LLMEngine, "_create_provider", return_value=Mock()):
        engine = LLMEngine(provider="ollama", model_name="qwen3")
    assert engine._fallback_provider is None


def test_llm_engine_fallback_wired():
    """fallback_model wires a second provider as the fallback."""
    primary = Mock()
    fallback = Mock()
    with patch.object(
        LLMEngine,
        "_create_provider",
        side_effect=[primary, fallback],
    ):
        engine = LLMEngine(
            provider="ollama",
            model_name="qwen3",
            fallback_model="llama3",
        )
    assert engine._fallback_provider is fallback


def test_llm_engine_supports_warmup_ollama():
    """supports_warmup() is True for ollama."""
    with patch.object(LLMEngine, "_create_provider", return_value=Mock()):
        engine = LLMEngine(provider="ollama", model_name="qwen3")
    assert engine.supports_warmup() is True


def test_llm_engine_supports_warmup_non_ollama():
    """supports_warmup() is False for non-ollama providers."""
    with patch.object(LLMEngine, "_create_provider", return_value=Mock()):
        engine = LLMEngine(provider="openai_compat", model_name="gpt-4")
    assert engine.supports_warmup() is False


# ── _create_provider ───────────────────────────────────────────────────────


def test_create_provider_unknown_raises():
    """Unknown provider string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown provider"):
        LLMEngine._create_provider("bogus", "model")


def test_create_provider_ollama():
    """_create_provider('ollama', ...) instantiates OllamaProvider."""
    with patch("agents7.llm.ollama.OllamaProvider") as mock_cls:
        result = LLMEngine._create_provider("ollama", "qwen3")
    mock_cls.assert_called_once_with("qwen3")
    assert result is mock_cls.return_value


def test_create_provider_anthropic():
    """_create_provider('anthropic', ...) instantiates AnthropicProvider."""
    with patch("agents7.llm.anthropic.AnthropicProvider") as mock_cls:
        result = LLMEngine._create_provider("anthropic", "claude-3")
    mock_cls.assert_called_once_with(
        "claude-3",
        api_key=None,
        base_url=None,
        default_max_tokens=4096,
    )
    assert result is mock_cls.return_value


def test_create_provider_openai_compat():
    """_create_provider('openai_compat', ...) instantiates OpenAICompatProvider."""
    with patch("agents7.llm.openai_compat.OpenAICompatProvider") as mock_cls:
        result = LLMEngine._create_provider(
            "openai_compat",
            "gpt-4",
            base_url="http://localhost:11434/v1",
        )
    mock_cls.assert_called_once_with(
        "gpt-4",
        base_url="http://localhost:11434/v1",
        api_key=None,
    )
    assert result is mock_cls.return_value
