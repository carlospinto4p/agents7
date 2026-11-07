"""Unit tests for LLMEngine initialization and provider wiring."""

from unittest.mock import Mock, patch

import pytest

from agents7.llm.base import LLMEngine
from agents7.llm.pricing import Pricing


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
    mock_cls.assert_called_once_with("qwen3", timeout=None)
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
        timeout=None,
    )
    assert result is mock_cls.return_value


def test_create_provider_openai_compat():
    """_create_provider('openai_compat', ...) instantiates
    OpenAICompatProvider."""
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
        timeout=None,
    )
    assert result is mock_cls.return_value


# ── Pricing ────────────────────────────────────────────────────────────────


def test_pricing_defaults_to_none():
    """Without pricing, an engine estimates no cost."""
    engine = LLMEngine("ollama", "test-model")

    assert engine.pricing is None


def test_pricing_is_passed_through():
    """A `Pricing` reaches the engine unchanged."""
    pricing = Pricing(input=0.44, output=1.32, cached_input=0.014)
    engine = LLMEngine("ollama", "test-model", pricing=pricing)

    assert engine.pricing is pricing


def test_legacy_flat_rate_is_converted_from_cents():
    """`cost_per_1m_tokens` is cents; `Pricing` is dollars.

    The old parameter has always been cents per million tokens,
    so it must divide by 100 on the way in — otherwise every
    existing caller's costs would jump by 100x.
    """
    engine = LLMEngine("ollama", "test-model", cost_per_1m_tokens=50.0)

    assert engine.pricing == Pricing.flat(0.5)
    # One million tokens at 50 cents/1M is 50 cents, as before.
    assert engine.pricing.cost_cents(
        prompt_tokens=0, tokens=1_000_000
    ) == pytest.approx(50.0)


def test_explicit_pricing_wins_over_legacy_rate():
    """`pricing` takes precedence when both are given."""
    pricing = Pricing(input=1.0, output=2.0)
    engine = LLMEngine(
        "ollama",
        "test-model",
        cost_per_1m_tokens=99.0,
        pricing=pricing,
    )

    assert engine.pricing is pricing


# ── A fallback on another provider ─────────────────────────────────────────
#
# `fallback_model` names a model on the primary's own provider,
# so it can never move a call from a paid endpoint to a local
# one — the case worth having, since an outage or a rate limit
# at a paid provider is exactly when a local model rescues a
# run. `fallback_engine` is a whole engine, so it can.


def test_fallback_engine_uses_its_own_provider():
    """The fallback's provider is the one the dispatcher calls."""
    fallback = LLMEngine("ollama", "gemma3:12b")
    engine = LLMEngine(
        "anthropic",
        "claude-sonnet-5",
        api_key="test-key",
        fallback_engine=fallback,
    )

    assert engine._fallback_provider is fallback.llm_provider
    assert engine.provider == "anthropic"
    assert fallback.provider == "ollama"


def test_fallback_engine_brings_its_own_pricing():
    """A free local fallback is not billed at the paid rates."""
    fallback = LLMEngine("ollama", "gemma3:12b", pricing=Pricing.flat(0.0))
    engine = LLMEngine(
        "anthropic",
        "claude-sonnet-5",
        api_key="test-key",
        pricing=Pricing(input=3.0, output=15.0),
        fallback_engine=fallback,
    )

    assert engine._dispatcher._pricing == Pricing(input=3.0, output=15.0)
    assert engine._dispatcher._fallback_pricing == Pricing.flat(0.0)


def test_fallback_engine_brings_its_own_retry():
    """A local fallback retries on a budget of its own."""
    from agents7.llm.retry import RetryConfig

    fallback = LLMEngine(
        "ollama", "gemma3:12b", retry=RetryConfig(max_retries=5)
    )
    engine = LLMEngine(
        "anthropic",
        "claude-sonnet-5",
        api_key="test-key",
        retry=RetryConfig(max_retries=1),
        fallback_engine=fallback,
    )

    assert engine._dispatcher._retry_config.max_retries == 1
    assert engine._dispatcher._fallback_retry.max_retries == 5


def test_fallback_model_keeps_inheriting_the_primarys_rates():
    """A same-provider fallback has nowhere to declare its own."""
    with patch.object(LLMEngine, "_create_provider", return_value=Mock()):
        engine = LLMEngine(
            "anthropic",
            "claude-sonnet-5",
            pricing=Pricing(input=3.0, output=15.0),
            fallback_model="claude-haiku-4-5-20251001",
        )

    assert engine._dispatcher._fallback_pricing == Pricing(
        input=3.0, output=15.0
    )


def test_both_fallback_shapes_is_rejected():
    """They mean different things, so naming both is a mistake."""
    fallback = LLMEngine("ollama", "gemma3:12b")
    with pytest.raises(ValueError, match="not.*both"):
        LLMEngine(
            "anthropic",
            "claude-sonnet-5",
            api_key="test-key",
            fallback_model="claude-haiku-4-5-20251001",
            fallback_engine=fallback,
        )


def test_fallback_engine_with_a_cache_is_rejected():
    """Rejected rather than ignored — it would never be read."""
    from agents7.llm.cache import CacheConfig

    fallback = LLMEngine("ollama", "gemma3:12b", cache=CacheConfig())
    with pytest.raises(ValueError, match="must not declare a `cache`"):
        LLMEngine(
            "anthropic",
            "claude-sonnet-5",
            api_key="test-key",
            fallback_engine=fallback,
        )


def test_no_fallback_leaves_both_settings_unset():
    """An engine naming no fallback has none of its machinery."""
    with patch.object(LLMEngine, "_create_provider", return_value=Mock()):
        engine = LLMEngine("ollama", "test-model")

    assert engine._fallback_provider is None
    assert engine.fallback_engine is None
