"""Unit tests for retry with exponential backoff (no LLM required)."""

from unittest.mock import AsyncMock, Mock, patch

import ollama
import pytest

from agents7.agents.labeler import Labeler
from agents7.exceptions import LLMException
from agents7.llm.retry import RetryConfig, retry_async, retry_sync


# ===================================================================
# 1. RetryConfig validation
# ===================================================================


class TestRetryConfig:
    def test_defaults(self):
        """Default values are 3 retries, 1s base, 30s cap, 2x."""
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.base_delay == 1.0
        assert cfg.max_delay == 30.0
        assert cfg.backoff_factor == 2.0

    def test_custom_values(self):
        """Custom values are stored correctly."""
        cfg = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=60.0,
            backoff_factor=3.0,
        )
        assert cfg.max_retries == 5
        assert cfg.base_delay == 0.5
        assert cfg.max_delay == 60.0
        assert cfg.backoff_factor == 3.0

    def test_negative_max_retries(self):
        """Negative max_retries raises LLMException."""
        with pytest.raises(LLMException, match="max_retries"):
            RetryConfig(max_retries=-1)

    def test_invalid_base_delay(self):
        """Zero base_delay raises LLMException."""
        with pytest.raises(LLMException, match="base_delay"):
            RetryConfig(base_delay=0)

    def test_invalid_backoff_factor(self):
        """backoff_factor < 1 raises LLMException."""
        with pytest.raises(LLMException, match="backoff_factor"):
            RetryConfig(backoff_factor=0.5)


# ===================================================================
# 2. retry_sync logic
# ===================================================================


class TestRetrySyncLogic:
    def test_succeeds_first_attempt(self):
        """No retry needed, returns result directly."""
        fn = Mock(return_value={"response": "ok"})
        cfg = RetryConfig(max_retries=3, base_delay=0.01)

        result = retry_sync(fn, cfg, x=1)

        assert result == {"response": "ok"}
        assert fn.call_count == 1

    @patch("agents7.llm.retry.time.sleep")
    def test_succeeds_after_retry(self, mock_sleep):
        """RequestError on first call, success on second."""
        fn = Mock(
            side_effect=[
                ollama.RequestError("connection refused"),
                {"response": "ok"},
            ]
        )
        cfg = RetryConfig(max_retries=3, base_delay=0.01)

        result = retry_sync(fn, cfg, x=1)

        assert result == {"response": "ok"}
        assert fn.call_count == 2
        mock_sleep.assert_called_once()

    @patch("agents7.llm.retry.time.sleep")
    def test_retries_exhausted(self, mock_sleep):
        """All retries fail, raises LLMException."""
        fn = Mock(side_effect=ollama.RequestError("connection refused"))
        cfg = RetryConfig(max_retries=2, base_delay=0.01)

        with pytest.raises(LLMException, match="retries exhausted"):
            retry_sync(fn, cfg)

        # 1 initial + 2 retries = 3 calls
        assert fn.call_count == 3
        assert mock_sleep.call_count == 2

    def test_non_retryable_error(self):
        """ResponseError(404) raises immediately without retry."""
        error = ollama.ResponseError("not found")
        error.status_code = 404
        fn = Mock(side_effect=error)
        cfg = RetryConfig(max_retries=3, base_delay=0.01)

        with pytest.raises(ollama.ResponseError):
            retry_sync(fn, cfg)

        assert fn.call_count == 1

    @patch("agents7.llm.retry.time.sleep")
    def test_server_error_retried(self, mock_sleep):
        """ResponseError(500) triggers retry, then succeeds."""
        error = ollama.ResponseError("internal server error")
        error.status_code = 500
        fn = Mock(side_effect=[error, {"response": "recovered"}])
        cfg = RetryConfig(max_retries=3, base_delay=0.01)

        result = retry_sync(fn, cfg)

        assert result == {"response": "recovered"}
        assert fn.call_count == 2
        mock_sleep.assert_called_once()

    def test_no_retry_when_none(self):
        """config=None means error propagates as-is."""
        fn = Mock(side_effect=ollama.RequestError("connection refused"))

        with pytest.raises(ollama.RequestError):
            retry_sync(fn, None)

        assert fn.call_count == 1


# ===================================================================
# 3. retry_async logic
# ===================================================================


class TestRetryAsyncLogic:
    @pytest.mark.asyncio
    @patch("agents7.llm.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_succeeds_after_retry(self, mock_sleep):
        """Async RequestError on first call, success on second."""
        fn = AsyncMock(
            side_effect=[
                ollama.RequestError("connection refused"),
                {"response": "ok"},
            ]
        )
        cfg = RetryConfig(max_retries=3, base_delay=0.01)

        result = await retry_async(fn, cfg, x=1)

        assert result == {"response": "ok"}
        assert fn.call_count == 2
        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    @patch("agents7.llm.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_exhausted(self, mock_sleep):
        """All async retries fail, raises LLMException."""
        fn = AsyncMock(side_effect=ollama.RequestError("connection refused"))
        cfg = RetryConfig(max_retries=2, base_delay=0.01)

        with pytest.raises(LLMException, match="retries exhausted"):
            await retry_async(fn, cfg)

        assert fn.call_count == 3
        assert mock_sleep.call_count == 2


# ===================================================================
# 4. YAML config parsing
# ===================================================================


class TestRetryYAMLConfig:
    def test_yaml_with_retry(self, tmp_path):
        """Parses retry section into RetryConfig."""
        yaml_content = (
            "name: RetryLabeler\n"
            "version: 1\n"
            "engine:\n"
            "    provider: ollama\n"
            "    model: qwen3\n"
            "    retry:\n"
            "        max_retries: 5\n"
            "        base_delay: 2.0\n"
            "system_prompt: Test prompt.\n"
            "output_model: agents7.models.demo.PersonInfo\n"
        )
        yaml_file = tmp_path / "config.yml"
        yaml_file.write_text(yaml_content)

        agent = Labeler.from_yaml(str(yaml_file))
        cfg = agent.engine.retry_config

        assert cfg is not None
        assert cfg.max_retries == 5
        assert cfg.base_delay == 2.0
        # Defaults for unspecified fields
        assert cfg.max_delay == 30.0
        assert cfg.backoff_factor == 2.0

    def test_yaml_without_retry(self, tmp_path):
        """No retry section means engine.retry_config is None."""
        yaml_content = (
            "name: NoRetryLabeler\n"
            "version: 1\n"
            "engine:\n"
            "    provider: ollama\n"
            "    model: qwen3\n"
            "system_prompt: Test prompt.\n"
            "output_model: agents7.models.demo.PersonInfo\n"
        )
        yaml_file = tmp_path / "config.yml"
        yaml_file.write_text(yaml_content)

        agent = Labeler.from_yaml(str(yaml_file))

        assert agent.engine.retry_config is None
