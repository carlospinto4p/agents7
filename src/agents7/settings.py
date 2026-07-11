import os
from dataclasses import dataclass, field


def _load_setting(name: str, default):
    val = os.getenv(name)
    if val is None:
        return default
    return type(default)(val)


@dataclass(frozen=True)
class BatchSettings:
    """
    Settings for async batch processing.

    Attributes:
        max_concurrent (int | None): Maximum number of
            concurrent LLM requests. Overridable via the
            `BATCH_MAX_CONCURRENT` env var. Defaults to 5.
            `None` means unlimited concurrency.
    """

    max_concurrent: int | None = _load_setting("BATCH_MAX_CONCURRENT", 5)


@dataclass(frozen=True)
class OllamaSettings:
    """
    Settings for the Ollama LLM provider.

    Attributes:
        model_name (str): The Ollama model name. Overridable
            via the `OLLAMA_MODEL_NAME` env var. Defaults to
            `"qwen3"`.
    """

    model_name: str = _load_setting("OLLAMA_MODEL_NAME", "qwen3")


@dataclass(frozen=True)
class OpenAICompatSettings:
    """
    Settings for the OpenAI-compatible LLM provider.

    Lets an external OpenAI-compatible API (DeepSeek, OpenAI,
    vLLM, LM Studio, ...) be configured entirely from
    environment variables, so no keys or URLs are hardcoded.
    Each field is read from the environment at instantiation
    time (not at import), so setting the variable before
    building the provider is enough.

    Attributes:
        model_name (str): Model identifier (e.g.
            `"deepseek-chat"`). Read from the
            `OPENAI_COMPAT_MODEL_NAME` env var. Defaults to
            `""` (must be supplied explicitly or via the env
            var).
        base_url (str | None): API base URL (e.g.
            `"https://api.deepseek.com"`). Read from the
            `OPENAI_COMPAT_BASE_URL` env var. Defaults to
            `None` (the OpenAI default endpoint).
        api_key (str | None): API key. Read from the
            `OPENAI_COMPAT_API_KEY` env var. Defaults to
            `None`, in which case the `openai` SDK falls back
            to its own `OPENAI_API_KEY` / `OPENAI_BASE_URL`
            variables.
    """

    model_name: str = field(
        default_factory=lambda: os.getenv("OPENAI_COMPAT_MODEL_NAME", "")
    )
    base_url: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_COMPAT_BASE_URL")
    )
    api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_COMPAT_API_KEY")
    )
