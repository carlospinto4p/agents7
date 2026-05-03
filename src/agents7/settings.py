import os
from dataclasses import dataclass


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
