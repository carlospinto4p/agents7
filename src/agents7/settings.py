from dataclasses import dataclass

from cpinto.settings import load_setting


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

    max_concurrent: int | None = load_setting("BATCH_MAX_CONCURRENT", 5)


@dataclass(frozen=True)
class OllamaSettings:
    """
    Settings for the Ollama LLM provider.

    Attributes:
        model_name (str): The Ollama model name. Overridable
            via the `OLLAMA_MODEL_NAME` env var. Defaults to
            `"qwen3"`.
    """

    model_name: str = load_setting("OLLAMA_MODEL_NAME", "qwen3")
