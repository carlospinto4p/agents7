from dataclasses import dataclass

from cpinto.settings import load_setting


@dataclass
class BatchSettings:
    max_concurrent: int | None = load_setting("BATCH_MAX_CONCURRENT", 5)


@dataclass
class OllamaSettings:
    model_name: str = load_setting("OLLAMA_MODEL_NAME", "qwen3")
