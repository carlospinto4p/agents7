from dataclasses import dataclass

from cpinto.settings import load_env

DEFAULT_MAX_CONCURRENT = 5


@dataclass
class OllamaSettings:
    model_name: str = load_env("OLLAMA_MODEL_NAME", "qwen3")
