"""Data models for prompt version comparison."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PromptVersion:
    """A snapshot of prompt configuration for A/B comparison.

    Fields:
        name: User-facing label for this version.
        system_prompt: The system prompt text.
        examples: List of example dicts (input/output pairs).
        model_name: Ollama model name (e.g. ``"qwen3"``).
    """

    name: str
    system_prompt: str = ""
    examples: list[dict] = field(default_factory=list)
    model_name: str = "qwen3"
