"""Agents7 - A collection of agents for various tasks."""

from importlib import metadata

from agents7.agents.memory import MemoryConfig
from agents7.llm import CacheConfig, RetryConfig


__version__: str = metadata.version("agents7")

__all__ = [
    "CacheConfig",
    "MemoryConfig",
    "RetryConfig",
    "__version__",
]
