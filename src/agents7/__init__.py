"""Agents7 - A collection of agents for various tasks."""

from importlib import metadata

from agents7.model_selection import (
    BenchmarkResult as BenchmarkResult,
    ModelScore as ModelScore,
    TestCase as TestCase,
    find_best_model as find_best_model,
)


__version__: str = metadata.version("agents7")
