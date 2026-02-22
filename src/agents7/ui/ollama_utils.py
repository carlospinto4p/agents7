"""Ollama connectivity helpers for the Agent Builder UI."""

from __future__ import annotations


def check_connection() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        import httpx
        import ollama

        ollama.list()
        return True
    except ImportError:
        return False
    except (httpx.HTTPError, ollama.ResponseError):
        return False


def list_available_models() -> list[str]:
    """Return sorted model names from the local Ollama server.

    Returns an empty list when Ollama is unreachable.
    """
    try:
        import httpx
        import ollama

        response = ollama.list()
        names = [m.model for m in response.models]
        return sorted(names)
    except ImportError:
        return []
    except (httpx.HTTPError, ollama.ResponseError):
        return []
