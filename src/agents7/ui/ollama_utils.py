"""Ollama connectivity helpers for the Agent Builder UI."""

from __future__ import annotations

from datetime import timedelta

import streamlit as st

_cache_resource = st.cache_resource


def render_model_selector(
    sd: dict,
    key_prefix: str = "",
) -> None:
    """Render Ollama model selector widget.

    Checks connection, lists models if available,
    falls back to manual text_input.
    Writes chosen model into ``sd["model_name"]``.

    :param sd: Session state dict; reads/writes
        ``sd["model_name"]``.
    :param key_prefix: Streamlit widget key prefix to
        avoid duplicate keys across pages.
    """
    connected = check_connection()
    if connected:
        models = list_available_models()
        if models:
            current = sd.get("model_name", "qwen3")
            idx = models.index(current) if current in models else 0
            sd["model_name"] = st.selectbox(
                "Ollama model",
                models,
                index=idx,
                key=f"{key_prefix}model_select",
            )
        else:
            st.warning("Ollama is running but no models found.")
            sd["model_name"] = st.text_input(
                "Model name",
                value=sd.get("model_name", "qwen3"),
                key=f"{key_prefix}model_input",
            )
    else:
        st.warning(
            "Ollama not connected. Start it with "
            "`ollama serve` for auto-discovery."
        )
        sd["model_name"] = st.text_input(
            "Model name (manual)",
            value=sd.get("model_name", "qwen3"),
            key=f"{key_prefix}model_manual",
        )


def check_connection() -> bool:
    """Return True if the Ollama server is reachable.

    Results are cached for 30 seconds via
    ``_cached_check_connection``.
    """
    return _cached_check_connection()


@_cache_resource(ttl=timedelta(seconds=30))
def _cached_check_connection() -> bool:
    """Check Ollama connectivity (cached)."""
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

    Results are cached for 30 seconds via
    ``_cached_list_available_models``.
    Returns an empty list when Ollama is unreachable.
    """
    return _cached_list_available_models()


@_cache_resource(ttl=timedelta(seconds=30))
def _cached_list_available_models() -> list[str]:
    """Fetch Ollama model names (cached)."""
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
