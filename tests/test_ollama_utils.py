"""Unit tests for agents7.ui.ollama_utils.

Covers check_connection() and list_available_models() error
handling paths (HTTPError, ResponseError, success).
"""

import importlib
from unittest.mock import MagicMock, patch

import httpx
import ollama
import pytest


def _noop_cache(**_kwargs):
    """Passthrough stand-in for st.cache_resource."""

    def decorator(fn):
        return fn

    return decorator


@pytest.fixture()
def utils():
    """ollama_utils reloaded with a no-op cache_resource.

    Streamlit's cache_resource wraps the private functions at
    import time, making them impossible to test with fresh
    arguments.  Replacing cache_resource with a passthrough
    and reloading the module gives us the raw functions.
    """
    import streamlit

    original = streamlit.cache_resource
    streamlit.cache_resource = _noop_cache
    import agents7.ui.ollama_utils as mod

    importlib.reload(mod)
    yield mod
    streamlit.cache_resource = original
    importlib.reload(mod)


# ── check_connection ─────────────────────────────────────────────


def test_check_connection_returns_true(utils):
    """Returns True when ollama.list() succeeds."""
    with patch.object(ollama, "list"):
        assert utils.check_connection() is True


def test_check_connection_httpx_error(utils):
    """Returns False on httpx.HTTPError (connection refused/timeout)."""
    with patch.object(ollama, "list", side_effect=httpx.HTTPError("timeout")):
        assert utils.check_connection() is False


def test_check_connection_response_error(utils):
    """Returns False on ollama.ResponseError."""
    with patch.object(
        ollama,
        "list",
        side_effect=ollama.ResponseError("server error"),
    ):
        assert utils.check_connection() is False


# ── list_available_models ─────────────────────────────────────────


def test_list_models_returns_sorted(utils):
    """Returns model names sorted alphabetically."""
    response = MagicMock()
    response.models = [
        MagicMock(model="zoo"),
        MagicMock(model="alpha"),
        MagicMock(model="beta"),
    ]
    with patch.object(ollama, "list", return_value=response):
        assert utils.list_available_models() == ["alpha", "beta", "zoo"]


def test_list_models_httpx_error(utils):
    """Returns empty list on httpx.HTTPError."""
    with patch.object(ollama, "list", side_effect=httpx.HTTPError("timeout")):
        assert utils.list_available_models() == []


def test_list_models_response_error(utils):
    """Returns empty list on ollama.ResponseError."""
    with patch.object(
        ollama,
        "list",
        side_effect=ollama.ResponseError("server error"),
    ):
        assert utils.list_available_models() == []
