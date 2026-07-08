"""client helpers — env resolution and model string building."""
import pytest
from lib.core.client import _model_str, _resolve_env


def test_resolve_env_substitutes(monkeypatch):
    monkeypatch.setenv("MY_TEST_KEY", "sk-123")
    out = _resolve_env({"apikey": "${MY_TEST_KEY}", "model": "m"})
    assert out["apikey"] == "sk-123"


def test_resolve_env_missing_var_raises(monkeypatch):
    monkeypatch.delenv("MY_MISSING_KEY", raising=False)
    with pytest.raises(KeyError, match="MY_MISSING_KEY"):
        _resolve_env({"apikey": "${MY_MISSING_KEY}"})


def test_resolve_env_leaves_literal_values():
    assert _resolve_env({"apikey": "sk-literal"})["apikey"] == "sk-literal"


def test_model_str_prepends_source():
    assert _model_str({"source": "anthropic", "model": "claude-sonnet-4-6"}) == "anthropic/claude-sonnet-4-6"


def test_model_str_openai_and_already_prefixed():
    assert _model_str({"source": "openai", "model": "gpt-4o"}) == "gpt-4o"
    assert _model_str({"source": "deepinfra", "model": "deepinfra/x"}) == "deepinfra/x"
