"""client helpers — env resolution, model string building, and empty-content handling."""
from types import SimpleNamespace

import pytest
from lib.core.client import _model_str, _resolve_env, call


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


def _fake_response(content, finish_reason="stop", reasoning_content=None):
    message = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def test_call_raises_on_none_content(monkeypatch):
    monkeypatch.setattr(
        "litellm.completion",
        lambda **kwargs: _fake_response(None, finish_reason="content_filter"),
    )
    with pytest.raises(RuntimeError, match="returned no content"):
        call({"source": "openai", "model": "gpt-4o", "apikey": "sk-x"}, [])


def test_call_raises_on_empty_string_content(monkeypatch):
    # content="" (finish_reason="length") must be treated as a dead turn too, not None-only.
    monkeypatch.setattr(
        "litellm.completion",
        lambda **kwargs: _fake_response("", finish_reason="length"),
    )
    with pytest.raises(RuntimeError, match="returned no content"):
        call({"source": "openai", "model": "gpt-4o", "apikey": "sk-x"}, [])


def test_call_returns_nonempty_content(monkeypatch):
    monkeypatch.setattr(
        "litellm.completion",
        lambda **kwargs: _fake_response("hello"),
    )
    content, _, reasoning = call({"source": "openai", "model": "gpt-4o", "apikey": "sk-x"}, [])
    assert content == "hello"
    assert reasoning is None


def test_call_returns_reasoning_content_when_present(monkeypatch):
    monkeypatch.setattr(
        "litellm.completion",
        lambda **kwargs: _fake_response("hello", reasoning_content="thinking it through..."),
    )
    content, _, reasoning = call({"source": "openai", "model": "gpt-4o", "apikey": "sk-x"}, [])
    assert content == "hello"
    assert reasoning == "thinking it through..."


def test_call_default_max_tokens_is_12000(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response("hello")

    monkeypatch.setattr("litellm.completion", fake_completion)
    call({"source": "openai", "model": "gpt-4o", "apikey": "sk-x"}, [])
    assert captured["max_tokens"] == 12000
