"""client helpers — env resolution, model string building, and empty-content handling."""
from types import SimpleNamespace

import litellm
import pytest
from lib.core.client import ContentFilterError, EmptyStopError, SafetyRejectionError, _model_str, _resolve_env, call


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


def _fake_response(content, finish_reason="stop", reasoning_content=None, usage=None):
    message = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


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


def test_call_raises_content_filter_error_specifically_on_content_filter(monkeypatch):
    # A provider-side block (finish_reason=content_filter) must raise the
    # distinguishable ContentFilterError, not the plain RuntimeError other
    # no-content cases raise -- simulate() branches on this to decide
    # whether a same-turn model fallback applies.
    monkeypatch.setattr(
        "litellm.completion",
        lambda **kwargs: _fake_response(None, finish_reason="content_filter"),
    )
    with pytest.raises(ContentFilterError):
        call({"source": "openai", "model": "gpt-4o", "apikey": "sk-x"}, [])


def test_call_raises_plain_runtime_error_on_length_not_content_filter(monkeypatch):
    # finish_reason=length (token-budget exhaustion) is a different failure
    # mode than a content block and must NOT be treated as one.
    monkeypatch.setattr(
        "litellm.completion",
        lambda **kwargs: _fake_response("", finish_reason="length"),
    )
    try:
        call({"source": "openai", "model": "gpt-4o", "apikey": "sk-x"}, [])
        assert False, "expected an exception"
    except ContentFilterError:
        assert False, "length should not raise ContentFilterError"
    except RuntimeError:
        pass


def test_call_raises_empty_stop_error_on_stop_with_no_content(monkeypatch):
    # finish_reason=stop reports a normal, successful completion -- but with
    # empty content there's nothing usable, and it's not a length or
    # content_filter case, so it gets its own distinguishable error rather
    # than falling into the generic no-content RuntimeError.
    monkeypatch.setattr(
        "litellm.completion",
        lambda **kwargs: _fake_response("", finish_reason="stop"),
    )
    with pytest.raises(EmptyStopError):
        call({"source": "openai", "model": "gpt-4o", "apikey": "sk-x"}, [])


def test_call_raises_safety_rejection_error_on_provider_403(monkeypatch):
    # xAI (and possibly others) can reject a request outright at the
    # infrastructure level -- a 403 permission-denied litellm.APIError, a
    # different exception class entirely from BadRequestError, before any
    # response/finish_reason exists.
    def raise_safety_error(**kwargs):
        raise litellm.APIError(
            status_code=403,
            message="{'code': 'permission-denied', 'error': 'Content violates usage guidelines. "
                    "Failed check: SAFETY_CHECK_TYPE_CSAM'}",
            llm_provider="openai",
            model="grok-4.6",
        )

    monkeypatch.setattr("litellm.completion", raise_safety_error)
    with pytest.raises(SafetyRejectionError):
        call({"source": "openai", "model": "grok-4.6", "apikey": "sk-x"}, [])


def test_call_does_not_convert_unrelated_api_errors(monkeypatch):
    # A litellm.APIError for an unrelated reason (server error, etc.) must
    # stay a plain APIError -- only the safety-rejection shape converts.
    def raise_other_error(**kwargs):
        raise litellm.APIError(
            status_code=500,
            message="internal server error",
            llm_provider="openai",
            model="grok-4.6",
        )

    monkeypatch.setattr("litellm.completion", raise_other_error)
    try:
        call({"source": "openai", "model": "grok-4.6", "apikey": "sk-x"}, [])
        assert False, "expected an exception"
    except SafetyRejectionError:
        assert False, "unrelated APIError should not become SafetyRejectionError"
    except litellm.APIError:
        pass


def test_call_raises_content_filter_error_on_openai_moderation_bad_request(monkeypatch):
    # OpenAI's moderation layer can reject the request outright -- a
    # BadRequestError from litellm.completion() itself, with no response
    # object and no finish_reason at all, unlike Anthropic's
    # finish_reason=content_filter shape. Must still normalize to the same
    # ContentFilterError so simulate()'s fallback/recording logic applies.
    def raise_moderation_error(**kwargs):
        raise litellm.BadRequestError(
            message="OpenAIException - This content was flagged for possible cybersecurity risk. "
                    "If this seems wrong, try rephrasing your request.",
            model="gpt-5.6-terra",
            llm_provider="openai",
        )

    monkeypatch.setattr("litellm.completion", raise_moderation_error)
    with pytest.raises(ContentFilterError):
        call({"source": "openai", "model": "gpt-5.6-terra", "apikey": "sk-x"}, [])


def test_call_raises_content_filter_error_on_other_moderation_categories(monkeypatch):
    # "flagged for possible <category> risk" spans multiple categories
    # (cybersecurity, biological, ... more may appear without notice) --
    # matched on the stable "flagged for possible" prefix, not each
    # category's full text, so a new category is caught without a code change.
    def raise_moderation_error(**kwargs):
        raise litellm.BadRequestError(
            message="OpenAIException - This content was flagged for possible biological risk. "
                    "If this seems wrong, try rephrasing your request.",
            model="gpt-5.6-luna",
            llm_provider="openai",
        )

    monkeypatch.setattr("litellm.completion", raise_moderation_error)
    with pytest.raises(ContentFilterError):
        call({"source": "openai", "model": "gpt-5.6-luna", "apikey": "sk-x"}, [])


def test_call_does_not_convert_unrelated_bad_request_errors(monkeypatch):
    # A BadRequestError for an unrelated reason (bad params, etc.) must stay
    # a plain BadRequestError -- only moderation-shaped messages get
    # normalized to ContentFilterError.
    def raise_other_error(**kwargs):
        raise litellm.BadRequestError(
            message="invalid max_tokens value",
            model="gpt-5.6-terra",
            llm_provider="openai",
        )

    monkeypatch.setattr("litellm.completion", raise_other_error)
    with pytest.raises(litellm.BadRequestError):
        call({"source": "openai", "model": "gpt-5.6-terra", "apikey": "sk-x"}, [])
    try:
        call({"source": "openai", "model": "gpt-5.6-terra", "apikey": "sk-x"}, [])
    except ContentFilterError:
        assert False, "unrelated BadRequestError should not become ContentFilterError"
    except litellm.BadRequestError:
        pass


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


def test_call_default_max_tokens_is_32000(monkeypatch):
    # Raised 12000 -> 16000 -> 32000 -- reasoning models (e.g. qwen3.8-27b)
    # can burn most or all of the budget on hidden reasoning_content alone,
    # leaving nothing for the actual visible response (EmptyStopError /
    # length-exhaustion RuntimeError), and 16000 still wasn't always enough
    # on some scenarios.
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response("hello")

    monkeypatch.setattr("litellm.completion", fake_completion)
    call({"source": "openai", "model": "gpt-4o", "apikey": "sk-x"}, [])
    assert captured["max_tokens"] == 32000


def test_call_uses_manual_price_when_litellm_has_none(monkeypatch):
    # litellm has no pricing entry for this deepinfra model, so
    # completion_cost() raises/returns nothing usable -- cost must fall back
    # to the manual $/token table instead of silently reporting $0.
    usage = SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    monkeypatch.setattr(
        "litellm.completion",
        lambda **kwargs: _fake_response("hello", usage=usage),
    )
    monkeypatch.setattr(
        "litellm.completion_cost",
        lambda response: (_ for _ in ()).throw(Exception("no pricing data")),
    )
    _, usage_obj, _ = call(
        {"source": "deepinfra", "model": "deepseek-ai/DeepSeek-V4-Flash-0731", "apikey": "sk-x"}, []
    )
    # 1M input @ $0.08/M + 1M output @ $0.18/M = $0.26
    assert usage_obj.cost == pytest.approx(0.26)


def test_call_uses_manual_price_for_grok_via_openai_source(monkeypatch):
    # grok is routed as source: openai with a custom base_url (xAI's
    # OpenAI-compatible endpoint), so its _model_str is "openai/grok-4.6" --
    # litellm has no pricing entry for it either, same manual-fallback gap.
    usage = SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    monkeypatch.setattr(
        "litellm.completion",
        lambda **kwargs: _fake_response("hello", usage=usage),
    )
    monkeypatch.setattr(
        "litellm.completion_cost",
        lambda response: (_ for _ in ()).throw(Exception("no pricing data")),
    )
    _, usage_obj, _ = call(
        {"source": "openai", "model": "grok-4.6", "apikey": "sk-x", "base_url": "https://api.x.ai/v1"}, []
    )
    # 1M input @ $2.00/M + 1M output @ $6.00/M = $8.00
    assert usage_obj.cost == pytest.approx(8.00)


def test_call_leaves_zero_cost_for_unknown_model(monkeypatch):
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=1000)
    monkeypatch.setattr(
        "litellm.completion",
        lambda **kwargs: _fake_response("hello", usage=usage),
    )
    monkeypatch.setattr(
        "litellm.completion_cost",
        lambda response: (_ for _ in ()).throw(Exception("no pricing data")),
    )
    _, usage_obj, _ = call({"source": "deepinfra", "model": "some/unlisted-model", "apikey": "sk-x"}, [])
    assert usage_obj.cost == 0.0
