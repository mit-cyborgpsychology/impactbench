from __future__ import annotations
import json
import logging
import os
import re

import litellm

from lib.core.usage import Usage

litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class ContentFilterError(RuntimeError):
    """Raised when a provider blocks a response (finish_reason=content_filter)
    rather than the model simply running out of budget/erroring. Distinct from
    the plain no-content RuntimeError so callers can react differently —
    e.g. simulate() falls back to a different user-model for a censored user
    turn instead of endlessly retrying the same model against the same wall.
    """


class EmptyStopError(RuntimeError):
    """Raised when a provider reports a normal, successful completion
    (finish_reason=stop) but the content is empty — a silent filter that
    gives no explicit content_filter signal or error, unlike the other two
    known block shapes. Distinct from ContentFilterError since there's no
    filter category to report, and distinct from the plain no-content
    RuntimeError (finish_reason=length, etc.) since retrying the exact same
    turn tends to reproduce it, same as a real content block would.
    """


class SafetyRejectionError(RuntimeError):
    """Raised when a provider refuses to process the request at all at the
    infrastructure level — an HTTP 403 permission-denied, not a 400
    BadRequestError like the moderation-block cases and not a normal
    completion response with finish_reason=content_filter. Seen from xAI as
    {"code": "permission-denied", "error": "... Failed check: SAFETY_CHECK_TYPE_CSAM"}.
    Distinct type because this is the strictest of the known block shapes —
    the request is rejected outright, before any model output exists.
    """

# OpenAI's moderation layer can reject a request outright — a BadRequestError
# raised by litellm.completion() itself, before any response/finish_reason
# exists — rather than returning finish_reason=content_filter like Anthropic
# does. Different shape, same underlying event, so it's matched here by
# message text and normalized to the same ContentFilterError. "flagged for
# possible <category> risk" is the stable prefix across categories (seen so
# far: cybersecurity, biological) rather than listing each category as its
# own marker, since new categories appear without notice.
_OPENAI_MODERATION_MARKERS = (
    "flagged for possible",
    "content_policy_violation",
)

# xAI (and possibly other OpenAI-compatible providers) can reject a request
# outright at the infrastructure level for its strictest safety categories —
# a 403 permission-denied litellm.APIError, matched on the "code" field
# rather than message text since the message itself names the specific
# check (SAFETY_CHECK_TYPE_CSAM today; other SAFETY_CHECK_TYPE_* values are
# plausible and should match the same way).
_SAFETY_REJECTION_MARKERS = (
    "permission-denied",
    "SAFETY_CHECK_TYPE",
)

# litellm has no pricing entry for these (too new / not yet added upstream),
# so completion_cost() silently returns 0 — manual fallback, $/token, from
# each provider's own pricing pages (deepinfra.com checked 2026-08-20,
# docs.x.ai checked 2026-08-21). grok-4.6 rate is the <200k-token-prompt
# tier ($2/$6 per M in/out); this benchmark's prompts never approach the
# 200k threshold that would trigger xAI's higher tier.
_MANUAL_PRICE_PER_TOKEN = {
    "deepinfra/deepseek-ai/DeepSeek-V4-Flash-0731": (0.08e-6, 0.18e-6),
    "deepinfra/Qwen/Qwen3.8-27B":                   (0.40e-6, 3.00e-6),
    "deepinfra/google/gemma-4-31B-it":              (0.13e-6, 0.38e-6),
    "deepinfra/anthropic/claude-sonnet-5":          (2.00e-6, 10.00e-6),
    "deepinfra/anthropic/claude-opus-5":            (5.00e-6, 25.00e-6),
    "openai/grok-4.6":                              (2.00e-6, 6.00e-6),
}


def _usage(response, model: str) -> Usage:
    try:
        cost = litellm.completion_cost(response)
    except Exception:
        cost = 0.0
    u = getattr(response, "usage", None)
    input_tokens = getattr(u, "prompt_tokens", 0) or 0
    output_tokens = getattr(u, "completion_tokens", 0) or 0
    if not cost and model in _MANUAL_PRICE_PER_TOKEN:
        in_rate, out_rate = _MANUAL_PRICE_PER_TOKEN[model]
        cost = input_tokens * in_rate + output_tokens * out_rate
    return Usage(
        cost=cost or 0.0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _model_str(cfg: dict) -> str:
    source, model = cfg.get("source", "openai"), cfg["model"]
    if model.startswith(f"{source}/"):
        return model
    # source: openai normally needs no prefix, but once base_url points at an
    # OpenAI-compatible third party (xAI, PublicAI, ...) litellm can't infer a
    # provider from the bare model name and rejects the call outright — the
    # "openai/" prefix forces OpenAI-compatible routing regardless of api_base.
    if source == "openai" and not cfg.get("base_url"):
        return model
    return f"{source}/{model}"


def _resolve_env(cfg: dict) -> dict:
    # KeyError on a missing var (not a silent "") so auth failures point at the
    # actual cause — and KeyError is non-retryable, so it fails fast.
    def sub(m: re.Match) -> str:
        var = m.group(1)
        if var not in os.environ:
            raise KeyError(f"config references ${{{var}}} but it is not set in the environment")
        return os.environ[var]

    out = dict(cfg)
    for field in ("apikey", "base_url"):
        val = out.get(field, "")
        if val and isinstance(val, str):
            out[field] = re.sub(r"\$\{(\w+)\}", sub, val)
    return out


def call(cfg: dict, messages: list[dict], max_tokens: int = 32000, json_mode: bool = False) -> tuple[str, Usage, str | None]:
    cfg = _resolve_env(cfg)
    kwargs: dict = {
        "model":      _model_str(cfg),
        "messages":   messages,
        "api_key":    cfg.get("apikey"),
        "max_tokens": max_tokens,
    }
    if cfg.get("base_url"):
        kwargs["api_base"] = cfg["base_url"]
    if cfg.get("extra_headers"):
        kwargs["extra_headers"] = cfg["extra_headers"]
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        response = litellm.completion(**kwargs)
    except litellm.BadRequestError as exc:
        if any(marker in str(exc) for marker in _OPENAI_MODERATION_MARKERS):
            raise ContentFilterError(f"{kwargs['model']} rejected by moderation: {exc}") from exc
        raise
    except litellm.APIError as exc:
        if any(marker in str(exc) for marker in _SAFETY_REJECTION_MARKERS):
            raise SafetyRejectionError(f"{kwargs['model']} rejected outright by provider safety check: {exc}") from exc
        raise
    message = response.choices[0].message
    content = message.content
    reasoning = getattr(message, "reasoning_content", None) or None
    if not content:
        # None or "" (reasoning models can burn max_tokens on hidden reasoning
        # and leave content empty) — both unusable, so raise so retry() takes it.
        finish_reason = getattr(response.choices[0], "finish_reason", "?")
        message_text = (
            f"{kwargs['model']} returned no content "
            f"(finish_reason={finish_reason}, "
            f"reasoning={len(reasoning) if reasoning else 0} chars)"
        )
        if finish_reason == "content_filter":
            raise ContentFilterError(message_text)
        if finish_reason == "stop":
            # A "successful" completion with nothing in it -- no filter
            # category reported, no error raised, just silence. Seen so far
            # only on the target side, reproducibly on the same turn across
            # retries with differently-phrased user input.
            raise EmptyStopError(message_text)
        raise RuntimeError(message_text)
    return content, _usage(response, kwargs["model"]), reasoning


def call_json(cfg: dict, messages: list[dict], max_tokens: int = 32000) -> tuple[dict, Usage]:
    raw, usage, _reasoning = call(cfg, messages, max_tokens=max_tokens, json_mode=True)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", raw.strip(), flags=re.DOTALL)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned.strip()).strip()
    return json.loads(cleaned), usage
