"""
LLM calls via litellm. source is prepended to model ("source/model") except for openai.
Every call returns its Usage — cost is never tracked in shared state.
"""
from __future__ import annotations
import json
import logging
import os
import re

import litellm

from lib.core.usage import Usage

litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


def _usage(response) -> Usage:
    try:
        cost = litellm.completion_cost(response)
    except Exception:
        cost = 0.0
    u = getattr(response, "usage", None)
    return Usage(
        cost=cost or 0.0,
        input_tokens=getattr(u, "prompt_tokens", 0) or 0,
        output_tokens=getattr(u, "completion_tokens", 0) or 0,
    )


def _model_str(cfg: dict) -> str:
    source, model = cfg.get("source", "openai"), cfg["model"]
    if source == "openai" or model.startswith(f"{source}/"):
        return model
    return f"{source}/{model}"


def _resolve_env(cfg: dict) -> dict:
    out = dict(cfg)
    for field in ("apikey", "base_url"):
        val = out.get(field, "")
        if val and isinstance(val, str):
            out[field] = re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), val)
    return out


def call(cfg: dict, messages: list[dict], max_tokens: int = 2048, json_mode: bool = False) -> tuple[str, Usage]:
    cfg = _resolve_env(cfg)
    kwargs: dict = {
        "model":      _model_str(cfg),
        "messages":   messages,
        "api_key":    cfg.get("apikey"),
        "max_tokens": max_tokens,
    }
    if cfg.get("base_url"):
        kwargs["api_base"] = cfg["base_url"]
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = litellm.completion(**kwargs)
    return response.choices[0].message.content, _usage(response)


def call_json(cfg: dict, messages: list[dict], max_tokens: int = 2048) -> tuple[dict, Usage]:
    raw, usage = call(cfg, messages, max_tokens=max_tokens, json_mode=True)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", raw.strip(), flags=re.DOTALL)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned.strip()).strip()
    return json.loads(cleaned), usage
