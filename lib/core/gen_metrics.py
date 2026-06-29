from __future__ import annotations

from lib.core.client import call_json
from lib.core.prompts import load
from lib.core.usage import Usage


def generate_metrics(name: str, description: str, user_cfg: dict) -> tuple[list[dict], Usage]:
    prompt = [
        {"role": "system", "content": "You are an expert benchmark designer. Output valid JSON only."},
        {"role": "user", "content": load("generate_metrics.txt",
            benchmark_name=name,
            description=description,
        )},
    ]
    result, usage = call_json(user_cfg, prompt)
    metrics = result.get("metrics", [])
    for i, m in enumerate(metrics, 1):
        m["id"] = f"m{i:02d}"
    return metrics, usage
