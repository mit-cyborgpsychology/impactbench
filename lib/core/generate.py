from __future__ import annotations
import json
import itertools
from pathlib import Path

from lib.core.client import call_json
from lib.core.prompts import load
from lib.core.usage import Usage


def generate_scenarios(metric: dict, goal: dict, run: dict, user_cfg: dict) -> tuple[list[dict], Usage]:
    prompt = [
        {"role": "system", "content": "You are an expert benchmark designer. Output valid JSON only."},
        {"role": "user", "content": load("generate_scenarios.txt",
            n=run["generation"]["scenarios_per_metric"],
            benchmark_name=goal["name"],
            benchmark_description=goal["description"],
            user_context=goal.get("scenario", {}).get("user_context", ""),
            metric_name=metric["name"],
            metric_type=metric["type"],
            metric_definition=metric["definition"],
            metric_examples=json.dumps(metric.get("examples", [])),
        )},
    ]
    result, usage = call_json(user_cfg, prompt)
    scenarios = result.get("scenarios", [])
    for i, sc in enumerate(scenarios, 1):
        # metric_id only: name/type are resolved from benchmark.yaml wherever they
        # are needed. Storing copies here let them go stale when a metric was
        # renamed or the type vocabulary changed.
        sc["id"]        = f"{metric['id']}_s{i:03d}"
        sc["metric_id"] = metric["id"]
    return scenarios, usage


def generate_scenarios_from_seeds(
    metric: dict, goal: dict, run: dict, user_cfg: dict, seed_titles: list[str]
) -> tuple[list[dict], Usage]:
    """Like generate_scenarios, but for metrics whose seed scenario titles were
    already supplied (e.g. by whoever wrote the metric) instead of being
    invented from scratch. Expands each seed title, in order, into one full
    scenario object — count is len(seed_titles), not generation.scenarios_per_metric.
    Added as a separate function rather than folded into generate_scenarios so
    that function's from-scratch behavior is unaffected.
    """
    if not seed_titles:
        return [], Usage()

    seed_titles_str = "\n".join(f"{i}. {t}" for i, t in enumerate(seed_titles, 1))
    prompt = [
        {"role": "system", "content": "You are an expert benchmark designer. Output valid JSON only."},
        {"role": "user", "content": load("generate_scenarios_from_seed.txt",
            benchmark_name=goal["name"],
            benchmark_description=goal["description"],
            user_context=goal.get("scenario", {}).get("user_context", ""),
            metric_name=metric["name"],
            metric_type=metric["type"],
            metric_definition=metric["definition"],
            metric_examples=json.dumps(metric.get("examples", [])),
            seed_titles=seed_titles_str,
        )},
    ]
    result, usage = call_json(user_cfg, prompt)
    scenarios = result.get("scenarios", [])
    for i, sc in enumerate(scenarios, 1):
        # Same id/metric_id convention as generate_scenarios — see its comment.
        sc["id"]        = f"{metric['id']}_s{i:03d}"
        sc["metric_id"] = metric["id"]
        if i - 1 < len(seed_titles):
            sc["seed_title"] = seed_titles[i - 1]
    return scenarios, usage


def expand_demographics(scenarios: list[dict], run: dict, user_cfg: dict) -> tuple[list[dict], Usage]:
    demographics = run.get("demographics", {})
    if not demographics:
        variants = [{**sc, "id": f"{sc['id']}_v01", "demographic": {}} for sc in scenarios]
        return variants, Usage()

    combos = [
        dict(zip(demographics.keys(), combo))
        for combo in itertools.product(*demographics.values())
    ]
    system = {"role": "system", "content": "You are adapting a benchmark scenario for a specific demographic. Output valid JSON only."}
    variants = []
    usage = Usage()

    for sc in scenarios:
        for i, demo in enumerate(combos, 1):
            demo_str = ", ".join(f"{k}: {v}" for k, v in demo.items())
            prompt = [system, {"role": "user", "content": load("expand_demographic.txt",
                demographic=demo_str,
                scenario=json.dumps(sc, indent=2),
            )}]
            result, u = call_json(user_cfg, prompt)
            usage = usage + u
            variants.append({**sc, **result, "id": f"{sc['id']}_v{i:02d}", "demographic": demo})

    return variants, usage
