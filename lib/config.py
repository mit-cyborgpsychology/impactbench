from __future__ import annotations
from pathlib import Path

CONCURRENCY_DEFAULTS = {
    "gen_scenarios": 5,
    "simulate": 10,
    "evaluate": 20,
}


def run_cfg(cfg: dict) -> dict:
    return cfg.get("run", {})


def force(cfg: dict) -> bool:
    return bool(run_cfg(cfg).get("force", False))


def dry_run(cfg: dict) -> bool:
    return bool(run_cfg(cfg).get("dry_run", False))


def verbose(cfg: dict) -> bool:
    return bool(run_cfg(cfg).get("verbose", False))


def concurrency(cfg: dict, phase: str) -> int:
    conc = run_cfg(cfg).get("concurrency", {})
    if phase in conc:
        return conc[phase]
    if phase == "gen_scenarios_expand":
        return concurrency(cfg, "gen_scenarios") * 2
    return CONCURRENCY_DEFAULTS[phase]


def benchmarks_dir(cfg: dict) -> Path:
    # run.benchmarks_dir lets a config point the pipeline at a private benchmarks
    # tree (e.g. a pilot under v2-user-model/) without touching lib/paths.py or
    # the shared benchmarks/ folder. Absent key -> identical to today's BENCHMARKS.
    from lib.paths import ROOT, BENCHMARKS
    rel = run_cfg(cfg).get("benchmarks_dir")
    return (ROOT / rel) if rel else BENCHMARKS


def cache_dir(cfg: dict) -> Path:
    # run.cache_dir mirrors run.benchmarks_dir: lets a config point row_cache at
    # a private cache tree so two configs sharing a benchmark *name* (e.g. each
    # v2-user-model user-model has its own "humanebench-subset") don't collide
    # on cache keys and silently serve each other's cached simulate/eval results.
    # Absent key -> identical to today's CACHE.
    from lib.paths import ROOT, CACHE
    rel = run_cfg(cfg).get("cache_dir")
    return (ROOT / rel) if rel else CACHE


def num_samples(cfg: dict) -> int:
    return cfg.get("generation", {}).get("num_samples", 1)


def perfunctory(cfg: dict) -> bool:
    return bool(cfg.get("perfunctory", False))


def landmarks(cfg: dict) -> bool:
    return bool(cfg.get("landmarks", True))


def demographic_filter(cfg: dict) -> dict:
    # run.demographic_filter restricts simulate to scenarios whose demographic
    # dict matches every key given here (e.g. {"age": "Child or teenager (6-17)"}
    # keeps only that age group, letting one scenarios.json — generated once for
    # every demographic — be re-run against a subset without regenerating).
    # Absent/empty -> no filtering, every scenario runs (today's behavior).
    return run_cfg(cfg).get("demographic_filter") or {}


def scenario_matches_filter(scenario: dict, filt: dict) -> bool:
    if not filt:
        return True
    demo = scenario.get("demographic", {})
    return all(demo.get(k) == v for k, v in filt.items())


def user_model_fallbacks(cfg: dict) -> list[dict]:
    return cfg.get("user_model_fallbacks", [])
