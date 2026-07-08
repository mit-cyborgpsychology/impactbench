"""Single place for config defaults. Call sites read cfg through these helpers
so the effective schema and its fallbacks are visible in one file."""
from __future__ import annotations

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


def num_samples(cfg: dict) -> int:
    return cfg.get("generation", {}).get("num_samples", 1)


def perfunctory(cfg: dict) -> bool:
    return bool(cfg.get("perfunctory", False))


def landmarks(cfg: dict) -> bool:
    return bool(cfg.get("landmarks", True))
