"""
bench-py runner.

    python main.py <benchmark> gen_metrics           # phase 1: generate metrics from description
    python main.py <benchmark> gen_scenarios         # phase 2: generate scenarios from metrics
    python main.py <benchmark> simulate <model>      # phase 3: run conversations
    python main.py <benchmark> evaluate <model>      # phase 4: score conversations
    python main.py <benchmark> aggregate             # aggregate all model runs
    python main.py <benchmark> all [model]           # all phases for one or all models
    python main.py all                               # all benchmarks × all targets

All behavior (concurrency, force, dry_run) is configured in config.yaml under run:
"""
from __future__ import annotations
import argparse
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from lib import config
from lib.paths import ROOT, BENCHMARKS
from lib.pipeline import gen_metrics, gen_scenarios, simulate, evaluate, aggregate


def load_config(path: str = "config.yaml") -> dict:
    return yaml.safe_load((ROOT / path).read_text()) or {}


def target_ids(cfg: dict) -> list[str]:
    return [t["id"] for t in cfg.get("targets", [])]


def list_benchmarks() -> list[str]:
    return sorted(
        p.name for p in BENCHMARKS.iterdir()
        if p.is_dir() and (p / "benchmark.yaml").exists()
    )


def phase_done(phase: str, benchmark: str, model: str = "") -> bool:
    bench_dir = BENCHMARKS / benchmark
    goal = bench_dir / "benchmark.yaml"

    if phase == "gen_metrics":
        return goal.exists() and bool(yaml.safe_load(goal.read_text()).get("metrics"))

    paths = {
        "gen_scenarios": bench_dir / "scenarios.json",
        "simulate":      bench_dir / "runs" / model / "conversations.json",
        "evaluate":      bench_dir / "runs" / model / "scores.json",
    }
    return phase in paths and paths[phase].exists()


def run_phase(phase: str, benchmark: str, cfg: dict, model: str = "") -> None:
    label = f"{phase} {benchmark}" + (f"/{model}" if model else "")
    if config.dry_run(cfg):
        return print(f"  [dry-run] {label}")
    # aggregate always re-runs: it is cheap and must pick up new model runs.
    if phase != "aggregate" and not config.force(cfg) and phase_done(phase, benchmark, model):
        return print(f"  [skip] {label} — set run.force: true in config.yaml to re-run")

    runners = {
        "gen_metrics":   lambda: gen_metrics.run(benchmark, cfg),
        "gen_scenarios": lambda: gen_scenarios.run(benchmark, cfg),
        "simulate":      lambda: simulate.run(benchmark, model, cfg),
        "evaluate":      lambda: evaluate.run(benchmark, model, cfg),
        "aggregate":     lambda: aggregate.run(benchmark),
    }
    runners[phase]()


def run_all(benchmark: str, models: list[str], cfg: dict) -> None:
    print(f"\n[{benchmark}] 1/5 gen_metrics")
    run_phase("gen_metrics", benchmark, cfg)
    print(f"\n[{benchmark}] 2/5 gen_scenarios")
    run_phase("gen_scenarios", benchmark, cfg)
    for model in models:
        print(f"\n[{benchmark}/{model}] 3/5 simulate")
        run_phase("simulate", benchmark, cfg, model)
        print(f"\n[{benchmark}/{model}] 4/5 evaluate")
        run_phase("evaluate", benchmark, cfg, model)
    print(f"\n[{benchmark}] 5/5 aggregate")
    run_phase("aggregate", benchmark, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="bench-py runner",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("benchmark", help="benchmark name or 'all'")
    parser.add_argument("--config", default="config.yaml", help="config file to use")
    parser.add_argument("phase", nargs="?", default="all",
                        choices=["gen_metrics", "gen_scenarios", "simulate", "evaluate", "aggregate", "all"])
    parser.add_argument("model", nargs="?")
    args = parser.parse_args()

    cfg = load_config(args.config)
    targets = target_ids(cfg)

    if args.benchmark == "all":
        if args.phase != "all" or args.model:
            parser.error("'all' benchmarks runs every phase and target — "
                         "phase/model arguments would be ignored")
        for b in list_benchmarks():
            run_all(b, targets, cfg)

    elif args.phase == "all":
        models = [args.model] if args.model else targets
        run_all(args.benchmark, models, cfg)

    elif args.phase in ("simulate", "evaluate") and not args.model:
        parser.error(f"model required for '{args.phase}'")

    else:
        run_phase(args.phase, args.benchmark, cfg, args.model or "")

    print("\ndone.")


if __name__ == "__main__":
    main()
