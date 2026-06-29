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

ROOT = Path(__file__).parent

from lib.pipeline import gen_metrics, gen_scenarios, simulate, evaluate, aggregate


def load_config(path: str = "config.yaml") -> dict:
    return yaml.safe_load((ROOT / path).read_text()) or {}


def target_ids(cfg: dict) -> list[str]:
    return [t["id"] for t in cfg.get("targets", [])]


def list_benchmarks() -> list[str]:
    return sorted(
        p.name for p in (ROOT / "benchmarks").iterdir()
        if p.is_dir() and (p / "benchmark.yaml").exists()
    )


def phase_done(phase: str, benchmark: str, model: str = "") -> bool:
    bench_dir = ROOT / "benchmarks" / benchmark
    goal = bench_dir / "benchmark.yaml"
    has_metrics = goal.exists() and bool(
        yaml.safe_load(goal.read_text()).get("metrics")
    )
    paths = {
        "gen_metrics":   Path("__done__") if has_metrics else Path("__missing__"),
        "gen_scenarios": bench_dir / "scenarios.json",
        "simulate":      bench_dir / "runs" / model / "conversations.json",
        "evaluate":      bench_dir / "runs" / model / "scores.json",
        "aggregate":     bench_dir / "results.json",
    }
    return paths.get(phase, Path("")).exists()


def _skip(label: str) -> None:
    print(f"  [skip] {label} — set run.force: true in config.yaml to re-run")


def _dry(label: str) -> None:
    print(f"  [dry-run] {label}")


def run_gen_metrics(benchmark: str, cfg: dict) -> None:
    run_cfg = cfg.get("run", {})
    label = f"gen_metrics {benchmark}"
    if run_cfg.get("dry_run"):
        return _dry(label)
    if not run_cfg.get("force") and phase_done("gen_metrics", benchmark):
        return _skip(label)
    gen_metrics.run(benchmark, cfg)


def run_gen_scenarios(benchmark: str, cfg: dict) -> None:
    run_cfg = cfg.get("run", {})
    label = f"gen_scenarios {benchmark}"
    if run_cfg.get("dry_run"):
        return _dry(label)
    if not run_cfg.get("force") and phase_done("gen_scenarios", benchmark):
        return _skip(label)
    gen_scenarios.run(benchmark, cfg)


def run_simulate(benchmark: str, model: str, cfg: dict) -> None:
    run_cfg = cfg.get("run", {})
    label = f"simulate {benchmark}/{model}"
    if run_cfg.get("dry_run"):
        return _dry(label)
    if not run_cfg.get("force") and phase_done("simulate", benchmark, model):
        return _skip(label)
    simulate.run(benchmark, model, cfg)


def run_evaluate(benchmark: str, model: str, cfg: dict) -> None:
    run_cfg = cfg.get("run", {})
    label = f"evaluate {benchmark}/{model}"
    if run_cfg.get("dry_run"):
        return _dry(label)
    if not run_cfg.get("force") and phase_done("evaluate", benchmark, model):
        return _skip(label)
    evaluate.run(benchmark, model, cfg)


def run_aggregate(benchmark: str, cfg: dict) -> None:
    if cfg.get("run", {}).get("dry_run"):
        return _dry(f"aggregate {benchmark}")
    aggregate.run(benchmark)


def run_all(benchmark: str, models: list[str], cfg: dict) -> None:
    print(f"\n[{benchmark}] 1/5 gen_metrics")
    run_gen_metrics(benchmark, cfg)
    print(f"\n[{benchmark}] 2/5 gen_scenarios")
    run_gen_scenarios(benchmark, cfg)
    for model in models:
        print(f"\n[{benchmark}/{model}] 3/5 simulate")
        run_simulate(benchmark, model, cfg)
        print(f"\n[{benchmark}/{model}] 4/5 evaluate")
        run_evaluate(benchmark, model, cfg)
    print(f"\n[{benchmark}] 5/5 aggregate")
    run_aggregate(benchmark, cfg)


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
        for b in list_benchmarks():
            run_all(b, targets, cfg)

    elif args.phase == "all":
        models = [args.model] if args.model else targets
        run_all(args.benchmark, models, cfg)

    elif args.phase == "gen_metrics":
        run_gen_metrics(args.benchmark, cfg)

    elif args.phase == "gen_scenarios":
        run_gen_scenarios(args.benchmark, cfg)

    elif args.phase == "aggregate":
        run_aggregate(args.benchmark, cfg)

    elif args.phase in ("simulate", "evaluate"):
        if not args.model:
            parser.error(f"model required for '{args.phase}'")
        fn = run_simulate if args.phase == "simulate" else run_evaluate
        fn(args.benchmark, args.model, cfg)

    print("\ndone.")


if __name__ == "__main__":
    main()
