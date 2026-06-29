from __future__ import annotations
from pathlib import Path

from lib.core import gen_metrics as gen, cost
from lib.pipeline.utils import load_yaml, save_yaml

ROOT = Path(__file__).parent.parent.parent


def run(benchmark: str, cfg: dict) -> None:
    bench_dir = ROOT / "benchmarks" / benchmark
    goal = load_yaml(bench_dir / "benchmark.yaml")

    metrics, usage = gen.generate_metrics(
        name=goal["name"],
        description=goal["description"],
        user_cfg=cfg["user_model"],
    )

    goal["metrics"] = metrics
    save_yaml(bench_dir / "benchmark.yaml", goal)

    cost.report([usage.to_json()], bench_dir / "cost.json", "gen_metrics")
    print(f"  generated {len(metrics)} metrics")
