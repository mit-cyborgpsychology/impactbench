from __future__ import annotations
import json
from pathlib import Path

from lib.core import generate as gen, cost
from lib.task import concurrent, retry, row_cache, write_json
from lib.pipeline.utils import load_yaml

ROOT = Path(__file__).parent.parent.parent


def run(benchmark: str, cfg: dict) -> None:
    bench_dir = ROOT / "benchmarks" / benchmark
    goal = load_yaml(bench_dir / "benchmark.yaml")
    cache_dir = ROOT / ".cache" / benchmark / "gen_scenarios"
    workers = cfg.get("run", {}).get("concurrency", {}).get("gen_scenarios", 5)

    @concurrent(workers)
    @retry(3)
    @row_cache(cache_dir)
    def generate_scenarios(metric):
        scenarios, usage = gen.generate_scenarios(metric, goal, cfg, cfg["user_model"])
        return {"id": metric["id"], "scenarios": scenarios, "_usage": usage.to_json()}

    expand_workers = cfg.get("run", {}).get("concurrency", {}).get("gen_scenarios_expand", workers * 2)

    @concurrent(expand_workers)
    @retry(3)
    @row_cache(cache_dir / "expanded")
    def expand(row):
        variants, usage = gen.expand_demographics(row["scenarios"], cfg, cfg["user_model"])
        return {"id": f"{row['id']}_expanded", "variants": variants, "_usage": usage.to_json()}

    with_scenarios = generate_scenarios(goal["metrics"])
    with_variants = expand(with_scenarios)

    all_scenarios = [v for row in with_variants for v in row["variants"]]
    write_json(bench_dir / "scenarios.json", all_scenarios)

    usages = [r["_usage"] for r in with_scenarios] + [r["_usage"] for r in with_variants]
    cost.report(usages, bench_dir / "cost.json", "gen_scenarios")
    print(f"  generated {len(all_scenarios)} scenarios")
