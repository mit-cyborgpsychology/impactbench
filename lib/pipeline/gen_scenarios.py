from __future__ import annotations

from lib import config
from lib.core import generate as gen, cost
from lib.paths import BENCHMARKS, CACHE
from lib.task import concurrent, retry, row_cache, write_json
from lib.pipeline.utils import load_yaml


def run(benchmark: str, cfg: dict) -> None:
    bench_dir = BENCHMARKS / benchmark
    goal = load_yaml(bench_dir / "benchmark.yaml")
    cache_dir = CACHE / benchmark / "gen_scenarios"
    force = config.force(cfg)

    @concurrent(config.concurrency(cfg, "gen_scenarios"))
    @retry(3)
    @row_cache(cache_dir, force=force)
    def generate_scenarios(metric):
        scenarios, usage = gen.generate_scenarios(metric, goal, cfg, cfg["user_model"])
        return {"id": metric["id"], "scenarios": scenarios, "_usage": usage.to_json()}

    @concurrent(config.concurrency(cfg, "gen_scenarios_expand"))
    @retry(3)
    @row_cache(cache_dir / "expanded", force=force)
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
