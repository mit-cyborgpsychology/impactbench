# Phase 3: simulate conversations. Set generation.num_samples > 1 for multiple
# independent samples per scenario.
from __future__ import annotations
import json
from pathlib import Path

from lib.core import simulate as sim, cost
from lib.task import concurrent, retry, row_cache, write_json
from lib.pipeline.utils import load_yaml

ROOT = Path(__file__).parent.parent.parent


def run(benchmark: str, model: str, cfg: dict) -> None:
    bench_dir = ROOT / "benchmarks" / benchmark
    run_dir = bench_dir / "runs" / model
    goal = load_yaml(bench_dir / "benchmark.yaml")
    target = next(t for t in cfg["targets"] if t["id"] == model)
    metrics_by_id = {m["id"]: m for m in goal["metrics"]}
    scenarios = json.loads((bench_dir / "scenarios.json").read_text())
    num_samples = cfg.get("generation", {}).get("num_samples", 1)
    workers = cfg.get("run", {}).get("concurrency", {}).get("simulate", 10)

    # Cache key encodes sample index so each sample resumes independently.
    rows = [
        {**s, "target": target, "_sample": i}
        for s in scenarios
        for i in range(num_samples)
    ]

    @concurrent(workers)
    @retry(3)
    @row_cache(
        ROOT / ".cache" / benchmark / model,
        key=lambda r: f"{r['id']}__s{r['_sample']}__{r['target']['id']}.json",
    )
    def simulate_step(row):
        transcript, usage = sim.simulate(
            row, row["target"], goal, cfg, cfg["user_model"],
            metric=metrics_by_id.get(row["metric_id"]),
            perfunctory=cfg.get("perfunctory", False),
            pinpoint=cfg.get("landmarks", True),
        )
        return {**row, "transcript": transcript, "_usage": usage.to_json()}

    result = simulate_step(rows)
    write_json(run_dir / "conversations.json", result)

    cost.report((r["_usage"] for r in result), run_dir / "cost.json", "simulate")
    print(f"  simulated {len(result)} conversations ({num_samples} sample(s) × {len(scenarios)} scenarios)")
