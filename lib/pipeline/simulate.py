# Phase 3: simulate conversations. Set generation.num_samples > 1 for multiple
# independent samples per scenario.
from __future__ import annotations
import json

from lib import config
from lib.core import simulate as sim, cost
from lib.paths import BENCHMARKS, CACHE
from lib.task import concurrent, retry, row_cache, write_json
from lib.pipeline.utils import load_yaml


def _persistable(row: dict, transcript: list[dict], usage) -> dict:
    # Strip the full target cfg (it carries the apikey field) from anything
    # written to disk; keep only the id as target_model.
    out = {k: v for k, v in row.items() if k != "target"}
    return {**out, "target_model": row["target"]["id"],
            "transcript": transcript, "_usage": usage.to_json()}


def run(benchmark: str, model: str, cfg: dict) -> None:
    bench_dir = BENCHMARKS / benchmark
    run_dir = bench_dir / "runs" / model
    goal = load_yaml(bench_dir / "benchmark.yaml")
    target = next(t for t in cfg["targets"] if t["id"] == model)
    metrics_by_id = {m["id"]: m for m in goal["metrics"]}
    scenarios = json.loads((bench_dir / "scenarios.json").read_text())
    num_samples = config.num_samples(cfg)

    # Cache key encodes sample index so each sample resumes independently.
    rows = [
        {**s, "target": target, "_sample": i}
        for s in scenarios
        for i in range(num_samples)
    ]

    @concurrent(config.concurrency(cfg, "simulate"))
    @retry(3)
    @row_cache(
        CACHE / benchmark / model,
        key=lambda r: f"{r['id']}__s{r['_sample']}__{r['target']['id']}.json",
        force=config.force(cfg),
    )
    def simulate_step(row, resume=None, checkpoint=None):
        if resume:
            print(f"  [resume] {row['id']} sample={row['_sample']} "
                  f"from turn {len(resume['transcript']) // 2 + 1}", flush=True)
        else:
            print(f"  [start] {row['id']} sample={row['_sample']}", flush=True)

        def on_turn(transcript, usage):
            if checkpoint:
                checkpoint(_persistable(row, transcript, usage))

        transcript, usage = sim.simulate(
            row, row["target"], goal, cfg, cfg["user_model"],
            metric=metrics_by_id.get(row["metric_id"]),
            perfunctory=config.perfunctory(cfg),
            landmarks=config.landmarks(cfg),
            verbose=config.verbose(cfg),
            resume=resume,
            on_turn=on_turn,
        )
        return _persistable(row, transcript, usage)

    result = simulate_step(rows)
    write_json(run_dir / "conversations.json", result)

    cost.report((r["_usage"] for r in result), run_dir / "cost.json", "simulate")
    print(f"  simulated {len(result)} conversations ({num_samples} sample(s) × {len(scenarios)} scenarios)")
