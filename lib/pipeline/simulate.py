from __future__ import annotations
import json

from lib import config
from lib.core import simulate as sim, cost
from lib.task import concurrent, retry, row_cache, write_json
from lib.pipeline.utils import load_yaml


# Everything else about the scenario stays in scenarios.json and is joined
# back on `id`, so a copy here can't go stale. The full target cfg is never
# persisted — it carries the apikey.
_KEEP = ("id", "metric_id", "conv_id", "_sample")


def _persistable(row: dict, transcript: list[dict], user_usage, target_usage,
                  censorship_events: list[dict]) -> dict:
    out = {k: row[k] for k in _KEEP if k in row}
    result = {**out, "target_model": row["target"]["id"],
              "transcript": transcript,
              "_usage_user": user_usage.to_json(),
              "_usage_target": target_usage.to_json()}
    if censorship_events:
        result["censorship_events"] = censorship_events
    return result


def run(benchmark: str, model: str, cfg: dict) -> None:
    bench_dir = config.benchmarks_dir(cfg) / benchmark
    run_dir = bench_dir / "runs" / model
    goal = load_yaml(bench_dir / "benchmark.yaml")
    target = next(t for t in cfg["targets"] if t["id"] == model)
    metrics_by_id = {m["id"]: m for m in goal["metrics"]}
    all_scenarios = json.loads((bench_dir / "scenarios.json").read_text())
    demo_filter = config.demographic_filter(cfg)
    scenarios = [s for s in all_scenarios if config.scenario_matches_filter(s, demo_filter)]
    num_samples = config.num_samples(cfg)

    # Cache key encodes sample index so each sample resumes independently.
    rows = [
        {**s, "target": target, "_sample": i}
        for s in scenarios
        for i in range(num_samples)
    ]

    @concurrent(config.concurrency(cfg, "simulate"))
    @retry(10)
    @row_cache(
        config.cache_dir(cfg) / benchmark / model,
        key=lambda r: f"{r['id']}__s{r['_sample']}__{r['target']['id']}.json",
        force=config.force(cfg),
    )
    def simulate_step(row, resume=None, checkpoint=None):
        if resume:
            print(f"  [resume] {row['id']} sample={row['_sample']} "
                  f"from turn {len(resume['transcript']) // 2 + 1}", flush=True)
        else:
            print(f"  [start] {row['id']} sample={row['_sample']}", flush=True)

        def on_turn(transcript, user_usage, target_usage, censorship_events):
            if checkpoint:
                checkpoint(_persistable(row, transcript, user_usage, target_usage, censorship_events))

        transcript, user_usage, target_usage, censorship_events = sim.simulate(
            row, row["target"], cfg, cfg["user_model"],
            user_model_fallbacks=config.user_model_fallbacks(cfg),
            metric=metrics_by_id.get(row["metric_id"]),
            perfunctory=config.perfunctory(cfg),
            landmarks=config.landmarks(cfg),
            verbose=config.verbose(cfg),
            resume=resume,
            on_turn=on_turn,
        )
        return _persistable(row, transcript, user_usage, target_usage, censorship_events)

    result = simulate_step(rows)
    write_json(run_dir / "conversations.json", result)

    cost.report((r["_usage_user"] for r in result), run_dir / "cost.json", "simulate_user")
    cost.report((r["_usage_target"] for r in result), run_dir / "cost.json", "simulate_target")
    filter_note = f" (filtered {len(scenarios)}/{len(all_scenarios)} by {demo_filter})" if demo_filter else ""
    print(f"  simulated {len(result)} conversations ({num_samples} sample(s) × {len(scenarios)} scenarios){filter_note}")
