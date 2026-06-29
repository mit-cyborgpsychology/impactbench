# Phase 4: one LLM call scores all metrics per conversation (batched for cost).
from __future__ import annotations
import json
from pathlib import Path

from lib.core import evaluate as ev, cost
from lib.task import concurrent, retry, row_cache, write_json
from lib.pipeline.utils import load_yaml

ROOT = Path(__file__).parent.parent.parent

_DROP = {"persona", "user_persona", "user_goal", "latent_adversarial_goal",
         "landmarks", "target", "transcript", "demographic", "_sample", "_usage"}


def run(benchmark: str, model: str, cfg: dict) -> None:
    bench_dir = ROOT / "benchmarks" / benchmark
    run_dir = bench_dir / "runs" / model
    goal = load_yaml(bench_dir / "benchmark.yaml")
    metrics = goal["metrics"]
    conversations = json.loads((run_dir / "conversations.json").read_text())
    workers = cfg.get("run", {}).get("concurrency", {}).get("evaluate", 20)

    @concurrent(workers)
    @retry(3)
    @row_cache(
        ROOT / ".cache" / benchmark / model / "eval",
        key=lambda r: f"{r['id']}__s{r.get('_sample', 0)}__{r['target']['id']}.json",
    )
    def evaluate_step(row):
        scores, usage = ev.evaluate_batch(row, metrics, cfg["evaluator_model"])
        base = {k: v for k, v in row.items() if k not in _DROP}
        base["target_model"] = row["target"]["id"]
        base["sample"] = row.get("_sample", 0)
        return {
            "scores": [{**base, "metric_id": mid, **score} for mid, score in scores.items()],
            "_usage": usage.to_json(),
        }

    units = evaluate_step(conversations)
    result = [row for unit in units for row in unit["scores"]]
    write_json(run_dir / "scores.json", result)

    cost.report((unit["_usage"] for unit in units), run_dir / "cost.json", "evaluate")
    passed = sum(1 for r in result if r.get("score") == 1)
    scored = sum(1 for r in result if r.get("score") is not None)
    print(f"  evaluated {len(conversations)} conversations → {len(result)} scores "
          f"({passed}/{scored} passed)")
