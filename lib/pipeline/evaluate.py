# Phase 4: score conversations. One LLM call per conversation, judging the
# metric that conversation's scenario targets.
from __future__ import annotations
import json

from lib import config
from lib.core import evaluate as ev, cost
from lib.paths import BENCHMARKS, CACHE
from lib.task import concurrent, retry, row_cache, write_json
from lib.pipeline.utils import load_yaml

# Explicit whitelist of scenario fields carried into score rows — anything not
# listed here (personas, transcripts, target cfg, ...) stays out of scores.json.
_KEEP = ("id", "metric_id", "metric_name", "metric_type")


def _target_model(row: dict) -> str:
    # Older conversations.json rows embed the full target cfg instead of target_model.
    return row.get("target_model") or row["target"]["id"]


def run(benchmark: str, model: str, cfg: dict) -> None:
    bench_dir = BENCHMARKS / benchmark
    run_dir = bench_dir / "runs" / model
    goal = load_yaml(bench_dir / "benchmark.yaml")
    metrics = goal["metrics"]
    conversations = json.loads((run_dir / "conversations.json").read_text())

    @concurrent(config.concurrency(cfg, "evaluate"))
    @retry(3)
    @row_cache(
        CACHE / benchmark / model / "eval",
        key=lambda r: f"{r['id']}__s{r.get('_sample', 0)}__{_target_model(r)}.json",
        force=config.force(cfg),
    )
    def evaluate_step(row):
        print(f"  [start] {row['id']} sample={row.get('_sample', 0)}", flush=True)
        metric = [m for m in metrics if m["id"] == row["metric_id"]]
        scores, usage = ev.evaluate_batch(row, metric, cfg["evaluator_model"])
        base = {k: row[k] for k in _KEEP if k in row}
        base["target_model"] = _target_model(row)
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
