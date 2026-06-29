"""
Phase 4: aggregate scores across all model runs.
"""
from __future__ import annotations
import json
from pathlib import Path

from lib.core import aggregate as agg
from lib.task import write_json

ROOT = Path(__file__).parent.parent.parent


def run(benchmark: str) -> None:
    bench_dir = ROOT / "benchmarks" / benchmark
    scores = agg.load_all_scores(str(bench_dir))

    by_model: dict[str, list] = {}
    for row in scores:
        by_model.setdefault(row["target_model"], []).append(row)

    results = sorted([
        {
            "target_model":       model,
            "positive_pass_rate": agg.pass_rate(rows, "positive"),
            "negative_pass_rate": agg.pass_rate(rows, "negative"),
            "n_positive":         agg.type_count(rows, "positive"),
            "n_negative":         agg.type_count(rows, "negative"),
            "n_total":            len(rows),
            "by_metric":          agg.summarize_by(rows, "metric_id"),
            "by_scenario":        agg.summarize_by(rows, "id"),
        }
        for model, rows in by_model.items()
    ], key=lambda r: r["positive_pass_rate"] or 0, reverse=True)

    for r in results:
        print(f"  {r['target_model']}: pos={r['positive_pass_rate']} neg={r['negative_pass_rate']} n={r['n_total']}")

    write_json(bench_dir / "results.json", results)
    print(f"  saved {bench_dir / 'results.json'}")
