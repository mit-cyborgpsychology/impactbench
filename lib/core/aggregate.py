from __future__ import annotations
from pathlib import Path
import json

import yaml


def load_all_scores(bench_dir: str) -> list[dict]:
    """Load every run's score rows, attaching metric_type from benchmark.yaml.

    metric_type is the source of truth in benchmark.yaml, not in scores.json
    (which stores only metric_id). Resolve it per row here so aggregation can
    group by polarity without the score rows carrying a copy.
    """
    bench = Path(bench_dir)
    spec = yaml.safe_load((bench / "benchmark.yaml").read_text())
    metric_type = {m["id"]: m.get("type") for m in spec.get("metrics") or []}

    rows = []
    for path in sorted(bench.glob("runs/*/scores.json")):
        model = path.parent.name
        for row in json.loads(path.read_text()):
            rows.append({
                **row,
                "target_model": row.get("target_model", model),
                "metric_type": metric_type.get(row.get("metric_id")),
            })
    return rows


def pass_rate(rows: list[dict], behavior_type: str) -> float | None:
    matching = [r["passed"] for r in rows if r.get("metric_type") == behavior_type and r.get("passed") is not None]
    if not matching:
        return None
    return round(sum(matching) / len(matching), 4)


def type_count(rows: list[dict], behavior_type: str) -> int:
    return sum(1 for r in rows if r.get("metric_type") == behavior_type)


def summarize_by(rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list] = {}
    for r in rows:
        k = r.get(key, "unknown")
        groups.setdefault(k, []).append(r)

    out = {}
    for k, group in sorted(groups.items()):
        valid = [r for r in group if r.get("passed") is not None]
        n_passed = sum(1 for r in valid if r["passed"])
        out[k] = {
            "pass_rate": round(n_passed / len(valid), 4) if valid else None,
            "n_passed":  n_passed,
            "n_total":   len(valid),
        }
    return out
