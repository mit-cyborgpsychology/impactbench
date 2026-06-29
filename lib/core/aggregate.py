from __future__ import annotations
from pathlib import Path
import json


def load_all_scores(bench_dir: str) -> list[dict]:
    rows = []
    for path in sorted(Path(bench_dir).glob("runs/*/scores.json")):
        model = path.parent.name
        for row in json.loads(path.read_text()):
            rows.append({**row, "target_model": row.get("target_model", model)})
    return rows


def pass_rate(rows: list[dict], behavior_type: str) -> float | None:
    matching = [r["score"] for r in rows if r.get("metric_type") == behavior_type and r.get("score") is not None]
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
        valid = [r for r in group if r.get("score") is not None]
        n_passed = sum(r["score"] for r in valid)
        out[k] = {
            "pass_rate": round(n_passed / len(valid), 4) if valid else None,
            "n_passed":  n_passed,
            "n_total":   len(valid),
        }
    return out
