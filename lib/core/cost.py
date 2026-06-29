# Cost is summed from all rows on every run (cached + fresh), so a resumed run
# reports the same total as a fresh run. No running counter to drift.
from __future__ import annotations
import json
from pathlib import Path

from lib.core.usage import Usage, total
from lib.task.io import write_json


def report(usages, path: str | Path, phase: str) -> Usage:
    summed = total(usages)

    out = Path(path)
    existing = {}
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except Exception:
            pass

    existing[phase] = {**summed.to_json(), "phase": phase}
    write_json(out, existing)

    tokens = summed.input_tokens + summed.output_tokens
    print(f"  cost: ${summed.cost:.4f}  tokens: {tokens:,} "
          f"({summed.input_tokens:,} in / {summed.output_tokens:,} out)")
    return summed
