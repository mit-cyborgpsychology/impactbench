from __future__ import annotations
import json
from functools import wraps
from pathlib import Path
from typing import Callable

from lib.task.io import write_json


def row_cache(cache_dir: str | Path, key: Callable | None = None):
    cache_dir = Path(cache_dir)

    def decorator(fn):
        @wraps(fn)
        def wrapper(row):
            name = key(row) if key else f"{row['id']}.json"
            path = cache_dir / name
            if path.exists():
                try:
                    return json.loads(path.read_text())
                except (json.JSONDecodeError, OSError):
                    pass
            result = fn(row)
            write_json(path, result)
            return result
        return wrapper
    return decorator
