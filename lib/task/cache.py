from __future__ import annotations
import inspect
import json
from functools import wraps
from pathlib import Path
from typing import Callable

from lib.task.io import write_json

_SENTINEL = "__row_cache_partial__"


def row_cache(cache_dir: str | Path, key: Callable | None = None, force: bool = False):
    cache_dir = Path(cache_dir)

    def decorator(fn):
        wants_resume = "resume" in inspect.signature(fn).parameters
        wants_checkpoint = "checkpoint" in inspect.signature(fn).parameters

        @wraps(fn)
        def wrapper(row):
            name = key(row) if key else f"{row['id']}.json"
            path = cache_dir / name
            resume = None
            if path.exists() and not force:
                try:
                    cached = json.loads(path.read_text())
                except (json.JSONDecodeError, OSError):
                    cached = None
                if cached is not None:
                    if not (isinstance(cached, dict) and cached.get(_SENTINEL)):
                        return cached
                    resume = {k: v for k, v in cached.items() if k != _SENTINEL}

            kwargs = {}
            if wants_resume:
                kwargs["resume"] = resume
            if wants_checkpoint:
                kwargs["checkpoint"] = lambda partial: write_json(path, {**partial, _SENTINEL: True})

            result = fn(row, **kwargs)
            write_json(path, result)
            return result
        return wrapper
    return decorator
