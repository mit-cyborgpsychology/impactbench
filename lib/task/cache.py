from __future__ import annotations
import inspect
import json
from functools import wraps
from pathlib import Path
from typing import Callable

from lib.task.io import write_json

_SENTINEL = "__row_cache_partial__"


def row_cache(cache_dir: str | Path, key: Callable | None = None, force: bool = False):
    """Caches fn(row) to disk.

    force=True ignores existing cache entries (complete and partial) and
    recomputes every row, overwriting the cache. This is what makes
    run.force actually regenerate results instead of re-reading stale rows.

    If fn's signature accepts `resume` and `checkpoint` keyword parameters, it
    opts into partial-progress resuming:
      - resume: the partial result from a previous incomplete run of this row
        (dict), or None if there isn't one. fn can use this to skip work
        already done instead of starting over.
      - checkpoint(partial_result): call after each incremental step to
        persist progress. Checkpointed files are marked internally so a crash
        mid-way resumes instead of being mistaken for a finished result.

    fn's that don't declare these parameters are called exactly as before —
    existing callers are unaffected, and a complete cached result is still
    returned as-is.
    """
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
