from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, TextIO


def atomic_write(path: Path, write_fn: Callable[[TextIO], None]) -> None:
    """Write to a temp file in the target directory, then rename into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            write_fn(f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json(path: Path, value: object) -> None:
    atomic_write(path, lambda f: json.dump(value, f, indent=2))
