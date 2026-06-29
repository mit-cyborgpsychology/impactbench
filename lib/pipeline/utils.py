from __future__ import annotations
import tempfile
import os
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def save_yaml(path: Path, data: dict) -> None:
    """Atomic YAML write — write to temp file then rename into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise
