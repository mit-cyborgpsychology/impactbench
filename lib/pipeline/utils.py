from __future__ import annotations
from pathlib import Path

import yaml

from lib.task.io import atomic_write


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def save_yaml(path: Path, data: dict) -> None:
    atomic_write(path, lambda f: yaml.dump(data, f, allow_unicode=True, sort_keys=False))
