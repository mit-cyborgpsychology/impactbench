# Single source of truth for repo-relative paths — everything else imports from here.
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).parent.parent
BENCHMARKS = ROOT / "benchmarks"
PROMPTS = ROOT / "prompts"
CACHE = ROOT / ".cache"
