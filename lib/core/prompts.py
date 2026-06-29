from pathlib import Path
from string import Template

ROOT = Path(__file__).parent.parent.parent
_DIR = ROOT / "prompts"


def load(name: str, **kwargs) -> str:
    text = (_DIR / name).read_text()
    if not kwargs:
        return text
    try:
        return Template(text).substitute(**kwargs)
    except KeyError as e:
        raise KeyError(
            f"prompt template {name!r} references ${e.args[0]} "
            f"but it was not provided (given: {sorted(kwargs)})"
        ) from e
