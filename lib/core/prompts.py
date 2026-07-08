from string import Template

from lib.paths import PROMPTS


def load(name: str, **kwargs) -> str:
    text = (PROMPTS / name).read_text()
    if not kwargs:
        return text
    try:
        return Template(text).substitute(**kwargs)
    except KeyError as e:
        raise KeyError(
            f"prompt template {name!r} references ${e.args[0]} "
            f"but it was not provided (given: {sorted(kwargs)})"
        ) from e
