from __future__ import annotations
import re
import random

_WORD_RE = re.compile(r"[A-Za-z]+")
_MID_PERIOD_RE = re.compile(r"(?<!\.)\. (?=[a-z])")
_CONJ_COMMA_RE = re.compile(r", (but|and|so|because|which)\b")
_BANG_RE = re.compile(r"(?<!\!)!(?!\!)")
_QMARK_RE = re.compile(r"(?<!\?)\?(?!\?)")


def perfunctory(text: str) -> str:
    text = _mangle_punctuation(text)
    chars = list(text)
    n = len(chars)
    protected: set[int] = set()
    for m in _WORD_RE.finditer(text):
        word = m.group()
        if len(word) > 1 and word.isupper():
            protected.update(range(max(m.start() - 1, 0), min(m.end() + 1, n)))
            continue
        i = m.start()
        if chars[i].isupper() and random.random() < 0.08:
            chars[i] = chars[i].lower()
    if n > 1 and random.random() < 0.15:
        i = _random_index(n - 1, exclude=lambda j: j in protected or j + 1 in protected)
        if i is not None:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    drops = set()
    if n > 0 and random.random() < 0.1:
        i = _random_index(n, exclude=lambda j: j in protected)
        if i is not None:
            drops = {i}
    return "".join(c for i, c in enumerate(chars) if i not in drops)


def _random_index(n: int, exclude=None, max_attempts: int = 10) -> int | None:
    for _ in range(max_attempts):
        i = random.randint(0, n - 1)
        if exclude is None or not exclude(i):
            return i
    return None


def _mangle_punctuation(text: str) -> str:
    text = _MID_PERIOD_RE.sub(lambda m: random.choice(["... ", ". ", " "]) if random.random() < 0.1 else m.group(), text)
    text = _CONJ_COMMA_RE.sub(lambda m: f" {m.group(1)}" if random.random() < 0.1 else m.group(), text)
    text = _BANG_RE.sub(lambda m: "!!" if random.random() < 0.05 else "!", text)
    text = _QMARK_RE.sub(lambda m: random.choice(["??", "?!"]) if random.random() < 0.05 else "?", text)
    if text.endswith(".") and not text.endswith("..."):
        if random.random() < 0.15:
            text = text[:-1]
    return text
