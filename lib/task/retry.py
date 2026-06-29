from __future__ import annotations
import time
from functools import wraps
from typing import Callable


def retry(n: int, sleep_fn: Callable[[float], None] = time.sleep):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(n + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < n:
                        print(f"  [retry {attempt + 1}/{n}] {exc}")
                        sleep_fn(float(2 ** attempt))
            raise last_exc  # type: ignore
        return wrapper
    return decorator
