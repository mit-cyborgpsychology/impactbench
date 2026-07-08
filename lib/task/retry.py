from __future__ import annotations
import time
from functools import wraps
from typing import Callable

# Programming errors — retrying can't fix these, so fail fast instead of
# burning attempts and backoff sleeps. ValueError (incl. JSONDecodeError) stays
# retryable: LLM output is nondeterministic and may parse on the next attempt.
NON_RETRYABLE = (KeyError, TypeError, AttributeError, IndexError)


def retry(n: int, sleep_fn: Callable[[float], None] = time.sleep,
          non_retryable: tuple = NON_RETRYABLE):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(n + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if isinstance(exc, non_retryable):
                        raise
                    last_exc = exc
                    if attempt < n:
                        print(f"  [retry {attempt + 1}/{n}] {exc}")
                        sleep_fn(float(2 ** attempt))
            raise last_exc  # type: ignore
        return wrapper
    return decorator
