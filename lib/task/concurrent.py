from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps


def concurrent(workers: int):
    # workers=1 runs synchronously for deterministic testing.
    def decorator(fn):
        @wraps(fn)
        def wrapper(items: list) -> list:
            if not items:
                return []
            if workers == 1:
                results = []
                for i, item in enumerate(items):
                    results.append(fn(item))
                    print(f"  [{i + 1}/{len(items)}]", end="\r", flush=True)
                print()
                return results

            results: list = [None] * len(items)
            first_exc: Exception | None = None
            done = 0

            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_idx = {pool.submit(fn, item): i for i, item in enumerate(items)}
                print(f"  [0/{len(items)} done, {len(future_to_idx)} in flight]", end="\r", flush=True)
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    exc = future.exception()
                    if exc is not None:
                        if first_exc is None:
                            first_exc = exc
                    else:
                        results[idx] = future.result()
                        done += 1
                        print(f"  [{done}/{len(items)} done]", end="\r", flush=True)

            print()
            if first_exc is not None:
                raise first_exc
            return results
        return wrapper
    return decorator
