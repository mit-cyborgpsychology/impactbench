"""
Tests for lib/task decorators: row_cache, retry, concurrent.
Focus on non-happy paths: partial cache, transient errors, ordering.

The decorated function always takes a SINGLE row.
concurrent() turns it into a list function.
"""
import json
import pytest
from pathlib import Path
from lib.task import row_cache, retry, concurrent


# --- row_cache ---

def test_cache_miss_calls_fn(tmp_path):
    calls = []

    @row_cache(tmp_path)
    def step(row):
        calls.append(row["id"])
        return {**row, "done": True}

    # Use concurrent(1) to call over a list
    @concurrent(1)
    @row_cache(tmp_path)
    def batch_step(row):
        calls.append(row["id"])
        return {**row, "done": True}

    batch_step([{"id": "a"}, {"id": "b"}])
    assert calls == ["a", "b"]


def test_cache_hit_skips_fn(tmp_path):
    calls = []
    (tmp_path / "a.json").write_text(json.dumps({"id": "a", "done": True}))

    @concurrent(1)
    @row_cache(tmp_path)
    def step(row):
        calls.append(row["id"])
        return {**row, "done": True}

    results = step([{"id": "a"}, {"id": "b"}])
    assert calls == ["b"]  # a was cached
    assert results[0] == {"id": "a", "done": True}


def test_partial_resume(tmp_path):
    """Items already cached are skipped; uncached items run."""
    calls = []
    (tmp_path / "a.json").write_text(json.dumps({"id": "a", "done": True}))
    (tmp_path / "b.json").write_text(json.dumps({"id": "b", "done": True}))

    @concurrent(1)
    @row_cache(tmp_path)
    def step(row):
        calls.append(row["id"])
        return {**row, "done": True}

    step([{"id": "a"}, {"id": "b"}, {"id": "c"}])
    assert calls == ["c"]


def test_cache_write_is_atomic(tmp_path):
    @concurrent(1)
    @row_cache(tmp_path)
    def step(row):
        return {**row, "done": True}

    step([{"id": "x"}])
    p = tmp_path / "x.json"
    assert p.exists()
    assert json.loads(p.read_text()) == {"id": "x", "done": True}


def test_custom_key(tmp_path):
    @concurrent(1)
    @row_cache(tmp_path, key=lambda r: f"{r['model']}__{r['scenario']}.json")
    def step(row):
        return {**row, "done": True}

    step([{"id": "x", "model": "gpt-4o", "scenario": "s001"}])
    assert (tmp_path / "gpt-4o__s001.json").exists()


# --- retry ---

def test_retry_success_first_try():
    calls = []

    @retry(3, sleep_fn=lambda _: None)
    def fn(row):
        calls.append(row["id"])
        return row

    fn({"id": "a"})
    assert len(calls) == 1


def test_retry_recovers_on_transient():
    calls = []

    @retry(3, sleep_fn=lambda _: None)
    def fn(row):
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("transient")
        return row

    result = fn({"id": "a"})
    assert result == {"id": "a"}
    assert len(calls) == 3


def test_retry_exhaustion_raises():
    @retry(2, sleep_fn=lambda _: None)
    def fn(row):
        raise RuntimeError("always fails")

    with pytest.raises(RuntimeError, match="always fails"):
        fn({"id": "a"})


def test_retry_exponential_backoff():
    delays = []
    calls = [0]

    @retry(3, sleep_fn=delays.append)
    def fn(row):
        calls[0] += 1
        if calls[0] < 4:
            raise ValueError("fail")
        return row

    fn({"id": "a"})
    assert delays == [1.0, 2.0, 4.0]


# --- concurrent ---

def test_concurrent_preserves_order():
    @concurrent(4)
    def step(row):
        return {**row, "done": True}

    items = [{"id": str(i)} for i in range(10)]
    results = step(items)
    assert [r["id"] for r in results] == [str(i) for i in range(10)]


def test_concurrent_workers_1_is_synchronous():
    import threading
    thread_ids = []
    main_id = threading.get_ident()

    @concurrent(1)
    def step(row):
        thread_ids.append(threading.get_ident())
        return row

    step([{"id": "a"}, {"id": "b"}])
    assert all(tid == main_id for tid in thread_ids)


def test_concurrent_exception_propagates():
    @concurrent(4)
    def step(row):
        if row["id"] == "bad":
            raise ValueError("bad item")
        return row

    with pytest.raises(ValueError, match="bad item"):
        step([{"id": "a"}, {"id": "bad"}, {"id": "c"}])


def test_concurrent_empty_list():
    @concurrent(4)
    def step(row):
        return row

    assert step([]) == []


# --- composition: concurrent + retry + row_cache ---

def test_full_stack_cache_and_retry(tmp_path):
    """Verify all three decorators compose correctly."""
    calls = []

    @concurrent(1)
    @retry(2, sleep_fn=lambda _: None)
    @row_cache(tmp_path)
    def step(row):
        calls.append(row["id"])
        return {**row, "done": True}

    # First run — all miss cache
    step([{"id": "a"}, {"id": "b"}])
    assert calls == ["a", "b"]

    calls.clear()
    # Second run — all hit cache
    step([{"id": "a"}, {"id": "b"}])
    assert calls == []


def test_full_stack_retry_on_transient_then_cache(tmp_path):
    attempts = []

    @concurrent(1)
    @retry(2, sleep_fn=lambda _: None)
    @row_cache(tmp_path)
    def step(row):
        attempts.append(row["id"])
        if len([a for a in attempts if a == row["id"]]) < 2:
            raise ConnectionError("transient")
        return {**row, "done": True}

    results = step([{"id": "x"}])
    assert results[0] == {"id": "x", "done": True}
    assert attempts.count("x") == 2  # retried once

    # Now cached — no more calls
    before = len(attempts)
    step([{"id": "x"}])
    assert len(attempts) == before
