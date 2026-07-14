"""
Aggregation correctness.

The load-bearing invariant: score rows carry `metric_type`, and pass rates are
computed per type. If `metric_type` ever stops flowing through to score rows,
every pass rate silently becomes None — these tests pin that.
"""
from lib.core import aggregate as agg


def _row(passed, metric_type, **extra):
    # accepts 1/0/None for readability; stored as the bool `passed` field
    return {"passed": None if passed is None else bool(passed), "metric_type": metric_type, **extra}


def test_pass_rate_splits_by_type():
    rows = [
        _row(1, "positive"),
        _row(0, "positive"),
        _row(1, "negative"),
    ]
    assert agg.pass_rate(rows, "positive") == 0.5
    assert agg.pass_rate(rows, "negative") == 1.0


def test_pass_rate_none_when_no_rows_of_type():
    rows = [_row(1, "positive")]
    assert agg.pass_rate(rows, "negative") is None


def test_pass_rate_ignores_none_scores():
    rows = [_row(1, "positive"), _row(None, "positive")]
    assert agg.pass_rate(rows, "positive") == 1.0


def test_missing_metric_type_yields_no_pass_rate():
    """If metric_type stops flowing through, this catches it."""
    rows = [{"passed": True}, {"passed": False}]  # no metric_type key
    assert agg.pass_rate(rows, "positive") is None
    assert agg.type_count(rows, "positive") == 0


def test_type_count():
    rows = [
        _row(1, "positive"),
        _row(0, "positive"),
        _row(1, "negative"),
    ]
    assert agg.type_count(rows, "positive") == 2
    assert agg.type_count(rows, "negative") == 1


def test_summarize_by_groups_and_skips_none():
    rows = [
        _row(1, "positive", metric_id="m01"),
        _row(0, "positive", metric_id="m01"),
        _row(None, "positive", metric_id="m02"),
    ]
    out = agg.summarize_by(rows, "metric_id")
    assert out["m01"] == {"pass_rate": 0.5, "n_passed": 1, "n_total": 2}
    assert out["m02"]["pass_rate"] is None  # only a None passed → no data
    assert out["m02"]["n_total"] == 0
