"""
Cost tracking: correctness under resume and per-phase isolation.

The whole point of the Usage design is that cost is a pure projection of the
work artifacts (the cached rows), not a running counter. These tests pin that.
"""
import json
from lib.core.usage import Usage, total
from lib.core import cost


# --- Usage value semantics ---

def test_usage_add():
    assert (Usage(1.0, 100, 50) + Usage(0.5, 10, 5)).to_json() == {
        "cost": 1.5, "input_tokens": 110, "output_tokens": 55,
    }


def test_total_mixes_usage_and_json():
    summed = total([Usage(1.0, 2, 3), {"cost": 0.5, "input_tokens": 1, "output_tokens": 1}])
    assert summed.to_json() == {"cost": 1.5, "input_tokens": 3, "output_tokens": 4}


def test_from_json_tolerates_missing_keys():
    assert Usage.from_json(None).to_json() == {"cost": 0.0, "input_tokens": 0, "output_tokens": 0}
    assert Usage.from_json({"cost": 2.0}).to_json() == {"cost": 2.0, "input_tokens": 0, "output_tokens": 0}


# --- Resume correctness ---

def test_resume_reports_identical_cost(tmp_path):
    """Reporting the same rows twice (fresh, then resumed-from-cache) gives the same total."""
    rows = [Usage(1.0, 100, 50).to_json() for _ in range(3)]
    path = tmp_path / "cost.json"

    fresh = cost.report(iter(rows), path, "simulate")
    resumed = cost.report(iter(rows), path, "simulate")  # same artifacts, re-summed

    assert fresh.to_json() == resumed.to_json()
    assert json.loads(path.read_text())["simulate"]["cost"] == 3.0


def test_phases_are_isolated(tmp_path):
    """Writing a second phase does not accumulate the first phase's cost."""
    path = tmp_path / "cost.json"
    cost.report([Usage(3.0, 0, 0).to_json()], path, "simulate")
    cost.report([Usage(0.4, 0, 0).to_json()], path, "evaluate")

    data = json.loads(path.read_text())
    assert data["simulate"]["cost"] == 3.0
    assert data["evaluate"]["cost"] == 0.4


def test_report_empty_is_zero(tmp_path):
    summed = cost.report([], tmp_path / "cost.json", "generate")
    assert summed.to_json() == {"cost": 0.0, "input_tokens": 0, "output_tokens": 0}
