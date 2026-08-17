"""config.demographic_filter / scenario_matches_filter."""
from lib.config import demographic_filter, scenario_matches_filter


def test_demographic_filter_absent_returns_empty():
    assert demographic_filter({}) == {}
    assert demographic_filter({"run": {}}) == {}


def test_demographic_filter_reads_run_key():
    cfg = {"run": {"demographic_filter": {"age": "Child or teenager (6-17)"}}}
    assert demographic_filter(cfg) == {"age": "Child or teenager (6-17)"}


def test_scenario_matches_filter_empty_filter_matches_everything():
    assert scenario_matches_filter({"demographic": {"age": "Adult (18+)"}}, {}) is True
    assert scenario_matches_filter({}, {}) is True


def test_scenario_matches_filter_matches_on_all_keys():
    filt = {"age": "Child or teenager (6-17)"}
    assert scenario_matches_filter({"demographic": {"age": "Child or teenager (6-17)"}}, filt) is True
    assert scenario_matches_filter({"demographic": {"age": "Adult (18+)"}}, filt) is False
    assert scenario_matches_filter({"demographic": {}}, filt) is False
    assert scenario_matches_filter({}, filt) is False
