"""generate_scenarios_from_seeds — seeded scenario expansion (new, doesn't
touch generate_scenarios' from-scratch behavior)."""
from lib.core import generate
from lib.core.usage import Usage

METRIC = {"id": "m01", "name": "Avoid X", "type": "negative", "definition": "def", "examples": []}
GOAL = {"name": "Bench", "description": "desc", "scenario": {"user_context": "ctx"}}
RUN = {"generation": {"scenarios_per_metric": 3}}
USER_CFG = {"model": "m", "source": "anthropic", "apikey": "k"}


def test_no_seeds_returns_empty_without_calling_llm(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("call_json should not be called with no seed titles")
    monkeypatch.setattr(generate, "call_json", boom)

    scenarios, usage = generate.generate_scenarios_from_seeds(METRIC, GOAL, RUN, USER_CFG, [])
    assert scenarios == []
    assert usage == Usage()


def test_tags_ids_metric_id_and_seed_title_in_order(monkeypatch):
    canned = {
        "scenarios": [
            {"persona": "p1", "user_goal": "g1", "latent_adversarial_goal": "a1", "landmarks": []},
            {"persona": "p2", "user_goal": "g2", "latent_adversarial_goal": "a2", "landmarks": []},
        ]
    }
    monkeypatch.setattr(generate, "call_json", lambda cfg, prompt: (canned, Usage(cost=0.01)))

    seeds = ["first seed title", "second seed title"]
    scenarios, usage = generate.generate_scenarios_from_seeds(METRIC, GOAL, RUN, USER_CFG, seeds)

    assert usage == Usage(cost=0.01)
    assert [s["id"] for s in scenarios] == ["m01_s001", "m01_s002"]
    assert all(s["metric_id"] == "m01" for s in scenarios)
    assert [s["seed_title"] for s in scenarios] == seeds


def test_prompt_includes_seed_titles_and_metric_fields(monkeypatch):
    captured = {}

    def fake_call_json(cfg, prompt):
        captured["prompt"] = prompt
        return {"scenarios": []}, Usage()

    monkeypatch.setattr(generate, "call_json", fake_call_json)
    generate.generate_scenarios_from_seeds(METRIC, GOAL, RUN, USER_CFG, ["only seed"])

    user_msg = captured["prompt"][1]["content"]
    assert "only seed" in user_msg
    assert METRIC["definition"] in user_msg
    assert GOAL["name"] in user_msg
