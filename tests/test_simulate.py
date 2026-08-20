"""core/simulate helpers — message parsing, prompt blocks."""
from lib.core.simulate import _adversarial_block, _parse_message, _user_system


# --- _parse_message ---

def test_parse_message_single_line():
    raw = "Thought: I should open casually.\nMessage: hey, quick question"
    assert _parse_message(raw, perfunctory=False) == "hey, quick question"


def test_parse_message_keeps_multiline_content():
    raw = "Thought: reasoning here.\nMessage: first sentence.\nsecond sentence too."
    assert _parse_message(raw, perfunctory=False) == "first sentence.\nsecond sentence too."


def test_parse_message_terminate_returns_none():
    raw = "Thought: goal achieved.\nMessage: terminate conversation"
    assert _parse_message(raw, perfunctory=False) is None


def test_parse_message_falls_back_to_raw():
    assert _parse_message("just a bare reply", perfunctory=False) == "just a bare reply"


# --- _adversarial_block ---

def test_adversarial_block_empty_definition_does_not_crash():
    block = _adversarial_block(
        {"latent_adversarial_goal": ""},
        {"name": "m", "type": "positive", "definition": ""},
    )
    assert "Target metric: m" in block


def test_adversarial_block_no_metric_is_empty():
    assert _adversarial_block({}, None) == ""


# --- _user_system ---

def test_user_system_prefers_expanded_user_persona():
    scenario = {
        "persona": "The user had a disagreement in a close relationship.",
        "user_persona": "The user argued with their long-term partner about the lease.",
    }
    prompt = _user_system(scenario, turns=6, metric=None, perfunctory=False, landmarks=False)
    assert "long-term partner" in prompt
    assert "The user had a disagreement in a close relationship." not in prompt


def test_user_system_falls_back_to_persona_when_no_user_persona():
    scenario = {"persona": "The user had a disagreement in a close relationship."}
    prompt = _user_system(scenario, turns=6, metric=None, perfunctory=False, landmarks=False)
    assert "The user had a disagreement in a close relationship." in prompt
