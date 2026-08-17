"""core/simulate helpers — message parsing, perturbation, prompt blocks."""
import random
import re

from lib.core.simulate import _adversarial_block, _parse_message, _perfunctory


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


# --- _perfunctory ---

def test_perfunctory_short_text_never_invents_letters():
    # Swaps/drops can affect short text now (a flat per-message chance, not
    # a length gate), so occasional letter reordering/removal is expected —
    # but the letter set itself is never added to, only reordered or thinned.
    text = "Hey There, How Are You?"
    letters = set(text.lower()) - set(" ,?")
    for seed in range(50):
        random.seed(seed)
        out = _perfunctory(text)
        assert set(out.lower()) - set(" ,?!.") <= letters


def test_perfunctory_handles_empty_and_long_text():
    assert _perfunctory("") == ""
    long_text = "The Quick Brown Fox Jumps Over The Lazy Dog. " * 20
    out = _perfunctory(long_text)  # exercises swap and drop paths
    assert isinstance(out, str) and len(out) <= len(long_text)


def test_perfunctory_preserves_acronyms_and_shouting():
    # ALL-CAPS words (2+ letters) are deliberate — acronyms or emphasis — not
    # "forgot to hit shift" typos, so they must never be touched.
    text = "I am part of the LGBTQ+ community and NASA loves AI. EVERYBODY IS SHOUTING!!"
    for _ in range(20):
        out = _perfunctory(text)
        assert "LGBTQ+" in out
        assert "NASA" in out
        assert "AI" in out
        assert "EVERYBODY IS SHOUTING!!" in out


def test_perfunctory_never_extends_existing_ellipsis():
    # A "..." already in the text must never gain extra dots (regression: the
    # mid-period regex used to match the ellipsis's own trailing ". " and pad
    # it further, turning "and... every" into "and..... every").
    text = "ok so i looked closer and... every single post is the same summary."
    for seed in range(50):
        random.seed(seed)
        out = _perfunctory(text)
        assert "...." not in out


def test_perfunctory_mangles_punctuation():
    # Punctuation gets messier (dropped trailing period, "..." trail-offs,
    # skipped comma before conjunctions, doubled !/?) at least sometimes
    # across repeated runs.
    text = "honestly it feels easier, but idk why. is that bad?"
    seen_no_trailing_period = False
    for seed in range(50):
        random.seed(seed)
        out = _perfunctory(text)
        if not out.endswith("."):
            seen_no_trailing_period = True
    assert seen_no_trailing_period


# --- _adversarial_block ---

def test_adversarial_block_empty_definition_does_not_crash():
    block = _adversarial_block(
        {"latent_adversarial_goal": ""},
        {"name": "m", "type": "positive", "definition": ""},
    )
    assert "Target metric: m" in block


def test_adversarial_block_no_metric_is_empty():
    assert _adversarial_block({}, None) == ""
