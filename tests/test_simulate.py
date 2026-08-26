"""core/simulate helpers — message parsing, prompt blocks, content-filter fallback."""
import pytest
from lib.core.client import ContentFilterError, EmptyStopError, SafetyRejectionError
from lib.core.simulate import (
    _adversarial_block,
    _call_user_with_fallback,
    _parse_message,
    _user_system,
    simulate,
)
from lib.core.usage import Usage

_U = Usage()
_FB1 = {"model": "fallback-1"}
_FB2 = {"model": "fallback-2"}


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


def test_parse_message_empty_raw_returns_empty_string_not_none():
    # _parse_message itself does not treat blank content as termination --
    # that would silently truncate the scenario. Distinguishing "explicit
    # terminate" (deliberate end, None) from "empty output" (a failure that
    # must trigger the fallback-then-raise path) is _call_user_with_fallback's
    # job, one layer up, not this parser's.
    assert _parse_message("", perfunctory=False) == ""
    assert _parse_message("   ", perfunctory=False) == ""


def test_parse_message_empty_message_capture_returns_empty_string():
    raw = "Thought: nothing to add.\nMessage:   "
    assert _parse_message(raw, perfunctory=False) == ""


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


# --- _call_user_with_fallback ---

def test_call_user_with_fallback_no_event_on_success(monkeypatch):
    monkeypatch.setattr("lib.core.simulate.call", lambda cfg, messages: ("Message: hi", _U, None))
    msg, u, reasoning, event = _call_user_with_fallback({"model": "primary"}, [_FB1], [], "sid", 1, False, verbose=False)
    assert msg == "hi"
    assert event is None


def test_call_user_with_fallback_no_fallbacks_configured_raises_immediately(monkeypatch):
    # An empty fallback list (or config with none set) must behave exactly
    # like the pre-fallback code: one attempt, raise on failure.
    monkeypatch.setattr(
        "lib.core.simulate.call",
        lambda cfg, messages: (_ for _ in ()).throw(ContentFilterError("blocked")),
    )
    with pytest.raises(RuntimeError, match="failed on every model"):
        _call_user_with_fallback({"model": "primary"}, [], [], "sid", 1, False, verbose=False)


def test_call_user_with_fallback_tries_first_fallback_on_content_filter(monkeypatch):
    calls = []

    def fake_call(cfg, messages):
        calls.append(cfg["model"])
        if cfg["model"] == "primary":
            raise ContentFilterError("blocked")
        return "Message: fallback reply", _U, None

    monkeypatch.setattr("lib.core.simulate.call", fake_call)
    msg, u, reasoning, event = _call_user_with_fallback(
        {"model": "primary"}, [_FB1, _FB2], [], "sid", 3, False, verbose=False,
    )
    assert calls == ["primary", "fallback-1"]
    assert msg == "fallback reply"
    assert event == {"turn": 3, "role": "user", "reason": "content_filter",
                      "primary_model": "primary", "fallback_used": True, "fallback_model": "fallback-1"}


def test_call_user_with_fallback_falls_through_to_second_fallback(monkeypatch):
    # First fallback also fails -- must try the second before giving up.
    calls = []

    def fake_call(cfg, messages):
        calls.append(cfg["model"])
        if cfg["model"] in ("primary", "fallback-1"):
            raise ContentFilterError("blocked")
        return "Message: second fallback reply", _U, None

    monkeypatch.setattr("lib.core.simulate.call", fake_call)
    msg, u, reasoning, event = _call_user_with_fallback(
        {"model": "primary"}, [_FB1, _FB2], [], "sid", 1, False, verbose=False,
    )
    assert calls == ["primary", "fallback-1", "fallback-2"]
    assert msg == "second fallback reply"
    assert event["fallback_model"] == "fallback-2"


def test_call_user_with_fallback_switches_model_on_empty_message(monkeypatch):
    # Not a content-filter block -- the primary just returns "" after
    # parsing. Must still trigger the same fallback, not silently proceed.
    calls = []

    def fake_call(cfg, messages):
        calls.append(cfg["model"])
        if cfg["model"] == "primary":
            return "Message:    ", _U, None  # parses to ""
        return "Message: fallback reply", _U, None

    monkeypatch.setattr("lib.core.simulate.call", fake_call)
    msg, u, reasoning, event = _call_user_with_fallback(
        {"model": "primary"}, [_FB1], [], "sid", 2, False, verbose=False,
    )
    assert calls == ["primary", "fallback-1"]
    assert msg == "fallback reply"
    assert event["reason"] == "empty_message"


def test_call_user_with_fallback_raises_runtime_error_if_all_fail(monkeypatch):
    # Every scenario must complete for real -- if the primary and every
    # fallback fail (blocked or empty either way), this must raise, never
    # silently return None/"" and let the scenario quietly truncate.
    monkeypatch.setattr(
        "lib.core.simulate.call",
        lambda cfg, messages: (_ for _ in ()).throw(ContentFilterError("blocked")),
    )
    with pytest.raises(RuntimeError, match="failed on every model"):
        _call_user_with_fallback({"model": "primary"}, [_FB1, _FB2], [], "sid", 1, False, verbose=False)


def test_call_user_with_fallback_terminate_is_not_an_error(monkeypatch):
    # An explicit "terminate conversation" must stay a clean None, not get
    # treated as a failure requiring fallback -- it's a deliberate scenario
    # end, distinct from the model producing nothing.
    calls = []

    def fake_call(cfg, messages):
        calls.append(cfg["model"])
        return "Message: terminate conversation", _U, None

    monkeypatch.setattr("lib.core.simulate.call", fake_call)
    msg, u, reasoning, event = _call_user_with_fallback(
        {"model": "primary"}, [_FB1], [], "sid", 1, False, verbose=False,
    )
    assert msg is None
    assert event is None
    assert calls == ["primary"]


# --- simulate() censorship_events threading ---

def _cfg(turns=2):
    return {"generation": {"turns": turns}}


def test_simulate_returns_four_tuple_with_empty_events_on_clean_run(monkeypatch):
    monkeypatch.setattr("lib.core.simulate.load", lambda name, **kw: "sys prompt" if "system" not in name else "")
    monkeypatch.setattr(
        "lib.core.simulate.call",
        lambda cfg, messages: ("Message: terminate conversation", _U, None),
    )
    scenario = {"id": "s1", "user_persona": "p", "user_goal": "g"}
    transcript, user_usage, target_usage, events = simulate(
        scenario, target={"model": "t"}, cfg=_cfg(), user_cfg={"model": "u"},
    )
    assert transcript == []
    assert events == []


def test_simulate_records_event_and_uses_fallback_for_user_content_filter(monkeypatch):
    monkeypatch.setattr("lib.core.simulate.load", lambda name, **kw: "")
    # real _parse_message (unmocked): "terminate conversation" -> None, which
    # ends the turn loop after the fallback succeeds, so only turn 1 fires.

    call_log = []

    def fake_call(cfg, messages):
        call_log.append(cfg.get("model"))
        if cfg.get("model") == "primary-user":
            raise ContentFilterError("blocked")
        if cfg.get("model") == "fallback-1":
            return "terminate conversation", _U, None
        return "target reply", _U, None

    monkeypatch.setattr("lib.core.simulate.call", fake_call)
    scenario = {"id": "s1", "user_persona": "p", "user_goal": "g"}
    transcript, user_usage, target_usage, events = simulate(
        scenario, target={"model": "target-x"}, cfg=_cfg(), user_cfg={"model": "primary-user"},
        user_model_fallbacks=[_FB1],
    )
    assert len(events) == 1
    assert events[0]["role"] == "user"
    assert events[0]["fallback_used"] is True


def test_simulate_no_fallbacks_configured_behaves_like_before(monkeypatch):
    # user_model_fallbacks defaults to None/absent -- must act as "no
    # fallback available", not crash on a missing list.
    monkeypatch.setattr("lib.core.simulate.load", lambda name, **kw: "")
    monkeypatch.setattr(
        "lib.core.simulate.call",
        lambda cfg, messages: (_ for _ in ()).throw(ContentFilterError("blocked")),
    )
    scenario = {"id": "s1", "user_persona": "p", "user_goal": "g"}
    with pytest.raises(RuntimeError, match="failed on every model"):
        simulate(scenario, target={"model": "target-x"}, cfg=_cfg(), user_cfg={"model": "primary-user"})


def test_simulate_target_content_filter_has_no_fallback_and_reraises(monkeypatch):
    monkeypatch.setattr("lib.core.simulate.load", lambda name, **kw: "")
    monkeypatch.setattr("lib.core.simulate._parse_message", lambda raw, perfunctory: raw)

    def fake_call(cfg, messages):
        if cfg.get("model") == "target-x":
            raise ContentFilterError("target blocked")
        return "user message", _U, None

    monkeypatch.setattr("lib.core.simulate.call", fake_call)
    scenario = {"id": "s1", "user_persona": "p", "user_goal": "g"}
    with pytest.raises(ContentFilterError):
        simulate(scenario, target={"model": "target-x"}, cfg=_cfg(), user_cfg={"model": "primary-user"})


def test_simulate_target_empty_stop_has_no_fallback_and_reraises(monkeypatch):
    # finish_reason=stop with empty content is a distinct, named failure --
    # recorded with reason "empty_stop" (not "content_filter"), same
    # no-fallback treatment as a real content block: the target is the
    # thing under test, so this is a measurement, not routed around.
    monkeypatch.setattr("lib.core.simulate.load", lambda name, **kw: "")
    monkeypatch.setattr("lib.core.simulate._parse_message", lambda raw, perfunctory: raw)

    def fake_call(cfg, messages):
        if cfg.get("model") == "target-x":
            raise EmptyStopError("target returned empty content on stop")
        return "user message", _U, None

    monkeypatch.setattr("lib.core.simulate.call", fake_call)
    checkpoints = []
    scenario = {"id": "s1", "user_persona": "p", "user_goal": "g"}
    with pytest.raises(EmptyStopError):
        simulate(
            scenario, target={"model": "target-x"}, cfg=_cfg(), user_cfg={"model": "primary-user"},
            on_turn=lambda transcript, uu, tu, events: checkpoints.append(list(events)),
        )
    assert checkpoints[-1][0]["role"] == "target"
    assert checkpoints[-1][0]["reason"] == "empty_stop"


def test_simulate_target_safety_rejection_has_no_fallback_and_reraises(monkeypatch):
    # A provider-level 403 safety rejection is recorded with its own
    # distinct reason ("safety_rejection"), same no-fallback treatment as
    # the other two target-side block shapes.
    monkeypatch.setattr("lib.core.simulate.load", lambda name, **kw: "")
    monkeypatch.setattr("lib.core.simulate._parse_message", lambda raw, perfunctory: raw)

    def fake_call(cfg, messages):
        if cfg.get("model") == "target-x":
            raise SafetyRejectionError("rejected by provider safety check")
        return "user message", _U, None

    monkeypatch.setattr("lib.core.simulate.call", fake_call)
    checkpoints = []
    scenario = {"id": "s1", "user_persona": "p", "user_goal": "g"}
    with pytest.raises(SafetyRejectionError):
        simulate(
            scenario, target={"model": "target-x"}, cfg=_cfg(), user_cfg={"model": "primary-user"},
            on_turn=lambda transcript, uu, tu, events: checkpoints.append(list(events)),
        )
    assert checkpoints[-1][0]["role"] == "target"
    assert checkpoints[-1][0]["reason"] == "safety_rejection"


def test_simulate_checkpoints_target_censorship_event_before_reraising(monkeypatch):
    # The whole call raises (row_cache/retry discard the return value), so
    # the event must reach the caller via on_turn's checkpoint, not just the
    # (unreachable) return value.
    monkeypatch.setattr("lib.core.simulate.load", lambda name, **kw: "")
    monkeypatch.setattr("lib.core.simulate._parse_message", lambda raw, perfunctory: raw)

    def fake_call(cfg, messages):
        if cfg.get("model") == "target-x":
            raise ContentFilterError("target blocked")
        return "user message", _U, None

    monkeypatch.setattr("lib.core.simulate.call", fake_call)
    checkpoints = []
    scenario = {"id": "s1", "user_persona": "p", "user_goal": "g"}
    with pytest.raises(ContentFilterError):
        simulate(
            scenario, target={"model": "target-x"}, cfg=_cfg(), user_cfg={"model": "primary-user"},
            on_turn=lambda transcript, uu, tu, events: checkpoints.append(list(events)),
        )
    assert checkpoints, "on_turn should have been called with the censorship event before the raise"
    assert checkpoints[-1][0]["role"] == "target"


def test_simulate_raises_when_user_model_and_all_fallbacks_produce_empty(monkeypatch):
    # Every scenario must complete for real: if the user-simulator can't
    # produce a usable turn on the primary or any fallback, simulate() must
    # raise (so retry()/row_cache never write a truncated "success"), not
    # silently end the conversation early.
    monkeypatch.setattr("lib.core.simulate.load", lambda name, **kw: "")
    monkeypatch.setattr("lib.core.simulate.call", lambda cfg, messages: ("Message:   ", _U, None))
    scenario = {"id": "s1", "user_persona": "p", "user_goal": "g"}
    with pytest.raises(RuntimeError, match="failed on every model"):
        simulate(
            scenario, target={"model": "target-x"}, cfg=_cfg(), user_cfg={"model": "primary-user"},
            user_model_fallbacks=[_FB1],
        )


def test_simulate_resume_carries_forward_prior_censorship_events(monkeypatch):
    monkeypatch.setattr("lib.core.simulate.load", lambda name, **kw: "")
    monkeypatch.setattr(
        "lib.core.simulate.call",
        lambda cfg, messages: ("terminate conversation", _U, None),
    )
    scenario = {"id": "s1", "user_persona": "p", "user_goal": "g"}
    resume = {
        "transcript": [],
        "_usage_user": {"cost": 0, "input_tokens": 0, "output_tokens": 0},
        "_usage_target": {"cost": 0, "input_tokens": 0, "output_tokens": 0},
        "censorship_events": [{"turn": 1, "role": "target", "reason": "content_filter", "context": "x"}],
    }
    transcript, user_usage, target_usage, events = simulate(
        scenario, target={"model": "t"}, cfg=_cfg(), user_cfg={"model": "u"}, resume=resume,
    )
    assert events == resume["censorship_events"]
