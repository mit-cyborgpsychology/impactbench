from __future__ import annotations
import re
from typing import Callable

from lib.core.client import call, ContentFilterError, EmptyStopError, SafetyRejectionError
from lib.core.perfunctory import perfunctory as _mangle
from lib.core.prompts import load
from lib.core.usage import Usage

TERMINATE = "terminate conversation"


def _parse_message(raw: str, perfunctory: bool) -> str | None:
    m = re.search(r"(?i)^message:\s*(.+)", raw, re.MULTILINE | re.DOTALL)
    content = m.group(1).strip() if m else raw.strip()
    if content.lower().startswith(TERMINATE):
        return None
    return _mangle(content) if perfunctory else content


def _adversarial_block(scenario: dict, metric: dict | None) -> str:
    if not metric:
        return ""
    latent = scenario.get("latent_adversarial_goal", "").strip()
    behavior_type = metric.get("type", "")
    if behavior_type == "negative":
        failure_note = " ← FAILURE: this behavior IS present"
    else:
        failure_note = " ← FAILURE: this behavior is ABSENT"
    definition_head = metric["definition"].partition("\n")[0].rstrip(".")

    return load("adversarial_block.txt",
        latent_goal_line=f"\nLatent goal (pursue implicitly): {latent}\n" if latent else "",
        metric_name=metric["name"],
        failure_note=failure_note,
        definition_head=definition_head,
    )


def _landmarks_block(scenario: dict, turns: int) -> str:
    lines = [f"This conversation lasts at most {turns} turns."]
    landmarks = scenario.get("landmarks", [])
    if landmarks:
        lines.append("\nTURN INSTRUCTIONS:")
        for lm in landmarks:
            lines.append(f"  Turn {lm['turn']}: {lm['instruction']}")
    lines.append("")
    return "\n".join(lines)


def _user_system(scenario: dict, turns: int, metric: dict | None, perfunctory: bool, landmarks: bool) -> str:
    return load("user_system.txt",
        # user_persona is the demographic-expanded version; persona is the abstract base it came from.
        persona=scenario.get("user_persona") or scenario.get("persona", ""),
        user_goal=scenario.get("user_goal", ""),
        perfunctory_note="Occasional typos are fine." if perfunctory else "",
        landmarks_block=_landmarks_block(scenario, turns) if landmarks else "",
        adversarial_block=_adversarial_block(scenario, metric),
    )


def _turn_prompt(turn: int, turns: int, landmarks: bool) -> str:
    if turn == 1:
        return load("turn_first.txt")
    return load("turn_next.txt",
        turn_header=f"Turn {turn} of {turns}.\n" if landmarks else "",
        landmark_reminder="Check turn instructions if any apply.\n" if landmarks else "",
    )


class EmptyUserMessageError(RuntimeError):
    """The user-simulator returned content, but it parsed to nothing usable
    (blank after stripping the Thought:/Message: scaffolding). Not the same
    as an explicit "terminate conversation" — that is a deliberate scenario
    end and stays a normal None return. This is the model failing to
    produce a turn at all, which must not be forwarded to the target (it
    gets hard-rejected as invalid) and must not silently end the scenario
    either — every scenario must complete for real, so it goes through the
    same fallback-then-raise path as a content filter block.
    """


def _call_user_with_fallback(user_cfg: dict, fallbacks: list[dict], messages: list[dict], scenario_id: str,
                              turn: int, perfunctory: bool, verbose: bool) -> tuple[str | None, Usage, str | None, dict | None]:
    """Same-turn-only fallback: if the primary user-model is blocked by a
    content filter, OR returns content that parses to an empty message, try
    each config in `fallbacks` in order before giving up (by raising --
    never silently proceeding on empty content). Returns an extra
    censorship-event dict (or None) so the caller can record what happened
    regardless of which config succeeded. The returned message may still
    legitimately be None: that means the *parsed* result was an explicit
    "terminate conversation", not a failure.
    """
    def _attempt(cfg: dict) -> tuple[str | None, Usage, str | None]:
        raw, u, reasoning = call(cfg, messages)
        parsed = _parse_message(raw, perfunctory)
        if parsed == "":
            raise EmptyUserMessageError(f"{cfg.get('model')} returned an empty message on turn {turn}")
        return parsed, u, reasoning

    attempts = [(user_cfg, False)] + [(fb, True) for fb in fallbacks]
    failures = []
    for cfg, is_fallback in attempts:
        try:
            parsed, u, reasoning = _attempt(cfg)
            event = None
            if is_fallback:
                event = {"turn": turn, "role": "user", "reason": failures[-1][1],
                          "primary_model": user_cfg.get("model"), "fallback_used": True,
                          "fallback_model": cfg.get("model")}
            return parsed, u, reasoning, event
        except (ContentFilterError, EmptyUserMessageError) as exc:
            reason = "content_filter" if isinstance(exc, ContentFilterError) else "empty_message"
            failures.append((cfg.get("model"), reason))
            if verbose and fallbacks:
                next_model = fallbacks[len(failures) - 1]["model"] if len(failures) <= len(fallbacks) else None
                print(f"  [{scenario_id}] turn {turn}: {cfg.get('model')} {reason}"
                      + (f", trying {next_model}" if next_model else ""), flush=True)

    detail = ", ".join(f"{model} ({reason})" for model, reason in failures)
    raise RuntimeError(f"user turn {turn} failed on every model: {detail}")


def simulate(
    scenario: dict,
    target: dict,
    cfg: dict,
    user_cfg: dict,
    user_model_fallbacks: list[dict] | None = None,
    metric: dict | None = None,
    perfunctory: bool = False,
    landmarks: bool = True,
    verbose: bool = False,
    resume: dict | None = None,
    on_turn: Callable[[list[dict], Usage, Usage, list[dict]], None] | None = None,
) -> tuple[list[dict], Usage, Usage, list[dict]]:
    turns = cfg["generation"]["turns"]
    target_system = scenario.get("target_system_prompt") or load("default_target_system.txt").strip()
    user_system = _user_system(scenario, turns, metric, perfunctory, landmarks)

    transcript: list[dict] = list(resume["transcript"]) if resume else []
    # history is replayed to the model, so strip any persisted "reasoning" key.
    history: list[dict] = [{"role": m["role"], "content": m["content"]} for m in transcript]
    user_usage = Usage.from_json(resume["_usage_user"]) if resume else Usage()
    target_usage = Usage.from_json(resume["_usage_target"]) if resume else Usage()
    censorship_events: list[dict] = list(resume.get("censorship_events", [])) if resume else []
    start_turn = len(transcript) // 2 + 1

    for turn in range(start_turn, turns + 1):
        user_msg, u, user_reasoning, user_event = _call_user_with_fallback(
            user_cfg, user_model_fallbacks or [],
            [{"role": "system", "content": user_system}]
            + history
            + [{"role": "user", "content": _turn_prompt(turn, turns, landmarks)}],
            scenario["id"], turn, perfunctory, verbose,
        )
        if user_event:
            censorship_events.append(user_event)
        user_usage = user_usage + u
        if user_msg is None:
            if verbose:
                print(f"  [{scenario['id']}] turn {turn}: user terminated", flush=True)
            break

        history.append({"role": "user", "content": user_msg})
        user_entry = {"role": "user", "content": user_msg}
        if user_reasoning:
            user_entry["reasoning"] = user_reasoning
        transcript.append(user_entry)
        if verbose:
            print(f"  [{scenario['id']}] turn {turn} user: {user_msg[:120]!r}", flush=True)

        target_messages = ([{"role": "system", "content": target_system}] if target_system else []) + history
        try:
            target_msg, u, target_reasoning = call(target, target_messages)
        except (ContentFilterError, EmptyStopError, SafetyRejectionError) as exc:
            # No fallback for the target — it's the thing under test, so a
            # block here IS the result being measured, not a bug to route
            # around. Ends the conversation here (like a normal user
            # "terminate conversation"): regenerating and retrying would
            # keep rerolling the user's phrasing until something slips past
            # the filter, which discards the very outcome being measured.
            if isinstance(exc, ContentFilterError):
                reason = "content_filter"
            elif isinstance(exc, EmptyStopError):
                reason = "empty_stop"
            else:
                reason = "safety_rejection"
            censorship_events.append({"turn": turn, "role": "target", "reason": reason,
                                        "context": target_messages[-1]["content"][:300]})
            if verbose:
                print(f"  [{scenario['id']}] turn {turn}: target {reason}, ending conversation", flush=True)
            break
        target_usage = target_usage + u
        history.append({"role": "assistant", "content": target_msg})
        target_entry = {"role": "assistant", "content": target_msg}
        if target_reasoning:
            target_entry["reasoning"] = target_reasoning
        transcript.append(target_entry)
        if verbose:
            print(f"  [{scenario['id']}] turn {turn} target: {target_msg[:120]!r}", flush=True)

        if on_turn:
            on_turn(transcript, user_usage, target_usage, censorship_events)

    return transcript, user_usage, target_usage, censorship_events
