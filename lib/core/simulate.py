from __future__ import annotations
import re
import random
from typing import Callable

from lib.core.client import call
from lib.core.prompts import load
from lib.core.usage import Usage

TERMINATE = "terminate conversation"


_WORD_RE = re.compile(r"[A-Za-z]+")
_MID_PERIOD_RE = re.compile(r"(?<!\.)\. (?=[a-z])")
_CONJ_COMMA_RE = re.compile(r", (but|and|so|because|which)\b")
_BANG_RE = re.compile(r"(?<!\!)!(?!\!)")
_QMARK_RE = re.compile(r"(?<!\?)\?(?!\?)")


def _perfunctory(text: str) -> str:
    text = _mangle_punctuation(text)
    chars = list(text)
    n = len(chars)
    protected: set[int] = set()
    for m in _WORD_RE.finditer(text):
        word = m.group()
        if len(word) > 1 and word.isupper():
            protected.update(range(max(m.start() - 1, 0), min(m.end() + 1, n)))
            continue
        i = m.start()
        if chars[i].isupper() and random.random() < 0.08:
            chars[i] = chars[i].lower()
    if n > 1 and random.random() < 0.15:
        i = _random_index(n - 1, exclude=lambda j: j in protected or j + 1 in protected)
        if i is not None:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    drops = set()
    if n > 0 and random.random() < 0.1:
        i = _random_index(n, exclude=lambda j: j in protected)
        if i is not None:
            drops = {i}
    return "".join(c for i, c in enumerate(chars) if i not in drops)


def _random_index(n: int, exclude=None, max_attempts: int = 10) -> int | None:
    for _ in range(max_attempts):
        i = random.randint(0, n - 1)
        if exclude is None or not exclude(i):
            return i
    return None


def _mangle_punctuation(text: str) -> str:
    text = _MID_PERIOD_RE.sub(lambda m: random.choice(["... ", ". ", " "]) if random.random() < 0.1 else m.group(), text)
    text = _CONJ_COMMA_RE.sub(lambda m: f" {m.group(1)}" if random.random() < 0.1 else m.group(), text)
    text = _BANG_RE.sub(lambda m: "!!" if random.random() < 0.05 else "!", text)
    text = _QMARK_RE.sub(lambda m: random.choice(["??", "?!"]) if random.random() < 0.05 else "?", text)
    if text.endswith(".") and not text.endswith("..."):
        if random.random() < 0.15:
            text = text[:-1]
    return text


def _parse_message(raw: str, perfunctory: bool) -> str | None:
    m = re.search(r"(?i)^message:\s*(.+)", raw, re.MULTILINE | re.DOTALL)
    content = m.group(1).strip() if m else raw.strip()
    if content.lower().startswith(TERMINATE):
        return None
    return _perfunctory(content) if perfunctory else content


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
        persona=scenario.get("persona", scenario.get("user_persona", "")),
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


def simulate(
    scenario: dict,
    target: dict,
    goal: dict,
    cfg: dict,
    user_cfg: dict,
    metric: dict | None = None,
    perfunctory: bool = False,
    landmarks: bool = True,
    verbose: bool = False,
    resume: dict | None = None,
    on_turn: Callable[[list[dict], Usage, Usage], None] | None = None,
) -> tuple[list[dict], Usage, Usage]:
    turns = cfg["generation"]["turns"]
    target_system = (
        scenario.get("target_system_prompt")
        or goal.get("scenario", {}).get("user_context")
        or load("default_target_system.txt")
    )
    user_system = _user_system(scenario, turns, metric, perfunctory, landmarks)

    transcript: list[dict] = list(resume["transcript"]) if resume else []
    history: list[dict] = list(transcript)
    user_usage = Usage.from_json(resume["_usage_user"]) if resume else Usage()
    target_usage = Usage.from_json(resume["_usage_target"]) if resume else Usage()
    start_turn = len(transcript) // 2 + 1

    for turn in range(start_turn, turns + 1):
        raw, u = call(
            user_cfg,
            [{"role": "system", "content": user_system}]
            + history
            + [{"role": "user", "content": _turn_prompt(turn, turns, landmarks)}],
        )
        user_usage = user_usage + u
        user_msg = _parse_message(raw, perfunctory)
        if user_msg is None:
            if verbose:
                print(f"  [{scenario['id']}] turn {turn}: user terminated", flush=True)
            break

        history.append({"role": "user", "content": user_msg})
        transcript.append({"role": "user", "content": user_msg})
        if verbose:
            print(f"  [{scenario['id']}] turn {turn} user: {user_msg[:120]!r}", flush=True)

        target_msg, u = call(target, [{"role": "system", "content": target_system}] + history)
        target_usage = target_usage + u
        history.append({"role": "assistant", "content": target_msg})
        transcript.append({"role": "assistant", "content": target_msg})
        if verbose:
            print(f"  [{scenario['id']}] turn {turn} target: {target_msg[:120]!r}", flush=True)

        if on_turn:
            on_turn(transcript, user_usage, target_usage)

    return transcript, user_usage, target_usage
