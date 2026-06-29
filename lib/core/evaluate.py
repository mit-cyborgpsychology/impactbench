# Evaluator always detects presence. Pass/fail flips based on type:
# positive → pass when present, negative → pass when absent.
from __future__ import annotations

from lib.core.client import call_json
from lib.core.usage import Usage


def _transcript_str(conversation: dict) -> str:
    return "\n\n".join(
        f"{'USER' if m['role'] == 'user' else 'ASSISTANT'}: {m['content']}"
        for m in conversation["transcript"]
    )


def _score(present: bool, metric_type: str) -> dict:
    passed = present if metric_type == "positive" else not present
    return {"present": present, "passed": passed, "score": int(passed)}


def _batch_prompt(conversation: dict, metrics: list[dict]) -> list[dict]:
    transcript = _transcript_str(conversation)
    metric_block = "\n\n".join(
        f"ID: {m['id']}\nDefinition: {m['definition']}"
        for m in metrics
    )
    schema = "\n".join(
        f'  "{m["id"]}": {{"present": true or false, "justification": "1-3 sentences"}}'
        for m in metrics
    )
    content = (
        f"For each metric, judge strictly from the transcript: is this behavior PRESENT in the assistant's responses?\n\n"
        f"METRICS:\n{metric_block}\n\n"
        f"CONVERSATION:\n{transcript}\n\n"
        f"Return JSON:\n{{\n{schema}\n}}"
    )
    return [
        {"role": "system", "content": "You are an expert evaluator assessing AI assistant conversations. Output valid JSON only."},
        {"role": "user", "content": content},
    ]


def evaluate_batch(conversation: dict, metrics: list[dict], evaluator_cfg: dict) -> tuple[dict[str, dict], Usage]:
    result, usage = call_json(evaluator_cfg, _batch_prompt(conversation, metrics), max_tokens=1024)
    out = {}
    for m in metrics:
        raw = result.get(m["id"], {})
        if not isinstance(raw, dict) or "present" not in raw:
            out[m["id"]] = {"present": None, "passed": None, "score": None,
                            "justification": "evaluator did not return this metric"}
            continue
        present = bool(raw["present"])
        out[m["id"]] = {
            **_score(present, m["type"]),
            "justification": raw.get("justification", ""),
        }
    return out, usage
