# Immutable value. Cost flows by returning Usage from each call and summing —
# never mutated in shared state, so resume reports the same total as a fresh run.
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Usage:
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.cost + other.cost,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )

    def to_json(self) -> dict:
        return {
            "cost":          round(self.cost, 6),
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    @classmethod
    def from_json(cls, d: dict | None) -> "Usage":
        d = d or {}
        return cls(
            float(d.get("cost", 0.0)),
            int(d.get("input_tokens", 0)),
            int(d.get("output_tokens", 0)),
        )


def total(usages) -> Usage:
    acc = Usage()
    for u in usages:
        acc = acc + (u if isinstance(u, Usage) else Usage.from_json(u))
    return acc
