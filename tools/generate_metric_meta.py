#!/usr/bin/env python3
"""Fill in each metric's "This matters because" sentence, in benchmark.yaml.

Writes ``mattersBecause`` directly onto the metric it belongs to::

    metrics:
      - id: m01
        name: Crisis resource provision
        type: positive
        definition: Provides at least one specific crisis intervention resource...
        mattersBecause: This matters because when someone is in mental health crisis...

publish.py reads it from there into the published ``metric-details.json`` (the
`contributor` shown alongside it is the benchmark's own ``name`` — one value per
benchmark, so it isn't stored per metric).

This is NOT run by the publish workflow: the sentences come from an LLM and are
meant to be read before they ship. Run it when metrics are added or reworded,
review the diff, commit the benchmark.yaml changes.

Idempotent — a metric that already has ``mattersBecause`` is skipped unless
``--force``.

Usage (from bench-py/)::

    ANTHROPIC_API_KEY=sk-ant-... python tools/generate_metric_meta.py
    python tools/generate_metric_meta.py --force              # regenerate everything
    python tools/generate_metric_meta.py --only humanebench   # one benchmark
    python tools/generate_metric_meta.py --dry-run            # no API calls

Environment variables:
    ANTHROPIC_API_KEY  required unless --dry-run
    ANTHROPIC_MODEL    default: claude-sonnet-4-5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARKS_DIR = ROOT / "benchmarks"

SYSTEM_PROMPT = """You write a single short sentence explaining why an AI safety / wellbeing metric matters to everyday users. Use plain, concrete language. No hedging, no marketing fluff, no preamble."""

USER_TEMPLATE = """Benchmark: {benchmark_name}
Contributor (the org / project that authored this metric): {contributor}
Metric name: {metric_name}
Metric type: {metric_type}  (positive = behavior we want AI to do; negative = behavior we want AI to avoid)
Behavior measured: {definition}

Write exactly one short sentence (max ~28 words) starting with the words "This matters because" that explains the real-world stakes for the user.

Rules:
- For POSITIVE metrics, frame it around what users gain when AI does this well (e.g. "This matters because it is important for AI to encourage ... so that humans ...").
- For NEGATIVE metrics, frame it around what users avoid when AI restrains this behavior (e.g. "This matters because it is crucial that AI does not mislead ... which can lead to ...").
- Anchor the sentence in the specific behavior described above; don't be generic.
- Do not mention the benchmark name, the contributor, "this metric", or the word "AI" more than once.
- Return only the sentence — no quotes, no explanation."""


def slug_from_name(name: str) -> str:
    return (
        name.lower()
        .replace("&", "and")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
        .replace(",", "")
        .replace(":", "")
        .replace(".", "")
        .replace("'", "")
        .replace("\u2019", "")
        .replace(" ", "-")
    )


# nutritional-label is a curated re-selection of metrics that already exist in
# other benchmarks; publish.py resolves its metrics to their source, so they never
# need their own prose. e2e-test/mcab are scaffolding.
SKIP_BENCHMARKS = {"e2e-test", "mcab", "nutritional-label"}


def discover_benchmarks(data_repo: Path) -> list[Path]:
    return sorted(
        p for p in data_repo.glob("*/benchmark.yaml")
        if p.is_file() and p.parent.name not in SKIP_BENCHMARKS
    )


def call_anthropic(client: Any, model: str, prompt: str, retries: int = 3) -> str:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    return block.text.strip().strip('"').strip()
            raise RuntimeError("no text block in response")
        except Exception as exc:  # noqa: BLE001 - want to retry on anything transient
            last_err = exc
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{retries} after {wait}s: {exc}", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"anthropic call failed after {retries} attempts: {last_err}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="regenerate even if entry already exists")
    parser.add_argument("--only", help="only process this benchmark folder name", default=None)
    parser.add_argument("--dry-run", action="store_true", help="skip API calls; just refresh contributor field")
    parser.add_argument("--benchmarks-dir", default=str(DEFAULT_BENCHMARKS_DIR),
                        help="path to the impactbench-data checkout (default: ../benchmarks)")
    parser.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"))
    args = parser.parse_args()

    data_repo = Path(args.benchmarks_dir).resolve()
    if not data_repo.exists():
        print(f"error: benchmarks dir not found at {data_repo}", file=sys.stderr)
        return 1

    benchmarks = discover_benchmarks(data_repo)
    if args.only:
        benchmarks = [b for b in benchmarks if b.parent.name == args.only]
        if not benchmarks:
            print(f"error: --only {args.only!r} matched no benchmark", file=sys.stderr)
            return 1

    client = None
    if not args.dry_run:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("error: ANTHROPIC_API_KEY not set (or pass --dry-run)", file=sys.stderr)
            return 1
        import anthropic  # local import so --dry-run works without the lib

        client = anthropic.Anthropic()

    total = 0
    generated = 0
    for yaml_path in benchmarks:
        benchmark_slug = yaml_path.parent.name
        doc = yaml.safe_load(yaml_path.read_text())
        benchmark_name = doc.get("name", benchmark_slug)
        metrics = doc.get("metrics") or []
        dirty = False

        for metric in metrics:
            mid = metric.get("id")
            if not mid:
                continue
            total += 1
            if not args.force and metric.get("mattersBecause"):
                continue
            if args.dry_run or client is None:
                continue

            prompt = USER_TEMPLATE.format(
                benchmark_name=benchmark_name,
                contributor=metric.get("contributor") or benchmark_name,
                metric_name=metric.get("name", mid),
                metric_type=metric.get("type", "positive"),
                definition=(metric.get("definition") or "").strip(),
            )
            print(f"  generating {benchmark_slug}__{mid}: {metric.get('name', mid)!r}")
            metric["mattersBecause"] = call_anthropic(client, args.model, prompt)
            generated += 1
            dirty = True
            # Persist after each call so a crash doesn't lose work.
            yaml_path.write_text(yaml.dump(doc, sort_keys=False, width=120, allow_unicode=True))

        if dirty:
            yaml_path.write_text(yaml.dump(doc, sort_keys=False, width=120, allow_unicode=True))

    missing = 0
    for yaml_path in benchmarks:
        doc = yaml.safe_load(yaml_path.read_text())
        missing += sum(1 for m in (doc.get("metrics") or []) if not m.get("mattersBecause"))

    print(f"done. {generated} generated, {total} metrics, {missing} still without mattersBecause")
    if args.dry_run and missing:
        print("(dry run — rerun without --dry-run to generate them)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
