"""
Build the impactbench website's data from impactbench-data and upload it to R2.

Outputs (written to tools/_out/, then uploaded under data/):
  taxonomy.json          areas > subareas > metrics; ids from taxonomy.json,
                         names/types hydrated from benchmark.yaml
  models.json            model metadata + surfaces, from models.yaml
  benchmark-data.json    { "modelId|age": { "benchmark__metricId": score } }
  scenario-index.json    per metric: its scenarios and each model's verdict
  metric-details.json    per metric: { definition, contributor, mattersBecause }
  nutrition.json         nutrition-label categories with per-model/per-age scores

Per-scenario detail files, uploaded under scenarios/:
  scenarios/{benchmark}/{modelId}/{scenarioId}.json

Reads from --benchmarks-dir (defaults to ../benchmarks, i.e. the impactbench-data
checkout): taxonomy.json, nutrition-label.json, models.yaml, and each
<bench>/{benchmark.yaml, scenarios.json, runs/<model>/{scores,conversations}.json}.

Uploads are content-hash compared against R2's ETag and skipped when unchanged,
so re-running with unchanged data is mostly no-ops.

Run from bench-py/:
    python tools/publish.py --dry-run                # build only, no upload
    python tools/publish.py                          # build + upload
    python tools/publish.py --only benchmark-data    # one output only
    python tools/publish.py --emit-scenarios         # also write scenarios locally
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

TOOLS_DIR = Path(__file__).parent
ROOT = TOOLS_DIR.parent
DEFAULT_BENCHMARKS_DIR = ROOT / "benchmarks"
WEB_DATA = ROOT / "tools" / "_out"  # local build output

R2_BUCKET = os.environ.get("R2_BUCKET", "impactbench-scenarios")

# Folders under benchmarks-dir that aren't real benchmarks (test scaffolding,
# stale duplicates) — matches the exclusion the old impactbench build script had.
SKIP_BENCHMARKS = {"e2e-test", "mcab"}

# Populated from <benchmarks-dir>/models.yaml at startup (see load_models).
#   MODEL_ID_MAP:   run-dir name -> public model id
#   MODEL_METADATA: run-dir name -> {id, name, provider, releaseYear}
MODEL_ID_MAP: dict[str, str] = {}
MODEL_METADATA: dict[str, dict] = {}


def load_models(models_path: Path) -> None:
    """Load the model registry into MODEL_ID_MAP / MODEL_METADATA.

    models.yaml lives with the data (impactbench-data) and is the single source
    of truth for model identity + display metadata. bench-py/config.yaml stays
    responsible only for *how to call* each model.
    """
    if not models_path.exists():
        fail(f"models.yaml not found at {models_path}")

    doc = yaml.safe_load(models_path.read_text()) or {}
    entries = doc.get("models") or []
    if not entries:
        fail(f"no models defined in {models_path}")

    for m in entries:
        missing = [k for k in ("run_dir", "id", "name", "provider", "releaseYear", "surfaces") if k not in m]
        if missing:
            fail(f"models.yaml entry {m.get('id') or m.get('run_dir') or '?'} is missing: {', '.join(missing)}")
        MODEL_ID_MAP[m["run_dir"]] = m["id"]
        MODEL_METADATA[m["run_dir"]] = {
            "id": m["id"],
            "name": m["name"],
            "provider": m["provider"],
            "releaseYear": m["releaseYear"],
            "surfaces": m["surfaces"],
        }


# ── helpers ────────────────────────────────────────────────────────────────────

def iter_benchmarks(benchmarks_dir: Path):
    for d in sorted(benchmarks_dir.iterdir()):
        if d.name in SKIP_BENCHMARKS:
            continue
        yp = d / "benchmark.yaml"
        if yp.exists():
            yield d

def load_spec(bench_dir: Path) -> dict:
    return yaml.safe_load((bench_dir / "benchmark.yaml").read_text())

def score_value(row: dict) -> float:
    """Return the [0,1] score, oriented so 1 = good.

    `passed` is already polarity-adjusted by the pipeline (positive metrics pass
    when present, negative when absent), so it maps straight to 1.0 / 0.0. Do NOT
    re-invert for negative metrics — the old web_export.py did, and double-inverted
    every negative metric.
    """
    return 1.0 if row["passed"] else 0.0

def load_metric_aliases(benchmarks_dir: Path) -> dict[str, str]:
    """Map curated metric keys to the source metric they were drawn from.

    nutritional-label is a curated selection of metrics that already exist in
    other benchmarks; each of its metrics carries a `source: {benchmark,
    metric_id}` block. Its scores belong under the source metric's key, not
    under nutritional-label__mNN (which isn't in the taxonomy at all). Without
    this, models only ever run on nutritional-label (apertus-*) would produce
    no benchmark-data rows.
    """
    aliases: dict[str, str] = {}
    spec_path = benchmarks_dir / "nutritional-label" / "benchmark.yaml"
    if not spec_path.exists():
        return aliases
    spec = yaml.safe_load(spec_path.read_text())
    for m in spec.get("metrics", []):
        src = m.get("source") or {}
        if src.get("benchmark") and src.get("metric_id"):
            aliases[f"nutritional-label__{m['id']}"] = f"{src['benchmark']}__{src['metric_id']}"
    return aliases


def check_curated_drift(benchmarks_dir: Path) -> list[str]:
    """Report curated metrics whose copied content diverged from their source.

    nutritional-label/benchmark.yaml duplicates name/type/definition/examples
    from the metric it points at, because the pipeline needs them to run the
    eval. Published output always uses the *source* metric's content (see
    load_metric_aliases), so a divergence here never reaches the site — but it
    means two files disagree about the same metric, which is how the names
    silently drifted apart in the first place.
    """
    spec_path = benchmarks_dir / "nutritional-label" / "benchmark.yaml"
    if not spec_path.exists():
        return []

    specs: dict[str, dict] = {}
    def source_metric(bench: str, mid: str) -> dict | None:
        if bench not in specs:
            p = benchmarks_dir / bench / "benchmark.yaml"
            if not p.exists():
                specs[bench] = {}
            else:
                specs[bench] = {m["id"]: m for m in (yaml.safe_load(p.read_text()).get("metrics") or [])}
        return specs[bench].get(mid)

    drift = []
    for m in yaml.safe_load(spec_path.read_text()).get("metrics") or []:
        src = m.get("source") or {}
        if not src.get("benchmark") or not src.get("metric_id"):
            continue
        origin = source_metric(src["benchmark"], src["metric_id"])
        if origin is None:
            continue  # dangling ref: already reported by validate_nutrition_label
        for field in ("name", "type", "definition", "examples"):
            if m.get(field) != origin.get(field):
                drift.append(
                    f"nutritional-label__{m['id']}.{field} != "
                    f"{src['benchmark']}__{src['metric_id']}.{field}"
                )
    return drift


def load_metric_registry(benchmarks_dir: Path) -> dict[str, dict]:
    """benchmark__metricId -> {name, type, definition, contributor, mattersBecause}.

    benchmark.yaml is the single source of truth for metric content. taxonomy.json
    stores only structure (which metric ids sit under which area/subarea); names
    and types are hydrated from here at build time so editing a metric in one
    place can't leave the site showing a stale name.

    `contributor` is read from the metric when present, else falls back to the
    benchmark's own name. `mattersBecause` is written by generate_metric_meta.py
    directly onto the metric.
    """
    registry: dict[str, dict] = {}
    for bench_dir in iter_benchmarks(benchmarks_dir):
        spec = load_spec(bench_dir)
        default_contributor = spec.get("name", bench_dir.name)
        for m in spec.get("metrics") or []:
            registry[f"{bench_dir.name}__{m['id']}"] = {
                "name": m.get("name", ""),
                "type": m.get("type", ""),
                "definition": m.get("definition", ""),
                "contributor": m.get("contributor") or default_contributor,
                "mattersBecause": m.get("mattersBecause", ""),
            }
    return registry


def hydrate_taxonomy(taxonomy_path: Path, registry: dict[str, dict]) -> tuple[dict, set[str], list[str]]:
    """Expand the stored taxonomy into the shape the site consumes.

    On disk, a subarea stores only its `groups` (named clusters of metric ids) —
    that's the editorial structure, and the flat `metrics` list the site reads is
    just those ids in order. So build `metrics` here and hydrate each entry's
    name/type from benchmark.yaml.

    Returns (hydrated_taxonomy, valid_ids, dangling_ids). A taxonomy id that no
    benchmark.yaml defines is dangling — a hard error, since the taxonomy would
    be pointing at a metric that no longer exists.
    """
    tax = json.loads(taxonomy_path.read_text())
    valid_ids: set[str] = set()
    dangling: list[str] = []

    for area in tax.get("areas", []):
        for sub in area.get("subareas", []):
            metrics = []
            seen: set[str] = set()
            for group in sub.get("groups") or []:
                for mid in group.get("metric_ids", []):
                    if mid in seen:
                        continue
                    seen.add(mid)
                    valid_ids.add(mid)
                    entry = registry.get(mid)
                    if entry is None:
                        dangling.append(mid)
                        continue
                    metrics.append({"id": mid, "name": entry["name"], "type": entry["type"]})
            sub["metrics"] = metrics

    return tax, valid_ids, dangling


def validate_nutrition_label(label_path: Path, registry: dict[str, dict]) -> list[str]:
    """Return nutrition-label metric refs that no benchmark.yaml defines."""
    if not label_path.exists():
        return []
    dangling = []
    for cat in json.loads(label_path.read_text()):
        for key in cat.get("metrics", []):
            if key not in registry:
                dangling.append(f"{cat['id']} -> {key}")
    return dangling


# ── builders ───────────────────────────────────────────────────────────────────

def find_unregistered_models(benchmarks_dir: Path) -> set[str]:
    """Run-dir names present in the data but absent from models.yaml.

    These get silently dropped from every output (this is how apertus-* went
    missing before), so surface them rather than letting them vanish.
    """
    unknown: set[str] = set()
    for bench_dir in iter_benchmarks(benchmarks_dir):
        runs_dir = bench_dir / "runs"
        if not runs_dir.exists():
            continue
        for model_dir in runs_dir.iterdir():
            if model_dir.is_dir() and model_dir.name not in MODEL_ID_MAP:
                unknown.add(model_dir.name)
    return unknown


def build_benchmark_data(benchmarks_dir: Path, valid_ids: set[str] | None,
                         aliases: dict[str, str]) -> tuple[dict, list, set[str]]:
    """Returns (benchmark_data, models_list, unmatched_metric_keys)."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    unmatched: set[str] = set()

    for bench_dir in iter_benchmarks(benchmarks_dir):
        benchmark = bench_dir.name
        runs_dir = bench_dir / "runs"
        if not runs_dir.exists():
            continue
        for model_dir in sorted(runs_dir.iterdir()):
            meta = MODEL_METADATA.get(model_dir.name)
            if not meta:
                continue
            sc_path = model_dir / "scores.json"
            if not sc_path.exists():
                continue
            for row in json.loads(sc_path.read_text()):
                metric_key = f"{benchmark}__{row['metric_id']}"
                metric_key = aliases.get(metric_key, metric_key)
                if valid_ids is not None and metric_key not in valid_ids:
                    unmatched.add(metric_key)
                    continue
                age = "child" if row["id"].endswith("_v01") else "adult"
                key = f"{meta['id']}|{age}"
                acc[key][metric_key].append(score_value(row))

    output = {
        key: {mk: round(sum(v) / len(v), 4) for mk, v in sorted(metrics.items())}
        for key, metrics in sorted(acc.items())
    }

    # Every model in models.yaml is published, with the surfaces it declares there.
    # A curated-only model (e.g. apertus, surfaces: [nutritional-label]) has no
    # rows in benchmark-data for the full taxonomy, but still appears in models.json
    # so the Nutrition Label page can show it.
    models = [
        {"id": m["id"], "name": m["name"], "provider": m["provider"],
         "version": m["id"], "releaseYear": m["releaseYear"],
         "surfaces": m["surfaces"]}
        for m in MODEL_METADATA.values()
    ]
    return output, models, unmatched


def build_scenario_index_and_details(benchmarks_dir: Path, valid_ids: set[str] | None,
                                     registry: dict[str, dict], aliases: dict[str, str]) -> tuple[dict, dict, set[str]]:
    """Returns (scenario_index, metric_details, unmatched_metric_keys).

    metric_details is keyed by benchmark__metricId and carries
    {definition, contributor, mattersBecause}, all resolved from the registry
    (benchmark.yaml). scenario_index lists each metric's scenarios with each
    model's verdict.
    """
    index: dict[str, dict[str, dict]] = defaultdict(dict)
    details: dict[str, dict] = {}
    unmatched: set[str] = set()

    # Definitions come straight from the registry (benchmark.yaml, the single
    # source of truth). Metrics defined in a benchmark but absent from the
    # taxonomy are reported as unmatched — a soft warning, since a new metric
    # normally lands before someone updates the taxonomy.
    for metric_key, entry in registry.items():
        if aliases and metric_key in aliases:
            continue  # curated duplicate; its content lives under the source key
        if valid_ids is not None and metric_key not in valid_ids:
            unmatched.add(metric_key)
            continue
        details[metric_key] = {
            "definition": entry["definition"],
            "contributor": entry["contributor"],
            "mattersBecause": entry["mattersBecause"],
        }

    for bench_dir in iter_benchmarks(benchmarks_dir):
        benchmark = bench_dir.name

        sc_path = bench_dir / "scenarios.json"
        if not sc_path.exists():
            continue
        sc_meta = {}
        for sc in json.loads(sc_path.read_text()):
            age = "child" if sc["id"].endswith("_v01") else "adult"
            sc_meta[sc["id"]] = {
                "title": (sc.get("user_goal") or sc.get("persona") or sc["id"])[:80],
                "age": age,
            }

        runs_dir = bench_dir / "runs"
        if not runs_dir.exists():
            continue
        verdicts: dict[str, dict[str, str]] = defaultdict(dict)
        for model_dir in sorted(runs_dir.iterdir()):
            model_id = MODEL_ID_MAP.get(model_dir.name)
            if not model_id:
                continue
            sp = model_dir / "scores.json"
            if not sp.exists():
                continue
            for row in json.loads(sp.read_text()):
                verdicts[row["id"]][model_id] = "yes" if row["passed"] else "no"

        for sc_id, meta in sc_meta.items():
            metric_id = sc_id.split("_")[0]
            metric_key = f"{benchmark}__{metric_id}"
            # NOTE: no aliasing here, unlike benchmark-data/nutrition. A curated
            # nutritional-label scenario is a *different* scenario (different
            # prompt) that happens to target the same behavior as its source
            # metric — folding it into the source key would list it twice in the
            # scenario browser. Its scores still count via the nutrition path;
            # only the redundant scenario listing is dropped.
            if aliases and metric_key in aliases:
                continue  # curated re-run; its source metric owns the scenario listing
            if valid_ids is not None and metric_key not in valid_ids:
                unmatched.add(metric_key)
                continue
            index[metric_key][sc_id] = {
                "scenario_id": sc_id,
                "title": meta["title"],
                "age": meta["age"],
                "benchmark": benchmark,
                "verdicts": verdicts.get(sc_id, {}),
            }

    return {k: list(v.values()) for k, v in sorted(index.items())}, details, unmatched


def build_nutrition(label_path: Path, benchmarks_dir: Path, aliases: dict[str, str],
                    registry: dict[str, dict]) -> list:
    """Returns the nutrition-label categories, each with:

        {
          "id": "...", "label": "...", "description": "...",
          "models":  {model_id: {"child": x, "adult": y}},  # category score, per model/age
          "metrics": [{"id", "metric_id", "benchmark", "name", "type",
                       "label_metric_id"?}]                  # the metrics in the category
        }

    The category `models` scores are averaged from the metrics' scores. No
    per-metric score is emitted: the site recomputes each metric's score from
    benchmark-data.json for the selected model+age (see NutritionCatPanel).
    `label_metric_id` is present only on metrics the nutritional-label benchmark
    curated, mapping the source metric to the id it has within that benchmark.
    """
    if not label_path.exists():
        print(f"  [warn] nutrition label not found at {label_path}, skipping")
        return []

    categories = json.loads(label_path.read_text())
    # metric key ("benchmark__metricId") -> categories it belongs to
    metric_to_cats: dict[str, list[str]] = defaultdict(list)
    for cat in categories:
        for key in cat["metrics"]:
            metric_to_cats[key].append(cat["id"])

    # source metric key -> the id it has inside the nutritional-label benchmark
    # (the reverse of `aliases`), so /viewer can map an imported run's m01..m83
    # scores onto the source metrics shown here.
    curated_id_of = {
        source: curated.split("__", 1)[1]
        for curated, source in aliases.items()
    }

    # raw[cat_id][model_id][age][metric_key] -> [scores]
    raw: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for bench_dir in iter_benchmarks(benchmarks_dir):
        benchmark = bench_dir.name

        runs_dir = bench_dir / "runs"
        if not runs_dir.exists():
            continue
        for model_dir in sorted(runs_dir.iterdir()):
            model_id = MODEL_ID_MAP.get(model_dir.name)
            if not model_id:
                continue
            sp = model_dir / "scores.json"
            if not sp.exists():
                continue
            for row in json.loads(sp.read_text()):
                scenario_metric = row["id"].split("_")[0]
                if scenario_metric != row["metric_id"]:
                    continue
                # Curated nutritional-label metrics resolve to their source metric,
                # so e.g. apertus (which only runs nutritional-label) still lands
                # under the real benchmark__metric keys the categories reference.
                metric_key = f"{benchmark}__{row['metric_id']}"
                metric_key = aliases.get(metric_key, metric_key)
                age = "child" if row["id"].endswith("_v01") else "adult"
                val = score_value(row)
                for cat_id in metric_to_cats.get(metric_key, []):
                    raw[cat_id][model_id][age][metric_key].append(val)

    # average per metric per age, then per category per age
    cat_scores: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for cat_id, model_map in raw.items():
        for model_id, age_map in model_map.items():
            for age, metric_map in age_map.items():
                for _, vals in metric_map.items():
                    cat_scores[cat_id][model_id][age].append(sum(vals) / len(vals))

    nutrition = []
    for cat in categories:
        cid = cat["id"]
        models_out = {}
        for model_id, age_map in sorted(cat_scores[cid].items()):
            entry = {}
            for age, vals in age_map.items():
                if vals:
                    entry[age] = round(sum(vals) / len(vals), 4)
            if entry:
                models_out[model_id] = entry

        # No per-metric score here: the site recomputes each metric's score from
        # benchmark-data.json for the selected model+age (see NutritionCatPanel),
        # so a baked-in average would be dead weight.
        cat_metrics = []
        for key in cat["metrics"]:
            benchmark_slug, _, metric_id = key.partition("__")
            info = registry.get(key, {})
            entry = {
                "id":        key,
                "metric_id": metric_id,
                "benchmark": benchmark_slug,
                "name":      info.get("name", ""),
                "type":      info.get("type", ""),
            }
            # The id this metric has *within* the nutritional-label benchmark, if
            # any. /viewer imports a run of that benchmark, whose scores are keyed
            # m01..m83, and needs to map them onto the source metric shown here.
            curated = curated_id_of.get(key)
            if curated:
                entry["label_metric_id"] = curated
            cat_metrics.append(entry)

        nutrition.append({
            "id":          cid,
            "label":       cat["label"],
            "description": cat.get("description", ""),
            "models":      models_out,
            "metrics":     cat_metrics,
        })

    return nutrition


# Scenario fields the site's detail view needs, carried over from scenarios.json.
_SCENARIO_FIELDS = (
    "persona", "user_persona", "user_goal", "latent_adversarial_goal",
    "landmarks", "demographic",
)


def build_scenario_uploads(benchmarks_dir: Path) -> list[tuple[str, str]]:
    """Compose each published scenario-detail file from its sources.

    A run artifact stores only what its stage produced, so the payload the site's
    conversation view fetches is assembled here rather than carried in duplicate:

        conversations.json  transcript, conv_id, target
        scenarios.json      persona, user_goal, landmarks, ... (joined on id)
        scores.json         justification                      (joined on id)
    """
    uploads = []
    for bench_dir in iter_benchmarks(benchmarks_dir):
        benchmark = bench_dir.name
        runs_dir = bench_dir / "runs"
        if not runs_dir.exists():
            continue

        scenarios: dict[str, dict] = {}
        sc_path = bench_dir / "scenarios.json"
        if sc_path.exists():
            scenarios = {s["id"]: s for s in json.loads(sc_path.read_text())}

        for model_dir in sorted(runs_dir.iterdir()):
            model_id = MODEL_ID_MAP.get(model_dir.name)
            if not model_id:
                continue
            conv_path = model_dir / "conversations.json"
            if not conv_path.exists():
                continue

            justifications = {}
            sp = model_dir / "scores.json"
            if sp.exists():
                for row in json.loads(sp.read_text()):
                    if row.get("justification"):
                        justifications[row["id"]] = row["justification"]

            for conv in json.loads(conv_path.read_text()):
                sid = conv["id"]
                payload = dict(conv)

                scenario = scenarios.get(sid, {})
                for f in _SCENARIO_FIELDS:
                    if f in scenario:
                        payload[f] = scenario[f]

                if sid in justifications:
                    payload["justification"] = justifications[sid]

                key = f"scenarios/{benchmark}/{model_id}/{sid}.json"
                uploads.append((key, json.dumps(payload, separators=(",", ":"))))
    return uploads


# ── upload ─────────────────────────────────────────────────────────────────────

def get_s3():
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        print("boto3 not installed — run: pip install boto3 (or pip install -e '.[publish]')")
        sys.exit(1)

    endpoint = os.environ.get("R2_ENDPOINT")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    missing = [n for n, v in [("R2_ENDPOINT", endpoint), ("R2_ACCESS_KEY_ID", access_key), ("R2_SECRET_ACCESS_KEY", secret_key)] if not v]
    if missing:
        print(f"error: missing required env var(s): {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
    )


def remote_etags(s3, prefix: str) -> dict[str, str]:
    """{key: md5} for every object under `prefix`, via one paginated LIST.

    R2/S3 ETag is the MD5 hex digest for non-multipart uploads (true for all
    objects here — small JSON files), so listing the bucket once lets us decide
    what changed in memory instead of a HEAD per file. ~1 call per 1000 keys
    versus one round-trip per object.
    """
    etags: dict[str, str] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            etags[obj["Key"]] = obj["ETag"].strip('"')
    return etags


def _put(s3, key: str, body: bytes, content_type: str = "application/json"):
    s3.put_object(Bucket=R2_BUCKET, Key=key, Body=body, ContentType=content_type)


def upload_data_files(s3, files: dict[str, str]):
    remote = remote_etags(s3, "data/")
    uploaded = skipped = 0
    for name, content in files.items():
        key = f"data/{name}"
        body = content.encode()
        if remote.get(key) == hashlib.md5(body).hexdigest():
            print(f"  skipped {key}")
            skipped += 1
            continue
        _put(s3, key, body)
        print(f"  uploaded {key}")
        uploaded += 1
    print(f"  data files: {uploaded} uploaded, {skipped} skipped")


def upload_scenarios(s3, uploads: list[tuple[str, str]], workers: int = 32):
    total = len(uploads)
    # One LIST up front; then only PUT files whose content actually changed.
    remote = remote_etags(s3, "scenarios/")
    changed = [(k, v) for k, v in uploads
               if remote.get(k) != hashlib.md5(v.encode()).hexdigest()]
    skipped = total - len(changed)
    print(f"  {len(remote)} objects already in R2; {len(changed)} changed, {skipped} unchanged")

    uploaded = errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_put, s3, k, v.encode()): k for k, v in changed}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                fut.result()
                uploaded += 1
                if done % 500 == 0:
                    print(f"  {done}/{len(changed)}...")
            except Exception as e:
                errors += 1
                print(f"  ERROR {futures[fut]}: {e}")
    print(f"  scenarios: {uploaded} uploaded, {skipped} skipped, {errors} errors (of {total})")


# ── main ───────────────────────────────────────────────────────────────────────

def warn(msg: str):
    """GitHub Actions annotation + plain stderr line."""
    print(f"::warning::{msg}")
    print(f"  [warn] {msg}", file=sys.stderr)


def fail(msg: str):
    """Hard error: annotate for GitHub Actions and abort before uploading anything."""
    print(f"::error::{msg}")
    print(f"  [error] {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Build only, no upload")
    parser.add_argument("--emit-scenarios", action="store_true",
                         help="Also write scenario files to tools/_out/scenarios/ (for serving a full local copy)")
    parser.add_argument("--only", help="One of: benchmark-data, scenarios, nutrition, scenario-index")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--benchmarks-dir", default=str(DEFAULT_BENCHMARKS_DIR),
                         help="Path to the impactbench-data checkout (contains <bench>/, taxonomy.json, nutrition-label.json)")
    parser.add_argument("--taxonomy", default=None,
                         help="Path to taxonomy.json (default: <benchmarks-dir>/taxonomy.json)")
    parser.add_argument("--nutrition-label", default=None,
                         help="Path to nutrition-label.json (default: <benchmarks-dir>/nutrition-label.json)")
    parser.add_argument("--models", default=None,
                         help="Path to models.yaml (default: <benchmarks-dir>/models.yaml)")
    args = parser.parse_args()

    benchmarks_dir = Path(args.benchmarks_dir).resolve()
    if not benchmarks_dir.is_dir():
        print(f"error: benchmarks dir not found at {benchmarks_dir}", file=sys.stderr)
        sys.exit(1)

    taxonomy_path = Path(args.taxonomy) if args.taxonomy else benchmarks_dir / "taxonomy.json"
    label_path = Path(args.nutrition_label) if args.nutrition_label else benchmarks_dir / "nutrition-label.json"
    models_path = Path(args.models) if args.models else benchmarks_dir / "models.yaml"
    load_models(models_path)
    print(f"Loaded {len(MODEL_METADATA)} models from {models_path.name}")

    # benchmark.yaml is the single source of truth for metric name/type/definition.
    registry = load_metric_registry(benchmarks_dir)
    print(f"Loaded {len(registry)} metrics from benchmark.yaml files")

    aliases = load_metric_aliases(benchmarks_dir)
    if aliases:
        print(f"Loaded {len(aliases)} curated metric aliases (nutritional-label -> source metric)")

    drift = check_curated_drift(benchmarks_dir)
    if drift:
        warn(f"{len(drift)} curated metric field(s) diverge from their source metric. Published data "
             f"always uses the source's version, so resync nutritional-label/benchmark.yaml: "
             f"{'; '.join(drift[:10])}" + (" ..." if len(drift) > 10 else ""))

    unregistered = find_unregistered_models(benchmarks_dir)
    if unregistered:
        warn(f"{len(unregistered)} model(s) have run data but no models.yaml entry — "
             f"their scores are being dropped: {', '.join(sorted(unregistered))}")

    WEB_DATA.mkdir(parents=True, exist_ok=True)
    only = args.only

    data_files: dict[str, str] = {}
    all_unmatched: set[str] = set()

    valid_ids: set[str] | None = None
    if taxonomy_path.exists():
        taxonomy, valid_ids, dangling = hydrate_taxonomy(taxonomy_path, registry)
        if dangling:
            fail(f"{len(dangling)} taxonomy metric(s) not defined in any benchmark.yaml: "
                 f"{', '.join(sorted(dangling)[:20])}" + (" ..." if len(dangling) > 20 else ""))
        print(f"Loaded taxonomy: {len(valid_ids)} metric IDs (names/types hydrated from benchmark.yaml)")
        data_files["taxonomy.json"] = json.dumps(taxonomy, separators=(",", ":"))
    else:
        print(f"  [warn] taxonomy not found at {taxonomy_path} — skipping taxonomy filtering entirely", file=sys.stderr)

    label_dangling = validate_nutrition_label(label_path, registry)
    if label_dangling:
        fail(f"{len(label_dangling)} nutrition-label metric ref(s) not defined in any benchmark.yaml: "
             f"{', '.join(label_dangling[:20])}" + (" ..." if len(label_dangling) > 20 else ""))

    if not only or only == "benchmark-data":
        print("Building benchmark-data + models...")
        bd, models, unmatched = build_benchmark_data(benchmarks_dir, valid_ids, aliases)
        all_unmatched |= unmatched
        data_files["benchmark-data.json"] = json.dumps(bd, separators=(",", ":"))
        data_files["models.json"] = json.dumps({"models": models}, indent=2)
        print(f"  {len(bd)} model|age keys, {len(models)} models")

    if not only or only == "scenario-index":
        print("Building scenario-index + metric-details...")
        si, details, unmatched = build_scenario_index_and_details(benchmarks_dir, valid_ids, registry, aliases)
        all_unmatched |= unmatched
        missing_prose = [k for k, v in details.items() if not v["mattersBecause"]]
        if missing_prose:
            warn(f"{len(missing_prose)} metric(s) have no 'mattersBecause' — run "
                 f"tools/generate_metric_meta.py and commit the benchmark.yaml changes: "
                 f"{', '.join(sorted(missing_prose)[:10])}"
                 + (" ..." if len(missing_prose) > 10 else ""))
        data_files["scenario-index.json"] = json.dumps(si, separators=(",", ":"))
        data_files["metric-details.json"] = json.dumps(details, separators=(",", ":"))
        print(f"  {len(si)} metrics, {sum(len(v) for v in si.values())} scenarios")
        print(f"  {len(details)} metric details")

    if not only or only == "nutrition":
        print("Building nutrition...")
        nutrition = build_nutrition(label_path, benchmarks_dir, aliases, registry)
        if nutrition:
            data_files["nutrition.json"] = json.dumps(nutrition, indent=2)
            print(f"  {len(nutrition)} categories")

    if all_unmatched:
        warn(f"{len(all_unmatched)} metric key(s) not found in taxonomy.json (excluded from output): "
             f"{', '.join(sorted(all_unmatched)[:20])}" + (" ..." if len(all_unmatched) > 20 else ""))
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:
            with open(summary_path, "a") as f:
                f.write(f"\n### Taxonomy mismatches ({len(all_unmatched)})\n\n")
                for key in sorted(all_unmatched):
                    f.write(f"- `{key}`\n")

    # Write all data files locally
    for name, content in data_files.items():
        p = WEB_DATA / name
        p.write_text(content)
        print(f"  wrote {p}  ({p.stat().st_size // 1024} KB)")

    scenario_uploads = []
    if not only or only == "scenarios":
        print("Building scenario uploads...")
        scenario_uploads = build_scenario_uploads(benchmarks_dir)
        print(f"  {len(scenario_uploads)} scenario files")

    if args.emit_scenarios and scenario_uploads:
        # Mirror the R2 key layout under _out/ (data/*.json + scenarios/.../*.json),
        # so the whole thing can be served as a local stand-in for the R2 bucket.
        print(f"Writing local copy under {WEB_DATA}/ (data/ + scenarios/) ...")
        for name, content in data_files.items():
            p = WEB_DATA / "data" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        for key, content in scenario_uploads:
            p = WEB_DATA / key
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        print(f"  serve {WEB_DATA} and set PUBLIC_DATA_BASE to its URL")

    if args.dry_run:
        print("\nDry run — skipping upload.")
        return

    print("\nUploading to R2...")
    s3 = get_s3()

    if data_files:
        upload_data_files(s3, data_files)

    if scenario_uploads:
        print(f"Uploading {len(scenario_uploads)} scenario files...")
        upload_scenarios(s3, scenario_uploads, workers=args.workers)

    print("\nDone.")


if __name__ == "__main__":
    main()
