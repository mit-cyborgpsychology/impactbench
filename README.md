# Open Benchmark of AI Impact on Humans

Pipeline for running LLM behavioral benchmarks. Given a benchmark description, it generates metrics, constructs adversarial scenarios, simulates multi-turn conversations, and scores them.

## Setup

```bash
uv sync
cp .env.example .env   # add your API keys
```

`.env` keys (only the providers you use):

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
DEEPINFRA_TOKEN=...
XAI_API_KEY=...
```

Edit `config.yaml` to set which models to use as the user simulator, evaluator, and targets.

## Usage

```bash
python main.py <benchmark> all              # run all phases
python main.py <benchmark> gen_metrics      # phase 1: generate metrics from description
python main.py <benchmark> gen_scenarios    # phase 2: generate scenarios from metrics
python main.py <benchmark> simulate <model> # phase 3: run conversations
python main.py <benchmark> evaluate <model> # phase 4: score conversations
python main.py <benchmark> aggregate        # phase 5: aggregate across models
python main.py all                          # all benchmarks × all targets
```

Config is in `config.yaml`. Use `--config` to specify a different file.

## Structure

```
benchmarks/<name>/
  benchmark.yaml       # benchmark definition (name, description, metrics)
  scenarios.json       # generated scenarios
  runs/<model>/
    conversations.json
    scores.json
    cost.json
  results.json         # aggregated results across models

lib/
  core/               # pure functions: generate, simulate, evaluate, aggregate
  pipeline/           # phase runners: orchestrate core + caching + concurrency
  task/               # decorators: row_cache, concurrent, retry, write_json

prompts/              # prompt templates
tests/
main.py
config.yaml
```

## Benchmark definition

`benchmark.yaml` needs at minimum a `name` and `description`. Run `gen_metrics` to populate `metrics` from the description, or write them manually.

```yaml
name: My Benchmark
description: >
  What behavior you are testing and why it matters.

metrics: [] # populated by gen_metrics, or write manually
```
