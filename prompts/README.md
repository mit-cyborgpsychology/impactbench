# Prompts

All prompt text used by the pipeline lives here and is loaded via `lib/core/prompts.py:load()`.
No prompt strings should be inlined in `lib/core/*.py` — if you add a new LLM call, add a
`.txt` file here instead.

| File | Used by | Role |
|---|---|---|
| `benchmark_designer_system.txt` | `gen_metrics.py`, `generate.py` | system — metric + scenario generation |
| `generate_metrics.txt` | `gen_metrics.py` | user — generate metrics from a benchmark description |
| `generate_scenarios.txt` | `generate.py` | user — generate scenarios for one metric |
| `demographic_adapter_system.txt` | `generate.py` | system — demographic expansion |
| `expand_demographic.txt` | `generate.py` | user — adapt a scenario to one demographic combo |
| `user_system.txt` | `simulate.py` | system — user-simulator persona for a conversation |
| `adversarial_block.txt` | `simulate.py` | fragment — substituted into `user_system.txt` when a metric is being probed adversarially |
| `turn_first.txt` | `simulate.py` | user — opening turn instruction to the user-simulator |
| `turn_next.txt` | `simulate.py` | user — subsequent turn instruction to the user-simulator |
| `default_target_system.txt` | `simulate.py` | system — fallback target system prompt when a scenario defines none; blank on purpose, no system message is sent when empty |
| `evaluate_system.txt` | `evaluate.py` | system — evaluator role |
| `evaluate_batch.txt` | `evaluate.py` | user — score one conversation against a batch of metrics |

## Conventions

- `$name` placeholders are filled with `Template.substitute` — `load()` raises `KeyError`
  naming the missing key if a required one isn't passed.
- `load(name)` with no kwargs returns the file raw (must contain no `$` placeholders left unfilled).
- Fragments like `adversarial_block.txt` aren't sent to a model on their own — they're
  rendered first, then passed as a `$block_name` value into another template.
