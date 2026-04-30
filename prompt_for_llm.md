# Task: Implement KV-Cached Generation

You are working in a Python project with the following files:
- `model.py` — a small GPT-style transformer (do NOT modify)
- `generate_naive.py` — a slow reference implementation (do NOT modify)
- `generate.py` — currently a stub; you must implement it
- `prompts_visible.json` — example prompts you can use for development
- `judge.py` — the test harness

## Your job

Implement `generate(model, input_ids, max_tokens)` in `generate.py` such that:

1. The output is **bitwise identical** to `generate_naive.generate` on every prompt.
2. It runs **at least 2x faster** than the naive version on long generations.

## How to verify locally

```bash
python judge.py
```

The judge prints a JSON verdict and exits 0 on pass, 1 on fail.

## Hints

- Read `model.py` carefully. The `forward` method already supports a `past_kvs` argument.
- The naive version recomputes attention over the entire prefix every step. You can do better.
- Greedy decoding = `argmax` over the final-position logits.
- Don't modify `model.py` or `generate_naive.py`. The judge will use the originals.