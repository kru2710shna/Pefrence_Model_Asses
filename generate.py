"""

Requirements:
- Expose a function `generate(model, input_ids, max_tokens) -> torch.LongTensor`
  with the same signature as generate_naive.generate.
- Output must be BITWISE IDENTICAL to generate_naive on every prompt.
- Must run at least 2x faster than generate_naive on long generations.

Hints (look at model.py):
- model(input_ids, past_kvs=None) is the prefill call. It returns
  (logits, new_kvs) where new_kvs is a list of (k, v) per layer.
- model(input_ids, past_kvs=<previous>) appends to the cache. Pass only the
  NEW tokens here, not the whole sequence.
- Greedy decoding = argmax on the LAST timestep's logits.

Do not modify model.py or generate_naive.py.
"""

import torch


@torch.no_grad()
def generate(model, input_ids: torch.Tensor, max_tokens: int) -> torch.Tensor:
    # Prefill: process the entire prompt once, get logits + initial cache.
    logits, past_kvs = model(input_ids, past_kvs=None)
    next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    out = torch.cat([input_ids, next_tok], dim=1)

    # Decode: feed only the new token at each step, reuse the cache.
    for _ in range(max_tokens - 1):
        logits, past_kvs = model(next_tok, past_kvs=past_kvs)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_tok], dim=1)
    return out
