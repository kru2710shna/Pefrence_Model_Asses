"""
Naive autoregressive generation: re-runs the entire prefix through the
model at every step. O(T^2) per step, O(T^3) total. This is the file
the candidate must beat by 2x while matching outputs bitwise.
"""
import torch
from model import build_model


@torch.no_grad()
def generate(model, input_ids: torch.Tensor, max_tokens: int) -> torch.Tensor:
    """
    input_ids: (B, T0) prompt
    Returns:   (B, T0 + max_tokens) — the prompt with greedily-decoded continuation.
    """
    out = input_ids
    for _ in range(max_tokens):
        logits, _ = model(out, past_kvs=None)            # full recompute
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_tok], dim=1)
    return out


if __name__ == "__main__":
    torch.set_num_threads(1)
    m = build_model()
    prompt = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    print(generate(m, prompt, max_tokens=10))