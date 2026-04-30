"""
End-to-end episode smoke test. Runs without an actual LLM by playing back
a fixed action sequence (the reference solution).

This is the test you run before plugging in a real agent.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from env import Environment


REFERENCE_SOLUTION = '''\
import torch

@torch.no_grad()
def generate(model, input_ids, max_tokens):
    logits, past_kvs = model(input_ids, past_kvs=None)
    next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    out = torch.cat([input_ids, next_tok], dim=1)
    for _ in range(max_tokens - 1):
        logits, past_kvs = model(next_tok, past_kvs=past_kvs)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_tok], dim=1)
    return out
'''

NAIVE_COPY = '''\
import torch
from generate_naive import generate
'''

HARDCODED = '''\
import torch
def generate(model, input_ids, max_tokens):
    return torch.zeros((input_ids.shape[0], input_ids.shape[1] + max_tokens), dtype=torch.long)
'''


def run_episode(solution_code: str) -> dict:
    """Run one episode where the 'agent' writes `solution_code` and submits."""
    with Environment(episode_files=REPO) as env:
        env.step({"type": "write_file", "path": "generate.py", "content": solution_code})
        result = env.step({"type": "submit"})
    return {"reward": result.reward, "info": result.info}


def main():
    cases = [
        ("reference solution", REFERENCE_SOLUTION, True),
        ("naive copy",         NAIVE_COPY,         False),
        ("hardcoded zeros",    HARDCODED,          False),
    ]
    failures = []
    for name, code, expected_pass in cases:
        print(f"\n=== {name} ===")
        out = run_episode(code)
        actual_pass = out["reward"] == 1.0
        verdict = out["info"].get("verdict", {})
        stage = verdict.get("stage_failed", "—")
        print(f"  reward={out['reward']}  pass={actual_pass}  stage_failed={stage}")
        if actual_pass != expected_pass:
            failures.append(f"{name}: expected pass={expected_pass}, got pass={actual_pass}")

    print()
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All scenarios behaved as expected.")


if __name__ == "__main__":
    main()