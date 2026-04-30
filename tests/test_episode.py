"""
End-to-end episode smoke test. Runs without an actual LLM by playing back
a fixed action sequence (the reference solution).

This is the test you run before plugging in a real agent.
"""
import os
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


def run_episode(solution_code: str, speed_ratio: str) -> dict:
    """Run one episode with a specific SPEED_RATIO threshold."""
    # Set the threshold for this scenario; restore afterwards.
    prev = os.environ.get("SPEED_RATIO")
    os.environ["SPEED_RATIO"] = speed_ratio
    try:
        with Environment(episode_files=REPO) as env:
            env.step({"type": "write_file", "path": "generate.py", "content": solution_code})
            result = env.step({"type": "submit"})
    finally:
        if prev is None:
            del os.environ["SPEED_RATIO"]
        else:
            os.environ["SPEED_RATIO"] = prev
    return {"reward": result.reward, "info": result.info}


def main():
    # Per-scenario thresholds calibrated for Mac-Docker noise:
    # - Reference (real cache): typically 1.2× on Mac; 1.05 leaves margin.
    # - Naive copy (no cache):  exactly ~1.0×; 1.15 keeps it failing reliably.
    # - Hardcoded:              fails correctness regardless of threshold.
    # Production default in judge.py stays at 2.0×.
    cases = [
        ("reference solution", REFERENCE_SOLUTION, "1.05", True),
        ("naive copy",         NAIVE_COPY,         "1.15", False),
        ("hardcoded zeros",    HARDCODED,          "2.0",  False),
    ]
    failures = []
    for name, code, threshold, expected_pass in cases:
        print(f"\n=== {name} (threshold={threshold}) ===", flush=True)
        out = run_episode(code, threshold)
        actual_pass = out["reward"] == 1.0
        verdict = out["info"].get("verdict", {})
        stage = verdict.get("stage_failed", "—")
        speedup = verdict.get("gates", {}).get("speed", {}).get("speedup", "n/a")
        print(f"  reward={out['reward']}  pass={actual_pass}  "
              f"stage_failed={stage}  speedup={speedup}", flush=True)
        if actual_pass != expected_pass:
            failures.append(f"{name}: expected pass={expected_pass}, got pass={actual_pass}")

    print(flush=True)
    if failures:
        print("FAIL:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        sys.exit(1)
    print("All scenarios behaved as expected.", flush=True)


if __name__ == "__main__":
    main()