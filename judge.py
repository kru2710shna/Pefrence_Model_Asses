
import importlib.util
import json
import sys
import time
import traceback
import tracemalloc
from pathlib import Path

import torch

from model import build_model
from generate_naive import generate as generate_naive

HERE         = Path(__file__).parent
HIDDEN       = HERE / "prompts_hidden.json"
CANDIDATE    = HERE / "generate.py"
VERDICT_PATH = HERE / "verdict.json"

SPEED_RATIO_REQUIRED = 2.0
N_TIMING_RUNS        = 3
MEMORY_LINEAR_TOL    = 1.6


def emit(verdict: dict) -> int:
    VERDICT_PATH.write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0 if verdict.get("pass") else 1


def fail(stage: str, reason: str, extra: dict | None = None) -> dict:
    v = {"pass": False, "stage_failed": stage, "reason": reason, "score": 0.0}
    if extra:
        v.update(extra)
    return v


def load_candidate():
    if not CANDIDATE.exists():
        raise FileNotFoundError("generate.py not found")
    spec = importlib.util.spec_from_file_location("candidate_generate", CANDIDATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "generate") or not callable(mod.generate):
        raise AttributeError("generate.py does not expose a callable `generate`")
    return mod.generate


def gate_correctness(model, candidate_fn, hidden) -> tuple[bool, str]:
    for i, prompt in enumerate(hidden["prompts"]):
        ids = torch.tensor([prompt], dtype=torch.long)
        ref = generate_naive(model, ids, hidden["max_tokens"])
        try:
            cand = candidate_fn(model, ids, hidden["max_tokens"])
        except Exception as e:
            return False, f"prompt {i}: candidate raised {type(e).__name__}: {e}"
        if not isinstance(cand, torch.Tensor):
            return False, f"prompt {i}: candidate returned {type(cand).__name__}, expected Tensor"
        if cand.shape != ref.shape:
            return False, f"prompt {i}: shape mismatch — got {tuple(cand.shape)}, expected {tuple(ref.shape)}"
        if not torch.equal(ref, cand):
            diff = (ref != cand).nonzero(as_tuple=False)
            first = diff[0].tolist() if diff.numel() else None
            return False, f"prompt {i}: outputs differ (first divergence at {first})"
    return True, ""


def time_fn(fn, model, ids, max_tokens, runs=N_TIMING_RUNS) -> float:
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(model, ids, max_tokens)
        timings.append(time.perf_counter() - t0)
    timings.sort()
    return timings[len(timings) // 2]


def gate_speed(model, candidate_fn, hidden) -> tuple[bool, str, float]:
    spec = hidden["speed_test"]
    ids = torch.tensor([spec["prompt"]], dtype=torch.long)
    t_naive = time_fn(generate_naive, model, ids, spec["max_tokens"])
    t_cand  = time_fn(candidate_fn,  model, ids, spec["max_tokens"])
    speedup = t_naive / t_cand if t_cand > 0 else float("inf")
    ok = speedup >= SPEED_RATIO_REQUIRED
    msg = f"naive={t_naive:.3f}s candidate={t_cand:.3f}s speedup={speedup:.2f}x"
    return ok, msg, speedup


def peak_memory(fn, model, ids, max_tokens) -> int:
    tracemalloc.start()
    fn(model, ids, max_tokens)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def gate_memory(model, candidate_fn) -> tuple[bool, str, float]:
    short_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    long_ids  = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    p_short = peak_memory(candidate_fn, model, short_ids, 64)
    p_long  = peak_memory(candidate_fn, model, long_ids, 128)
    ratio = p_long / p_short if p_short > 0 else float("inf")
    linearity = max(0.0, min(1.0, (4.0 - ratio) / 2.0))
    ok = ratio < MEMORY_LINEAR_TOL
    msg = f"peak(64)={p_short}B peak(128)={p_long}B ratio={ratio:.2f}"
    return ok, msg, linearity


def main() -> int:
    torch.set_num_threads(1)

    # Gate 1: existence (catches import errors, missing function, syntax errors)
    try:
        candidate_fn = load_candidate()
    except Exception as e:
        return emit(fail("existence", f"{type(e).__name__}: {e}",
                         {"traceback": traceback.format_exc()}))

    hidden = json.loads(HIDDEN.read_text())
    model  = build_model()

    # Gate 2: correctness (already catches per-prompt exceptions internally)
    try:
        ok, msg = gate_correctness(model, candidate_fn, hidden)
    except Exception as e:
        return emit(fail("correctness", f"judge error: {type(e).__name__}: {e}",
                         {"traceback": traceback.format_exc()}))
    if not ok:
        return emit(fail("correctness", msg))

    # Gate 3: speed
    try:
        ok_speed, msg_speed, speedup = gate_speed(model, candidate_fn, hidden)
    except Exception as e:
        return emit(fail("speed", f"candidate raised during timing: "
                                  f"{type(e).__name__}: {e}"))

    # Gate 4: memory
    try:
        ok_mem, msg_mem, linearity = gate_memory(model, candidate_fn)
    except Exception as e:
        return emit(fail("memory", f"candidate raised during memory test: "
                                   f"{type(e).__name__}: {e}"))

    passed = ok_speed and ok_mem
    score_continuous = (
        0.5 * 1.0
        + 0.3 * min(speedup / SPEED_RATIO_REQUIRED, 1.0)
        + 0.2 * linearity
    )

    v = {
        "pass": passed,
        "score": 1.0 if passed else 0.0,
        "score_continuous": round(score_continuous, 4),
        "gates": {
            "existence":   {"pass": True},
            "correctness": {"pass": True},
            "speed":       {"pass": ok_speed, "details": msg_speed, "speedup": round(speedup, 3)},
            "memory":      {"pass": ok_mem,   "details": msg_mem,   "linearity": round(linearity, 3)},
        },
    }
    if not ok_speed:
        v["stage_failed"] = "speed"
        v["reason"] = msg_speed
    elif not ok_mem:
        v["stage_failed"] = "memory"
        v["reason"] = msg_mem

    return emit(v)


if __name__ == "__main__":
    sys.exit(main())