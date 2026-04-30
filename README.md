# Preference Model Assessment — KV-Cache RL Environment

An RL environment for training LLMs on a realistic ML engineering task: implement KV-caching in a transformer generation loop. The candidate must produce outputs that are **bitwise identical** to a naive reference implementation while running **at least 2× faster**.

The environment is dual-gate by design — correctness *and* speed must both be satisfied — which makes reward hacking nearly impossible. You can't pass correctness by copying the naive code (fails the speed gate), can't pass speed by hardcoding outputs (fails correctness on hidden prompts), can't pass either by returning early (fails the memory shape check).

---

## Why this task

Most RL-for-code environments either (a) check correctness only, which is reward-hackable by memorization, or (b) check a single metric, which is gameable by adversarial submissions that satisfy the metric without doing the underlying work. This environment is built around the observation that **two independent gates with different failure modes compose into a much stronger guarantee than either alone**.

KV-caching specifically was chosen because:

- It's a real task every inference engineer does. Realistic, not contrived.
- It has a clean correctness oracle (bitwise equality with the naive implementation). No fuzzy matching, no "close enough."
- It has a clean speed oracle (wall-clock ratio against the same naive implementation on the same hardware). No external benchmarks to drift.
- The two oracles can't be satisfied by the same shortcut. Each gate closes a different attack surface.
- It scales: the same dual-gate pattern generalizes to flash attention, paged attention, speculative decoding, quantization, and dozens of other inference optimizations.

---

## Repository layout

```
Pefrence_Model_Asses/
├── README.md                  # this file
├── ENVIRONMENT.md             # integration guide for RL trainers
├── requirements.txt           # torch, numpy
│
│   # --- The core environment files ---
├── model.py                   # tiny GPT-style transformer (frozen)
├── generate_naive.py          # slow reference implementation (frozen)
├── generate.py                # STUB — what the LLM-under-test must implement
├── judge.py                   # the test harness with four gates
├── prompts_visible.json       # dev prompts the candidate can see
├── prompts_hidden.json        # judge-only held-out prompts
├── prompt_for_llm.md          # the task description shown to the agent
├── run_env.sh                 # convenience runner
│
│   # --- The RL packaging layer ---
├── env/
│   ├── __init__.py
│   ├── sandbox.py             # Docker-based isolated runner
│   ├── environment.py         # reset/step/close API for trainers
│   ├── reward.py              # judge verdict → scalar reward
│   └── agent_prompt.py        # system prompt + task description
├── docker/
│   ├── Dockerfile             # sandbox image
│   └── entrypoint.sh
└── tests/
    └── test_episode.py        # end-to-end smoke test (no LLM needed)
```

---

## Quick start

### Prerequisites

- Python 3.11+
- Docker Desktop (or Docker Engine on Linux)
- ~2 GB free disk for the sandbox image

### Setup

```bash
# Clone / cd into the project
cd Pefrence_Model_Asses

# Create a virtualenv and install host-side deps
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Build the sandbox image (one-time, ~1–8 min depending on network)
docker build -t preference-env:latest -f docker/Dockerfile .
```

### Verify the environment works

The fastest way to confirm everything is wired correctly is to run the judge directly with the reference solution:

```bash
# 1. Drop the reference solution into generate.py (see "Reference solution" below)
# 2. Run the judge
python judge.py
```

Expected output:

```json
{
  "pass": true,
  "score": 1.0,
  "score_continuous": 1.0,
  "gates": {
    "existence":   {"pass": true},
    "correctness": {"pass": true},
    "speed":       {"pass": true, "details": "naive=0.699s candidate=0.131s speedup=5.33x", "speedup": 5.325},
    "memory":      {"pass": true, "details": "...", "linearity": 1.0}
  }
}
```

Then run the full Dockerized smoke test:

```bash
python -u tests/test_episode.py
```

This runs three scenarios (reference solution, naive copy, hardcoded zeros) end-to-end through the sandbox and verifies each lands at the expected gate.

---

## The four gates

The judge checks the candidate's `generate.py` against four independent gates. **All four must pass** for `pass: true`. Each gate closes a different failure mode.

### Gate 1: Existence

`generate.py` exists and exposes a callable `generate(model, input_ids, max_tokens) -> torch.LongTensor`.

**Catches:** missing files, syntax errors, missing function, wrong signature, import errors.

### Gate 2: Correctness

The candidate's output must be **bitwise identical** to `generate_naive.generate` on five held-out prompts (the candidate never sees these). Same model weights, same seed, greedy decoding.

**Catches:** any logic bug, any approximation, any "close enough" hack, hardcoded outputs (which would pass on a single visible prompt but diverge on the hidden ones).

The hidden prompt set is deliberately diverse: short prompts (1 token), long prompts (12 tokens), edge-of-vocab tokens (255), low-byte tokens (1–8). This makes pattern-matching against a memorized output set infeasible.

### Gate 3: Speed

On a 256-token long-generation benchmark, the candidate must complete in ≤ 0.5× the naive wall-clock time. Median of 3 runs, single-threaded (`torch.set_num_threads(1)`) to reduce noise.

**Catches:** anyone who tried to satisfy correctness by literally calling the naive function. Speedup will be ~1.0× and they fail.

### Gate 4: Memory shape

Peak memory at sequence length T₂ should be < 1.6× peak memory at T₁ (where T₂ = 2·T₁). A real KV cache scales O(T) in memory, so the ratio approaches 2.0. Naive recompute is O(T²) on attention scores, so its ratio approaches 4.0.

**Catches:** "I just made it return early" style cheats that pass speed by skipping work. Also catches some recompute-in-disguise patterns.

> **Known limitation:** on CPU, this gate currently uses `tracemalloc`, which only sees Python heap and not torch tensor storage. It functions more as a sanity check than a strict bound. See "Known limitations" below for the GPU fix.

---

## How to use as an RL environment

The `env/` package wraps everything behind a `reset`/`step`/`close` interface that's familiar to anyone who's used Gym or PettingZoo.

### Minimal example

```python
from env import Environment

with Environment(episode_files=".", reward_mode="binary") as env:
    initial_prompt = env.reset()
    print(initial_prompt)  # task description shown to the agent

    # Agent decides to write the reference solution
    env.step({
        "type": "write_file",
        "path": "generate.py",
        "content": REFERENCE_SOLUTION,
    })

    # Agent submits — judge runs, episode ends
    result = env.step({"type": "submit"})

    print(f"Reward: {result.reward}")          # 1.0 or 0.0
    print(f"Verdict: {result.info['verdict']}") # full diagnostic
```

### Action space

The agent has four actions, all JSON-serializable:

| Action | Description |
|---|---|
| `{"type": "write_file", "path": str, "content": str}` | Overwrite a file in the workspace |
| `{"type": "read_file", "path": str}` | Read a file's contents |
| `{"type": "exec", "cmd": list[str]}` | Run a shell command (120s timeout) |
| `{"type": "submit"}` | Declare done; judge runs and episode ends |

The agent gets a 50-action budget per episode. Hitting it terminates the episode with reward 0.

### Reward modes

- **`"binary"`** (default, recommended): 1.0 on full pass, 0.0 otherwise. Sparse but unambiguous.
- **`"shaped"`**: dense reward in [0, 0.99] with partial credit for passing earlier gates. Useful early in training when binary signal is too sparse. The shaped reward is capped below 1.0 so a non-passing run can never tie a passing one.

### Sandbox properties

Every episode runs in a fresh Docker container with:

- No network access (`--network=none`)
- Read-only root filesystem (writable bind-mount only at `/workspace`)
- 2 GB memory cap, 1 CPU
- Non-root user (uid 1000)
- 120s default timeout per `exec` call
- 180s timeout for `submit` (judge runs)

### Plugging into an RL trainer

```python
def rollout(policy, env_factory):
    env = env_factory()
    obs = env.reset()
    trajectory = []
    while True:
        action = policy(obs)            # LLM call producing an action dict
        result = env.step(action)
        trajectory.append((obs, action, result.reward))
        obs = result.observation
        if result.done:
            break
    env.close()
    return trajectory, result.reward
```

The action format is small enough to prompt an LLM to emit it as JSON directly, or wrap it in a tool-call schema if your training stack prefers that.

---

## Reference solution

For verification only — do **not** ship this with the candidate VM. Drop into `generate.py` to confirm the environment is wired correctly.

```python
import torch

@torch.no_grad()
def generate(model, input_ids, max_tokens):
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
```

On a typical CPU run, this hits ~5× speedup over naive.

---

## Reward hacking analysis

The dual-gate design was chosen specifically because every shortcut I could think of fails at exactly one gate, predictably:

| Attack | Gate that catches it | Why |
|---|---|---|
| Copy `generate_naive.py` verbatim | **Speed** | Speedup ≈ 1.0×, fails the 2.0× threshold |
| Hardcode the output for visible prompts | **Correctness** | Hidden prompts produce different outputs |
| Return zeros / garbage tokens | **Correctness** | Diverges on the very first hidden prompt |
| `time.sleep(-1)` style speed fakery | n/a | Doesn't exist; wall-clock measures actual elapsed time |
| Return early after a few tokens | **Correctness** | Output shape will mismatch the reference |
| Use a smaller model in candidate code | **Correctness** | Different weights → different argmax |
| Cache only K, not V (subtle bug) | **Correctness** | Attention values diverge from naive |
| Implement caching but with wrong masking | **Correctness** | Off-by-one in causal mask → wrong logits |

The smoke test in `tests/test_episode.py` exercises the first three of these automatically. If it ever stops behaving as expected, that's a regression in the environment itself.

---

## Determinism

Reproducibility matters because bitwise equality is the correctness oracle. The environment enforces:

- Fixed seed (`SEED = 1337`) in `model.build_model()`.
- Greedy decoding only (argmax), no sampling.
- Float32 throughout — no fp16/bf16 reduced-precision kernels with their nondeterministic accumulation.
- Single-threaded torch (`torch.set_num_threads(1)`) to eliminate thread-scheduling-based variance.
- Causal mask sliced based on `T_past`, so cached and uncached forward passes produce identical attention patterns.

---

## Performance notes

### Episode latency

| Stage | Native (CPU, Linux) | Docker on Apple Silicon |
|---|---|---|
| Container cold start | n/a | 1–2s |
| Torch import | 3–5s | 4–7s |
| 5 correctness prompts | 5–10s | 10–20s |
| Speed benchmark (3 runs × 2 impls) | 6–10s | 15–30s |
| Memory test | 2–4s | 4–8s |
| **Total per episode** | **~20–30s** | **~45–90s** |

Docker Desktop on Mac runs Linux containers in a VM, which adds overhead to every operation. For RL training at scale, run on a Linux host with native Docker.

### Speed gate stability

Median-of-3 with single-threaded torch keeps timing stable in our tests. On a noisy shared host, you may occasionally see the speedup dip near the 2.0× threshold for the reference solution. If this becomes an issue, bump `N_TIMING_RUNS` from 3 to 5 in `judge.py`.

---

## Known limitations

These are real holes you'd want to plug before deploying this for production RL training. Listed in priority order.

### 1. `prompts_hidden.json` lives in the bind-mounted workdir

The agent can't list it (filtered in `_list_files`) and can't read it via `read_file` (filtered in `step`), but a determined agent could read it via `exec(["cat", "prompts_hidden.json"])`.

**Fix:** mount it at a path the judge knows but the agent's workspace doesn't include — e.g. `/private/prompts_hidden.json` — and update `judge.py` to read from that path.

### 2. Memory gate is partially a placebo on CPU

`tracemalloc` doesn't see torch tensor storage (which lives in C++ allocators). The 8KB it measures is bookkeeping noise, not the actual cache.

**Fix:** run on GPU and switch to `torch.cuda.max_memory_allocated()` for the memory gate. Or add a behavioral test asserting that cache tensors retain their `id()` across timesteps.

### 3. Speed gate has nondeterministic timing variance

For RL training, occasional false negatives don't break learning. For high-stakes evaluation (e.g. final grading of a candidate), bump `N_TIMING_RUNS` to 5–7 and consider winsorizing rather than taking the median.

### 4. Single-task, single-difficulty

The environment as shipped is one task at one difficulty level. For curriculum learning, you'd want a family of tasks with progressive difficulty (e.g. "implement KV cache", "implement KV cache with sliding window", "implement KV cache with paged memory") sharing the same dual-gate harness.

---

## Design decisions and tradeoffs

A few choices in this codebase are worth explaining because the alternatives weren't obviously worse.

**Why a tiny GPT instead of using HuggingFace transformers?**
Self-contained, deterministic, and fast enough that the speed gate has real signal. Pulling in transformers would add a 200MB dependency and introduce nondeterminism from kernel-level optimizations we'd then have to disable.

**Why bitwise equality instead of allclose?**
Because the candidate's *correct* implementation produces literally identical outputs — the math is the same, just reorganized. Allowing tolerance (`torch.allclose(rtol=1e-5)`) would let through subtly broken implementations that compute the right shape with the wrong values. Bitwise is harsh but it's the right harshness for this task.

**Why Docker instead of subprocess + chroot?**
Docker is the de facto standard for ML training infra, and the isolation primitives (network, memory, CPU, read-only fs) come for free. Subprocess + chroot is more lightweight but harder to harden correctly.

**Why a 50-action budget?**
A reasonable solution needs 5–10 actions (read files, write `generate.py`, run judge, maybe iterate once on a bug, submit). 50 leaves plenty of headroom for early-training agents that get stuck in `read_file` loops, while still capping pathological behavior.

**Why is `prompts_hidden.json` in the bind-mount at all if we want to hide it from the agent?**
Convenience for the judge — same working directory, no path juggling. This is the limitation noted above and the first thing I'd fix in production.

---

## Troubleshooting

**"`docker: command not found`"**
Install Docker Desktop (Mac/Windows) or Docker Engine (Linux). Then run `docker build` again.

**"`Cannot connect to the Docker daemon`"**
Docker Desktop isn't running. Open it from Applications.

**Smoke test prints `=== reference solution ===` and hangs**
Not actually hung — Python output is buffered. Run with `python -u tests/test_episode.py` to see progress in real time. Each Docker episode takes 45–90s on Mac.

**Speed gate fails for the reference solution**
Almost always Docker Desktop overhead on Mac. Bump `N_TIMING_RUNS` to 5 in `judge.py`, or run the smoke test natively (`python judge.py` outside Docker) to confirm the underlying logic works.

**`NotImplementedError: Implement KV-cached generation here.`**
This is the stub `generate.py` raising correctly. The judge catches this exception and reports it as a correctness failure — that's the expected behavior on a fresh repo. To verify the environment, drop in the reference solution from "Reference solution" above.

**`generate.py: prompt 0: shape mismatch`**
Your `generate.py` is returning the wrong shape. The contract is: input shape `(B, T0)`, output shape `(B, T0 + max_tokens)`. The reference solution shows the right pattern.

---

## Extending this environment

The dual-gate harness in `judge.py` generalizes well. To add a new task:

1. Replace `model.py` and `generate_naive.py` with the new reference setup.
2. Update `prompts_hidden.json` with held-out test inputs appropriate to the new task.
3. Adjust the four gates in `judge.py` — the structure stays the same, the specifics change.
4. Update `prompt_for_llm.md` with the new task description.

The `env/` packaging layer is task-agnostic and shouldn't need changes.

Concrete extension ideas, in roughly increasing difficulty:

- **Flash attention.** Same model, same correctness oracle, but the speedup target is higher and memory matters more.
- **Sliding window attention.** Correctness oracle changes — outputs aren't bitwise equal to standard attention, but to a sliding-window reference. Tests whether the candidate can implement to a *spec* rather than match an existing implementation.
- **Speculative decoding.** Two models (draft + verifier), correctness against the verifier alone, speed against verifier alone. Much harder.
- **Quantization.** Outputs are no longer bitwise equal — replace with logit-distance check (e.g. KL divergence < threshold), keep the speed gate.

---

## License and attribution

Internal assessment artifact. Not for redistribution.

Built as a take-home for the Preference Model initial assessment.