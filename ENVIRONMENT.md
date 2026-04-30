# Preference Model Environment — Integration Guide

This document describes the RL environment for the KV-cache task. It is
intended for whoever is plugging this into an RL training loop.

## Quick start

```bash
# 1. Build the sandbox image (once)
docker build -t preference-env:latest -f docker/Dockerfile .

# 2. Smoke-test
python tests/test_episode.py
```

The smoke test runs three scenarios — reference solution, naive copy,
hardcoded outputs — and verifies the first passes and the others fail.
If this is green, the environment is wired correctly.

## Programmatic interface

```python
from env import Environment

with Environment(episode_files=".", reward_mode="binary") as env:
    initial_prompt = env.reset()
    # ... agent loop ...
    result = env.step({"type": "submit"})
    reward = result.reward            # 0.0 or 1.0
    verdict = result.info["verdict"]  # full diagnostic
```

The agent's only actions are `write_file`, `read_file`, `exec`, and `submit`.

## Reward modes

- `"binary"` (default): 1.0 on full pass, 0.0 otherwise. **Recommended.**
- `"shaped"`: continuous reward in [0, 0.99] for partial credit, 1.0 for
  full pass. Useful early in training when binary signal is too sparse.

## Known limitations / TODO before production

1. **`prompts_hidden.json` lives in the bind-mounted workdir.** The agent
   can't list it (filtered) and can't read it via `read_file` (filtered),
   but a sufficiently determined agent could read it via `exec(["cat", ...])`.
   The proper fix is to mount it at a path the judge knows but the agent's
   workspace doesn't include — e.g. mount it at `/private/prompts_hidden.json`
   and update `judge.py` to read from there.
2. **Memory gate is a placebo on CPU.** `tracemalloc` doesn't see tensor
   storage. For real memory enforcement, run the env on GPU and switch to
   `torch.cuda.max_memory_allocated`. See discussion in `judge.py`.
3. **Speed gate has nondeterministic timing variance.** Median-of-3 with
   `torch.set_num_threads(1)` keeps it stable in our tests, but on a noisy
   shared host you may see flakes near the 2.0× boundary. For RL training,
   this is fine (occasional false negatives don't break learning); for
   evaluation, bump `N_TIMING_RUNS` to 5–7.
4. **No streaming / partial observation.** Each `exec` returns full stdout
   at the end. For very long-running commands an agent can't stream; not
   relevant for this task but worth knowing.

## Wiring into an RL trainer

`Environment.step` returns a `StepResult` with `(observation, reward, done, info)` —
the same shape Gym/PettingZoo use. A typical integration:

```python
def rollout(policy, env_factory):
    env = env_factory()
    obs = env.reset()
    trajectory = []
    while True:
        action = policy(obs)               # LLM call producing an action dict
        result = env.step(action)
        trajectory.append((obs, action, result.reward))
        obs = result.observation
        if result.done:
            break
    env.close()
    return trajectory, result.reward
```

The action dict format is small enough that you can prompt the LLM directly
to emit it as JSON, or wrap it in a tool-call schema if your training stack
prefers that.