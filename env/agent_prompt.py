"""The system prompt + task description shown to the agent at episode start."""

SYSTEM_PROMPT = """\
You are a machine learning engineer. You will be given a coding task and a set
of files in a sandboxed workspace. You have four actions available:

  write_file(path, content) — overwrite a file
  read_file(path)           — read a file
  exec(cmd)                 — run a shell command (list-of-strings form)
  submit()                  — declare you're done; the judge will run

You should iterate: write code, run it, read errors, fix, repeat. When you
believe the task is complete, call submit(). The judge will run automatically
and produce your reward.

Constraints:
- No network access.
- 50 action budget per episode.
- Each exec call has a 120s timeout.
- Do not modify model.py, generate_naive.py, or judge.py.
"""


TASK_DESCRIPTION = """\
# Task: Implement KV-Cached Generation

Your job is to make `generate.py` produce greedy-decoded outputs that are
**bitwise identical** to `generate_naive.py` while running **at least 2x
faster** on long generations.

The model in `model.py` already supports caching — its forward method takes a
`past_kvs` argument. Naive recompute corresponds to passing `past_kvs=None`
every step; cached generation corresponds to passing the cache forward and
feeding only the new token each step.

## Files in your workspace
{file_listing}

## How to verify
Run `python judge.py`. The judge will print a JSON verdict and exit 0 on
pass / 1 on fail. When `verdict.json` shows `"pass": true`, call submit().
"""


def build_initial_prompt(file_listing: list[str]) -> str:
    listing = "\n".join(f"  - {f}" for f in file_listing) if file_listing else "  (empty)"
    return SYSTEM_PROMPT + "\n\n" + TASK_DESCRIPTION.format(file_listing=listing)