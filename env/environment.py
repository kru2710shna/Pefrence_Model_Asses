"""
The RL environment. A trainer interacts with this and nothing else.

Lifecycle:
    env = Environment(...)
    obs = env.reset()             # fresh episode, returns task description
    obs = env.step(action)        # agent took an action, get next obs
    ...
    reward, done, info = env.finish()  # judge runs, episode ends
    env.close()

Action protocol (kept deliberately small):
    {"type": "write_file", "path": "generate.py", "content": "..."}
    {"type": "read_file",  "path": "generate.py"}
    {"type": "exec",       "cmd": ["python", "judge.py"]}
    {"type": "submit"}            # agent declares done, judge runs
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent_prompt import build_initial_prompt
from .reward import Reward, RewardMode, format_reward, parse_verdict
from .sandbox import ExecResult, Sandbox


@dataclass
class StepResult:
    observation: str
    reward: float | None      # only non-None on the terminal step
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class Environment:
    MAX_ACTIONS_PER_EPISODE = 50      # hard cap — agents that loop forever fail

    def __init__(
        self,
        episode_files: Path | str,
        reward_mode: RewardMode = "binary",
        scratch_root: Path | None = None,
    ):
        self.sandbox = Sandbox(Path(episode_files), scratch_root=scratch_root)
        self.reward_mode: RewardMode = reward_mode
        self._action_count = 0
        self._done = False
        self._final_reward: Reward | None = None

    # ---------- lifecycle ----------

    def reset(self) -> str:
        """Start a new episode. Returns the initial observation (task prompt)."""
        self.sandbox.reset()
        self._action_count = 0
        self._done = False
        self._final_reward = None
        return build_initial_prompt(self._list_files())

    def step(self, action: dict) -> StepResult:
        """Execute one action. Returns observation/reward/done."""
        if self._done:
            raise RuntimeError("step() called after episode ended; call reset()")
        self._action_count += 1

        if self._action_count > self.MAX_ACTIONS_PER_EPISODE:
            return self._terminate_with_failure(
                "exceeded max actions per episode")

        kind = action.get("type")

        if kind == "write_file":
            try:
                self.sandbox.write_file(action["path"], action["content"])
                return StepResult(f"wrote {action['path']}", None, False)
            except Exception as e:
                return StepResult(f"write_file error: {e}", None, False)

        if kind == "read_file":
            try:
                content = self.sandbox.read_file(action["path"])
                return StepResult(content, None, False)
            except Exception as e:
                return StepResult(f"read_file error: {e}", None, False)

        if kind == "exec":
            res = self.sandbox.exec(action["cmd"])
            return StepResult(self._format_exec(res), None, False)

        if kind == "submit":
            return self._run_judge_and_terminate()

        return StepResult(f"unknown action type: {kind!r}", None, False)

    def close(self) -> None:
        self.sandbox.close()

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, *_):
        self.close()

    # ---------- internals ----------

    def _run_judge_and_terminate(self) -> StepResult:
        res = self.sandbox.exec(["python", "judge.py"], timeout_s=180)
        if res.timed_out:
            return self._terminate_with_failure("judge timed out (180s)")
        try:
            verdict = parse_verdict(res.stdout)
        except ValueError as e:
            return self._terminate_with_failure(
                f"could not parse verdict: {e}\n"
                f"stderr:\n{res.stderr[-1000:]}")

        reward = format_reward(verdict, mode=self.reward_mode)
        self._final_reward = reward
        self._done = True
        obs = (
            f"Judge verdict:\n{json.dumps(verdict, indent=2)}\n\n"
            f"Reward: {reward.value:.3f} ({'pass' if reward.pass_ else 'fail'})"
        )
        return StepResult(
            observation=obs,
            reward=reward.value,
            done=True,
            info={"verdict": verdict, "reward": reward.__dict__},
        )

    def _terminate_with_failure(self, reason: str) -> StepResult:
        self._done = True
        self._final_reward = Reward(0.0, False, "harness", {"reason": reason})
        return StepResult(
            observation=f"Episode terminated: {reason}",
            reward=0.0,
            done=True,
            info={"reason": reason},
        )

    def _list_files(self) -> list[str]:
        if self.sandbox.workdir is None:
            return []
        return sorted(
            str(p.relative_to(self.sandbox.workdir))
            for p in self.sandbox.workdir.rglob("*")
            if p.is_file() and "prompts_hidden" not in p.name
        )

    @staticmethod
    def _format_exec(res: ExecResult) -> str:
        parts = [f"exit_code: {res.exit_code}"]
        if res.timed_out:
            parts.append("(TIMED OUT)")
        if res.stdout:
            parts.append(f"stdout:\n{res.stdout[-2000:]}")
        if res.stderr:
            parts.append(f"stderr:\n{res.stderr[-1000:]}")
        return "\n".join(parts)