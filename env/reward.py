"""
Maps judge verdicts to RL rewards.

Two reward modes:
  - "binary": 1.0 on full pass, 0.0 otherwise. Sparse but unambiguous —
              the recommended default for production RL.
  - "shaped": dense reward in [0, 1] using the judge's score_continuous,
              with explicit partial credit for passing earlier gates.
              Useful early in training when binary signal is too sparse.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal


RewardMode = Literal["binary", "shaped"]


@dataclass
class Reward:
    value: float
    pass_: bool
    stage_failed: str | None
    diagnostics: dict


def format_reward(verdict: dict, mode: RewardMode = "binary") -> Reward:
    if mode == "binary":
        return Reward(
            value=1.0 if verdict.get("pass") else 0.0,
            pass_=bool(verdict.get("pass")),
            stage_failed=verdict.get("stage_failed"),
            diagnostics=verdict.get("gates", {}),
        )

    if mode == "shaped":
        # Partial credit ladder: each gate passed is worth a fixed amount.
        # This avoids the trap where the shaped reward exceeds the binary
        # reward — a passing run is always 1.0 in both modes.
        if verdict.get("pass"):
            return Reward(1.0, True, None, verdict.get("gates", {}))
        gates = verdict.get("gates", {})
        partial = 0.0
        if gates.get("existence", {}).get("pass"):    partial += 0.1
        if gates.get("correctness", {}).get("pass"):  partial += 0.4
        # Speed and memory contribute proportional credit even on fail.
        speedup = gates.get("speed", {}).get("speedup", 0.0) or 0.0
        partial += 0.3 * min(speedup / 2.0, 1.0)
        linearity = gates.get("memory", {}).get("linearity", 0.0) or 0.0
        partial += 0.2 * linearity
        # Cap at 0.99 so a non-passing run can never tie a passing one.
        return Reward(
            value=min(partial, 0.99),
            pass_=False,
            stage_failed=verdict.get("stage_failed"),
            diagnostics=gates,
        )

    raise ValueError(f"unknown reward mode: {mode}")


def parse_verdict(stdout: str) -> dict:
    """Pull the JSON verdict out of judge.py's stdout. Robust to extra logging."""
    # Find the last well-formed JSON object in stdout.
    decoder = json.JSONDecoder()
    last = None
    i = 0
    while i < len(stdout):
        try:
            obj, end = decoder.raw_decode(stdout[i:])
            last = obj
            i += end
        except json.JSONDecodeError:
            i += 1
    if last is None:
        raise ValueError(f"no JSON verdict found in stdout:\n{stdout[-500:]}")
    return last