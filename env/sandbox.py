"""
Docker-backed sandbox for running candidate code in isolation.

Design:
- One container per command (cheap with a pre-built image).
- Workdir is bind-mounted; file edits persist across exec calls within an
  episode, but reset() creates a fresh workdir from a curated template.
- No network, non-root user, hard timeout, capped memory and CPU.
- Returns (exit_code, stdout, stderr) — the only interface the agent gets.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


# Files copied into each fresh episode workdir. Anything not on this list
# (venv, .git, __pycache__, tests, env/, docker/, README, etc.) is excluded.
EPISODE_FILES = [
    "model.py",
    "generate_naive.py",
    "generate.py",
    "judge.py",
    "prompts_visible.json",
    "prompts_hidden.json",
    "prompt_for_llm.md",
    "requirements.txt",
]


class Sandbox:
    """Per-episode sandbox. One Sandbox = one episode = one workdir."""

    IMAGE = "preference-env:latest"
    DEFAULT_TIMEOUT_S = 120
    MEMORY_LIMIT = "2g"
    CPU_LIMIT = "1.0"

    def __init__(self, episode_files: Path, scratch_root: Path | None = None):
        # Resolve up front — this is the bug that bit us before. On macOS,
        # /tmp is a symlink to /private/tmp, so unresolved paths fail prefix
        # checks against resolved ones.
        self.episode_files = Path(episode_files).resolve()
        scratch = Path(scratch_root) if scratch_root else Path(tempfile.gettempdir())
        self.scratch_root = scratch.resolve()
        self.workdir: Path | None = None
        self._episode_id: str | None = None

    # ---------- lifecycle ----------

    def reset(self) -> Path:
        """Create a fresh workdir with only the curated episode files."""
        self.close()
        self._episode_id = uuid.uuid4().hex[:12]
        # Resolve the new workdir path immediately so all subsequent
        # comparisons use canonical paths.
        self.workdir = (self.scratch_root / f"pref_env_{self._episode_id}").resolve()
        self.workdir.mkdir(parents=True, exist_ok=False)

        for name in EPISODE_FILES:
            src = self.episode_files / name
            if not src.exists():
                raise FileNotFoundError(
                    f"episode file missing from template: {src}")
            shutil.copy2(src, self.workdir / name)

        return self.workdir

    def close(self) -> None:
        """Tear down the workdir. Idempotent."""
        if self.workdir and self.workdir.exists():
            shutil.rmtree(self.workdir, ignore_errors=True)
        self.workdir = None
        self._episode_id = None

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, *_):
        self.close()

    # ---------- exec ----------

    def exec(self, cmd: list[str], timeout_s: int = DEFAULT_TIMEOUT_S) -> ExecResult:
        """Run `cmd` inside the sandbox container, with workdir bind-mounted."""
        if self.workdir is None:
            raise RuntimeError("Sandbox.exec called before reset()")

        docker_cmd = [
            "docker", "run",
            "--rm",
            "--network=none",
            f"--memory={self.MEMORY_LIMIT}",
            f"--cpus={self.CPU_LIMIT}",
            "--read-only",
            "--tmpfs", "/tmp:size=64m",
            "-v", f"{self.workdir}:/workspace:rw",
            "-w", "/workspace",
            "--user", "1000:1000",
            self.IMAGE,
            *cmd,
        ]

        try:
            proc = subprocess.run(
                docker_cmd, capture_output=True, text=True, timeout=timeout_s,
            )
            return ExecResult(proc.returncode, proc.stdout, proc.stderr, False)
        except subprocess.TimeoutExpired as e:
            return ExecResult(
                exit_code=-1,
                stdout=(e.stdout or "") if isinstance(e.stdout, str) else "",
                stderr=(e.stderr or "") if isinstance(e.stderr, str) else "",
                timed_out=True,
            )

    # ---------- file helpers ----------

    def _safe_path(self, relpath: str) -> Path:
        """Resolve relpath under workdir, refusing escapes."""
        if self.workdir is None:
            raise RuntimeError("called before reset()")
        # Reject absolute paths and obvious traversal upfront.
        if Path(relpath).is_absolute():
            raise ValueError(f"absolute paths not allowed: {relpath}")
        candidate = (self.workdir / relpath).resolve()
        # Now both sides are resolved (workdir was resolved at construction
        # time, and we resolve the candidate here). Use Path.is_relative_to
        # rather than string prefix matching — string prefixes have an
        # off-by-one risk: /tmp/foo vs /tmp/foobar.
        if not candidate.is_relative_to(self.workdir):
            raise ValueError(f"path escapes workdir: {relpath}")
        return candidate

    def write_file(self, relpath: str, content: str) -> None:
        path = self._safe_path(relpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def read_file(self, relpath: str) -> str:
        return self._safe_path(relpath).read_text()