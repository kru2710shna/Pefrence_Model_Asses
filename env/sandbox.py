"""
Docker-backed sandbox for running candidate code in isolation.

Design:
- One container per command (cheap with a pre-built image).
- Workdir is bind-mounted so file edits persist across exec calls within an
  episode but reset between episodes (we copy a fresh workdir each reset).
- No network, no host filesystem access outside the workdir, non-root user,
  hard timeout, capped memory and CPU.
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


class Sandbox:
    """Per-episode sandbox. One Sandbox = one episode = one workdir."""

    IMAGE = "preference-env:latest"
    DEFAULT_TIMEOUT_S = 120
    MEMORY_LIMIT = "2g"
    CPU_LIMIT = "1.0"

    def __init__(self, episode_files: Path, scratch_root: Path | None = None):
        """
        episode_files: directory containing the read-only template for each
                       episode (model.py, judge.py, generate.py stub, etc.).
                       Copied into a fresh workdir on every reset().
        scratch_root:  where to put workdirs. Defaults to /tmp.
        """
        self.episode_files = Path(episode_files).resolve()
        self.scratch_root = Path(scratch_root) if scratch_root else Path(tempfile.gettempdir())
        self.workdir: Path | None = None
        self._episode_id: str | None = None

    # ---------- lifecycle ----------

    def reset(self) -> Path:
        """Create a fresh workdir from the template. Returns the workdir path."""
        self.close()
        self._episode_id = uuid.uuid4().hex[:12]
        self.workdir = self.scratch_root / f"pref_env_{self._episode_id}"
        shutil.copytree(self.episode_files, self.workdir)
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
        """
        Run `cmd` inside the sandbox container, with the workdir bind-mounted
        at /workspace. Returns ExecResult; never raises on subprocess errors.
        """
        if self.workdir is None:
            raise RuntimeError("Sandbox.exec called before reset()")

        docker_cmd = [
            "docker", "run",
            "--rm",
            "--network=none",
            f"--memory={self.MEMORY_LIMIT}",
            f"--cpus={self.CPU_LIMIT}",
            "--read-only",                                     # rootfs read-only
            "--tmpfs", "/tmp:size=64m",                        # but /tmp is writable
            "-v", f"{self.workdir}:/workspace:rw",             # workdir is writable
            "-w", "/workspace",
            "--user", "1000:1000",
            self.IMAGE,
            *cmd,
        ]

        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return ExecResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as e:
            # Kill any lingering containers from this episode (best effort).
            subprocess.run(
                ["docker", "ps", "-q", "--filter", f"volume={self.workdir}"],
                capture_output=True, text=True,
            )
            return ExecResult(
                exit_code=-1,
                stdout=(e.stdout or "") if isinstance(e.stdout, str) else "",
                stderr=(e.stderr or "") if isinstance(e.stderr, str) else "",
                timed_out=True,
            )

    # ---------- file helpers ----------
    # The agent will write files via these instead of via shell, to avoid
    # quoting hell with multiline Python source.

    def write_file(self, relpath: str, content: str) -> None:
        if self.workdir is None:
            raise RuntimeError("write_file called before reset()")
        path = (self.workdir / relpath).resolve()
        # Refuse path traversal.
        if not str(path).startswith(str(self.workdir)):
            raise ValueError(f"path escapes workdir: {relpath}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def read_file(self, relpath: str) -> str:
        if self.workdir is None:
            raise RuntimeError("read_file called before reset()")
        path = (self.workdir / relpath).resolve()
        if not str(path).startswith(str(self.workdir)):
            raise ValueError(f"path escapes workdir: {relpath}")
        return path.read_text()