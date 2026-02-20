"""UCI engine subprocess wrapper for match orchestration."""

from __future__ import annotations

import concurrent.futures
import subprocess
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from denoisr.gui.types import EngineConfig, TimeControl


class UCIEngine:
    """Manages a UCI engine as a subprocess with stdin/stdout communication."""

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def start(self, timeout: float = 10.0) -> None:
        """Launch the engine subprocess and complete UCI handshake."""
        self._process = subprocess.Popen(
            [self._config.command, *self._config.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok", timeout=timeout)
        self._send("isready")
        self._wait_for("readyok", timeout=timeout)

    def is_alive(self) -> bool:
        """Check if the engine process is running."""
        return self._process is not None and self._process.poll() is None

    def set_position(self, fen: str | None, moves: list[str]) -> None:
        """Send a position command to the engine."""
        if fen is not None:
            cmd = f"position fen {fen}"
        else:
            cmd = "position startpos"
        if moves:
            cmd += " moves " + " ".join(moves)
        self._send(cmd)

    def go(
        self,
        time_control: TimeControl,
        wtime_ms: int,
        btime_ms: int,
        timeout: float = 60.0,
    ) -> str:
        """Send go command and return the bestmove UCI string."""
        winc = int(time_control.increment * 1000)
        binc = winc
        cmd = f"go wtime {wtime_ms} btime {btime_ms} winc {winc} binc {binc}"
        self._send(cmd)
        response = self._wait_for("bestmove", timeout=timeout)
        parts = response.split()
        return parts[1]

    def quit(self) -> None:
        """Send quit and terminate the engine process."""
        if self._process is not None:
            try:
                self._send("quit")
                self._process.wait(timeout=5.0)
            except (BrokenPipeError, subprocess.TimeoutExpired, OSError):
                self._process.kill()
            finally:
                self._process = None

    def _send(self, command: str) -> None:
        with self._lock:
            if self._process is None or self._process.stdin is None:
                return
            self._process.stdin.write(command + "\n")
            self._process.stdin.flush()

    def _wait_for(self, prefix: str, timeout: float) -> str:
        """Read stdout lines until one starts with prefix. Raises TimeoutError."""
        if self._process is None or self._process.stdout is None:
            msg = "Engine process not running"
            raise RuntimeError(msg)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._readline(timeout=deadline - time.monotonic())
            if line is None:
                continue
            if line.startswith(prefix):
                return line
        msg = f"Timeout waiting for '{prefix}' from {self._config.name}"
        raise TimeoutError(msg)

    def _readline(self, timeout: float) -> str | None:
        """Read one line from stdout with timeout."""
        if self._process is None or self._process.stdout is None:
            return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._process.stdout.readline)
            try:
                line = future.result(timeout=max(timeout, 0.1))
                return line.strip() if line else None
            except concurrent.futures.TimeoutError:
                return None

    def __enter__(self) -> UCIEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.quit()
