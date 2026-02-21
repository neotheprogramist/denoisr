"""UCI engine subprocess wrapper for match orchestration."""

from __future__ import annotations

import subprocess
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from denoisr.engine.types import EngineConfig, TimeControl


class UCIEngine:
    """Manages a UCI engine as a subprocess with stdin/stdout communication."""

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._lines: list[str] = []
        self._stderr_lines: list[str] = []
        self._line_event = threading.Event()

    def start(self, timeout: float = 10.0) -> None:
        """Launch the engine subprocess and complete UCI handshake."""
        self._process = subprocess.Popen(
            [self._config.command, *self._config.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True,
        )
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop, daemon=True,
        )
        self._stderr_thread.start()

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

    def set_option(self, name: str, value: str) -> None:
        """Send a UCI setoption command and wait for acknowledgement."""
        self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._wait_for("readyok", timeout=10.0)

    def new_game(self) -> None:
        """Signal the start of a new game and wait for acknowledgement."""
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok", timeout=10.0)

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

    def _reader_loop(self) -> None:
        """Persistent thread that reads stdout lines into a buffer."""
        while self._process is not None:
            stdout = self._process.stdout if self._process else None
            if stdout is None:
                break
            line = stdout.readline()
            if not line:
                break
            with self._lock:
                self._lines.append(line.strip())
            self._line_event.set()

    def _stderr_loop(self) -> None:
        """Read stderr into a buffer for crash diagnostics."""
        while self._process is not None:
            stderr = self._process.stderr if self._process else None
            if stderr is None:
                break
            line = stderr.readline()
            if not line:
                break
            with self._lock:
                self._stderr_lines.append(line.rstrip())

    def _get_stderr_tail(self, max_lines: int = 30) -> str:
        """Return the last N stderr lines as a single string."""
        with self._lock:
            return "\n".join(self._stderr_lines[-max_lines:])

    def _wait_for(self, prefix: str, timeout: float) -> str:
        """Read buffered lines until one starts with prefix.

        Raises RuntimeError immediately if the process dies, or
        TimeoutError if the deadline expires.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Fast-fail if the engine process has died
            if self._process is not None and self._process.poll() is not None:
                if self._stderr_thread is not None:
                    self._stderr_thread.join(timeout=1.0)
                stderr_text = self._get_stderr_tail()
                msg = (
                    f"Engine '{self._config.name}' crashed"
                    f" (exit code {self._process.returncode})"
                )
                if stderr_text.strip():
                    msg += f":\n{stderr_text.strip()}"
                raise RuntimeError(msg)

            with self._lock:
                for i, line in enumerate(self._lines):
                    if line.startswith(prefix):
                        self._lines = self._lines[i + 1:]
                        return line
            self._line_event.clear()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._line_event.wait(timeout=min(remaining, 0.1))

        stderr_text = self._get_stderr_tail()
        msg = f"Timeout waiting for '{prefix}' from {self._config.name}"
        if stderr_text.strip():
            msg += f"\nstderr:\n{stderr_text.strip()}"
        raise TimeoutError(msg)

    def __enter__(self) -> UCIEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.quit()
