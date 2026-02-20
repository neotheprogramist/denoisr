# Chess GUI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the external CuteChess dependency with a built-in Tkinter chess GUI that supports human-vs-engine play and engine-vs-engine benchmarking with Elo/SPRT computation.

**Architecture:** Three layers: (1) `elo.py` — pure math for Elo/SPRT, (2) `match_engine.py` — headless UCI subprocess orchestration using `elo.py`, (3) `board_widget.py` + `app.py` — Tkinter GUI composing the board and match engine. Each layer is independently testable.

**Tech Stack:** Python stdlib only (`tkinter`, `math`, `subprocess`, `threading`, `queue`). Uses existing `chess` library for board state. No new dependencies.

**Scope note:** The MLX engine (`mlx_engine.py`, added recently) lacks a UCI wrapper and uses safetensors format. The GUI v1 launches engines as UCI subprocesses via `denoisr-play`, so MLX is out of scope. The `denoisr-export-mlx` script is also unaffected.

**Design doc:** `docs/plans/2026-02-19-chess-gui-design.md`

---

### Task 1: Elo + SPRT math module

**Files:**
- Create: `src/denoisr/gui/__init__.py`
- Create: `src/denoisr/gui/elo.py`
- Create: `tests/test_gui/__init__.py`
- Create: `tests/test_gui/test_elo.py`

**Step 1: Create the gui package with empty `__init__.py` files**

```python
# src/denoisr/gui/__init__.py
# (empty)
```

```python
# tests/test_gui/__init__.py
# (empty)
```

**Step 2: Write failing tests for Elo computation**

```python
# tests/test_gui/test_elo.py
import math

from denoisr.gui.elo import compute_elo, likelihood_of_superiority, sprt_test


class TestComputeElo:
    def test_even_score_is_zero_elo(self) -> None:
        elo, error = compute_elo(wins=50, draws=0, losses=50)
        assert abs(elo) < 0.1

    def test_all_wins_is_large_positive(self) -> None:
        elo, _error = compute_elo(wins=100, draws=0, losses=0)
        assert elo == float("inf")

    def test_all_losses_is_large_negative(self) -> None:
        elo, _error = compute_elo(wins=0, draws=0, losses=100)
        assert elo == float("-inf")

    def test_75_percent_score(self) -> None:
        # 75% score -> Elo ~191 (exact: -400*log10(1/0.75 - 1))
        elo, _error = compute_elo(wins=75, draws=0, losses=25)
        assert abs(elo - 190.85) < 1.0

    def test_all_draws_is_zero_elo(self) -> None:
        elo, _error = compute_elo(wins=0, draws=100, losses=0)
        assert abs(elo) < 0.1

    def test_error_decreases_with_more_games(self) -> None:
        _, error_small = compute_elo(wins=7, draws=3, losses=10)
        _, error_large = compute_elo(wins=70, draws=30, losses=100)
        assert error_large < error_small

    def test_zero_games_returns_zero(self) -> None:
        elo, error = compute_elo(wins=0, draws=0, losses=0)
        assert elo == 0.0
        assert error == 0.0


class TestLikelihoodOfSuperiority:
    def test_even_score_is_near_50(self) -> None:
        los = likelihood_of_superiority(wins=50, draws=0, losses=50)
        assert abs(los - 50.0) < 1.0

    def test_strong_winner_is_near_100(self) -> None:
        los = likelihood_of_superiority(wins=90, draws=5, losses=5)
        assert los > 99.0

    def test_strong_loser_is_near_0(self) -> None:
        los = likelihood_of_superiority(wins=5, draws=5, losses=90)
        assert los < 1.0

    def test_zero_games_returns_50(self) -> None:
        los = likelihood_of_superiority(wins=0, draws=0, losses=0)
        assert abs(los - 50.0) < 0.1


class TestSprtTest:
    def test_clear_h1_accepted(self) -> None:
        # Overwhelming wins — should accept H1 (engine is stronger)
        result = sprt_test(wins=90, draws=5, losses=5, elo0=0.0, elo1=50.0)
        assert result == "H1"

    def test_clear_h0_accepted(self) -> None:
        # Even result — should accept H0 (no improvement)
        result = sprt_test(wins=50, draws=0, losses=50, elo0=0.0, elo1=50.0)
        assert result == "H0"

    def test_inconclusive_returns_none(self) -> None:
        # Too few games — can't decide
        result = sprt_test(wins=3, draws=1, losses=2, elo0=0.0, elo1=50.0)
        assert result is None

    def test_zero_games_returns_none(self) -> None:
        result = sprt_test(wins=0, draws=0, losses=0, elo0=0.0, elo1=50.0)
        assert result is None
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_gui/test_elo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'denoisr.gui'`

**Step 4: Implement `elo.py`**

```python
# src/denoisr/gui/elo.py
"""Elo rating and SPRT computation for engine-vs-engine matches."""

from __future__ import annotations

import math


def compute_elo(wins: int, draws: int, losses: int) -> tuple[float, float]:
    """Compute Elo difference and 95% confidence error margin.

    Uses the standard logistic model: Elo = -400 * log10(1/score - 1).
    Error margin uses the trinomial variance model with z=1.96.
    """
    total = wins + draws + losses
    if total == 0:
        return (0.0, 0.0)

    score = (wins + draws / 2) / total

    if score <= 0.0:
        return (float("-inf"), 0.0)
    if score >= 1.0:
        return (float("inf"), 0.0)

    elo_diff = -400.0 * math.log10(1.0 / score - 1.0)

    # Trinomial variance of the score
    variance = (
        wins * (1.0 - score) ** 2
        + draws * (0.5 - score) ** 2
        + losses * score**2
    ) / total**2

    # Convert score error to Elo error via derivative
    derivative = 400.0 / (score * (1.0 - score) * math.log(10))
    error_95 = derivative * math.sqrt(variance) * 1.96

    return (elo_diff, error_95)


def likelihood_of_superiority(
    wins: int, draws: int, losses: int
) -> float:
    """Compute LOS (likelihood of superiority) as a percentage.

    Uses normal approximation of the trinomial score distribution.
    Returns a value between 0 and 100.
    """
    total = wins + draws + losses
    if total == 0:
        return 50.0

    score = (wins + draws / 2) / total

    if score <= 0.0:
        return 0.0
    if score >= 1.0:
        return 100.0

    variance = (
        wins * (1.0 - score) ** 2
        + draws * (0.5 - score) ** 2
        + losses * score**2
    ) / total**2

    if variance <= 0.0:
        return 50.0

    z = (score - 0.5) / math.sqrt(variance)
    los = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
    return los * 100.0


def sprt_test(
    wins: int,
    draws: int,
    losses: int,
    elo0: float,
    elo1: float,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> str | None:
    """Sequential Probability Ratio Test.

    Returns "H0" if null hypothesis accepted (Elo diff <= elo0),
    "H1" if alternative accepted (Elo diff >= elo1), or None to continue.
    Uses the binomial approximation of the log-likelihood ratio.
    """
    total = wins + draws + losses
    if total == 0:
        return None

    score = (wins + draws / 2) / total
    if score <= 0.0 or score >= 1.0:
        return None

    # SPRT decision boundaries (Wald)
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)

    # Expected scores under H0 and H1
    p0 = 1.0 / (1.0 + 10.0 ** (-elo0 / 400.0))
    p1 = 1.0 / (1.0 + 10.0 ** (-elo1 / 400.0))

    # Log-likelihood ratio (binomial model)
    llr = total * (
        score * math.log(p1 / p0)
        + (1.0 - score) * math.log((1.0 - p1) / (1.0 - p0))
    )

    if llr >= upper:
        return "H1"
    if llr <= lower:
        return "H0"
    return None
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_gui/test_elo.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/denoisr/gui/__init__.py src/denoisr/gui/elo.py tests/test_gui/__init__.py tests/test_gui/test_elo.py
git commit -m "feat: add Elo and SPRT computation module"
```

---

### Task 2: GUI data types

**Files:**
- Create: `src/denoisr/gui/types.py`
- Create: `tests/test_gui/test_gui_types.py`

**Step 1: Write failing tests for GUI data types**

```python
# tests/test_gui/test_gui_types.py
import pytest

from denoisr.gui.types import (
    EngineConfig,
    GameOutcome,
    GameResult,
    MatchConfig,
    TimeControl,
)


class TestTimeControl:
    def test_create(self) -> None:
        tc = TimeControl(base_seconds=10.0, increment=0.1)
        assert tc.base_seconds == 10.0
        assert tc.increment == 0.1

    def test_frozen(self) -> None:
        tc = TimeControl(base_seconds=10.0, increment=0.1)
        with pytest.raises(AttributeError):
            tc.base_seconds = 5.0  # type: ignore[misc]

    def test_rejects_negative_time(self) -> None:
        with pytest.raises(ValueError, match="base_seconds"):
            TimeControl(base_seconds=-1.0, increment=0.1)

    def test_rejects_negative_increment(self) -> None:
        with pytest.raises(ValueError, match="increment"):
            TimeControl(base_seconds=10.0, increment=-0.5)


class TestEngineConfig:
    def test_create(self) -> None:
        ec = EngineConfig(
            command="uv",
            args=("run", "denoisr-play", "--checkpoint", "model.pt"),
            name="Denoisr",
        )
        assert ec.command == "uv"
        assert ec.args == ("run", "denoisr-play", "--checkpoint", "model.pt")

    def test_frozen(self) -> None:
        ec = EngineConfig(command="uv", args=(), name="test")
        with pytest.raises(AttributeError):
            ec.name = "other"  # type: ignore[misc]


class TestGameResult:
    def test_create(self) -> None:
        gr = GameResult(
            moves=("e2e4", "e7e5"),
            result="1-0",
            reason="checkmate",
            engine1_color="white",
        )
        assert gr.result == "1-0"
        assert gr.reason == "checkmate"

    def test_rejects_invalid_result(self) -> None:
        with pytest.raises(ValueError, match="result"):
            GameResult(
                moves=(),
                result="2-0",
                reason="checkmate",
                engine1_color="white",
            )

    def test_rejects_invalid_color(self) -> None:
        with pytest.raises(ValueError, match="engine1_color"):
            GameResult(
                moves=(),
                result="1-0",
                reason="checkmate",
                engine1_color="red",
            )


class TestMatchConfig:
    def test_create(self) -> None:
        e1 = EngineConfig(command="eng1", args=(), name="Engine 1")
        e2 = EngineConfig(command="eng2", args=(), name="Engine 2")
        tc = TimeControl(base_seconds=10.0, increment=0.1)
        mc = MatchConfig(engine1=e1, engine2=e2, games=100, time_control=tc)
        assert mc.games == 100
        assert mc.concurrency == 1

    def test_rejects_zero_games(self) -> None:
        e1 = EngineConfig(command="eng1", args=(), name="E1")
        e2 = EngineConfig(command="eng2", args=(), name="E2")
        tc = TimeControl(base_seconds=10.0, increment=0.1)
        with pytest.raises(ValueError, match="games"):
            MatchConfig(engine1=e1, engine2=e2, games=0, time_control=tc)


class TestGameOutcome:
    def test_values(self) -> None:
        assert GameOutcome.WIN.name == "WIN"
        assert GameOutcome.DRAW.name == "DRAW"
        assert GameOutcome.LOSS.name == "LOSS"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gui/test_gui_types.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement `types.py`**

```python
# src/denoisr/gui/types.py
"""Data types for the chess GUI and match engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


@dataclass(frozen=True)
class TimeControl:
    base_seconds: float
    increment: float

    def __post_init__(self) -> None:
        if self.base_seconds < 0:
            raise ValueError(
                f"base_seconds must be >= 0, got {self.base_seconds}"
            )
        if self.increment < 0:
            raise ValueError(
                f"increment must be >= 0, got {self.increment}"
            )


@dataclass(frozen=True)
class EngineConfig:
    command: str
    args: tuple[str, ...]
    name: str


@dataclass(frozen=True)
class GameResult:
    moves: tuple[str, ...]
    result: str
    reason: str
    engine1_color: str

    def __post_init__(self) -> None:
        valid_results = {"1-0", "0-1", "1/2-1/2"}
        if self.result not in valid_results:
            raise ValueError(
                f"result must be one of {valid_results}, got {self.result!r}"
            )
        if self.engine1_color not in {"white", "black"}:
            raise ValueError(
                f"engine1_color must be 'white' or 'black', "
                f"got {self.engine1_color!r}"
            )


@dataclass(frozen=True)
class MatchConfig:
    engine1: EngineConfig
    engine2: EngineConfig
    games: int
    time_control: TimeControl
    concurrency: int = 1

    def __post_init__(self) -> None:
        if self.games <= 0:
            raise ValueError(f"games must be > 0, got {self.games}")
        if self.concurrency <= 0:
            raise ValueError(
                f"concurrency must be > 0, got {self.concurrency}"
            )


class GameOutcome(Enum):
    WIN = auto()
    DRAW = auto()
    LOSS = auto()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gui/test_gui_types.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/denoisr/gui/types.py tests/test_gui/test_gui_types.py
git commit -m "feat: add GUI data types (EngineConfig, TimeControl, GameResult, MatchConfig)"
```

---

### Task 3: UCI engine subprocess wrapper

**Files:**
- Create: `src/denoisr/gui/uci_engine.py`
- Create: `tests/test_gui/test_uci_engine.py`
- Create: `tests/test_gui/mock_engine.py` (helper script)

The `UCIEngine` class wraps a subprocess running a UCI engine. To test it without needing an actual chess engine, we create a tiny mock engine script that speaks minimal UCI.

**Step 1: Create the mock engine test helper**

```python
# tests/test_gui/mock_engine.py
"""Minimal UCI engine for testing. Responds to uci/isready/go with fixed moves."""

import sys

import chess


def main() -> None:
    board = chess.Board()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        if line == "uci":
            print("id name MockEngine")
            print("id author test")
            print("uciok")

        elif line == "isready":
            print("readyok")

        elif line.startswith("position"):
            parts = line.split()
            if "startpos" in parts:
                board = chess.Board()
            elif "fen" in parts:
                fen_idx = parts.index("fen")
                if "moves" in parts:
                    moves_idx = parts.index("moves")
                    fen = " ".join(parts[fen_idx + 1 : moves_idx])
                else:
                    moves_idx = len(parts)
                    fen = " ".join(parts[fen_idx + 1 :])
                board = chess.Board(fen)
            if "moves" in parts:
                moves_idx = parts.index("moves")
                for uci in parts[moves_idx + 1 :]:
                    board.push_uci(uci)

        elif line.startswith("go"):
            # Always play the first legal move
            move = next(iter(board.legal_moves))
            print(f"bestmove {move.uci()}")

        elif line == "quit":
            break

        sys.stdout.flush()


if __name__ == "__main__":
    main()
```

**Step 2: Write failing tests for UCIEngine**

```python
# tests/test_gui/test_uci_engine.py
import sys
from pathlib import Path

import chess
import pytest

from denoisr.gui.types import EngineConfig, TimeControl
from denoisr.gui.uci_engine import UCIEngine

MOCK_ENGINE = str(Path(__file__).parent / "mock_engine.py")


def _mock_config(name: str = "MockEngine") -> EngineConfig:
    return EngineConfig(
        command=sys.executable,
        args=(MOCK_ENGINE,),
        name=name,
    )


class TestUCIEngine:
    def test_start_and_quit(self) -> None:
        engine = UCIEngine(_mock_config())
        engine.start()
        assert engine.is_alive()
        engine.quit()
        assert not engine.is_alive()

    def test_go_returns_legal_move(self) -> None:
        engine = UCIEngine(_mock_config())
        engine.start()
        engine.set_position(fen=None, moves=[])
        move_uci = engine.go(
            time_control=TimeControl(base_seconds=10.0, increment=0.1),
            wtime_ms=10000,
            btime_ms=10000,
        )
        board = chess.Board()
        move = chess.Move.from_uci(move_uci)
        assert move in board.legal_moves
        engine.quit()

    def test_set_position_with_moves(self) -> None:
        engine = UCIEngine(_mock_config())
        engine.start()
        engine.set_position(fen=None, moves=["e2e4", "e7e5"])
        move_uci = engine.go(
            time_control=TimeControl(base_seconds=10.0, increment=0.1),
            wtime_ms=10000,
            btime_ms=10000,
        )
        board = chess.Board()
        board.push_uci("e2e4")
        board.push_uci("e7e5")
        move = chess.Move.from_uci(move_uci)
        assert move in board.legal_moves
        engine.quit()

    def test_context_manager(self) -> None:
        config = _mock_config()
        with UCIEngine(config) as engine:
            engine.start()
            assert engine.is_alive()
        assert not engine.is_alive()

    def test_timeout_raises(self) -> None:
        # Create an engine config that points to a nonexistent script
        config = EngineConfig(
            command=sys.executable,
            args=("-c", "import time; time.sleep(60)"),
            name="SlowEngine",
        )
        engine = UCIEngine(config)
        with pytest.raises(TimeoutError):
            engine.start(timeout=0.5)
        engine.quit()
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_gui/test_uci_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 4: Implement `uci_engine.py`**

```python
# src/denoisr/gui/uci_engine.py
"""UCI engine subprocess wrapper for match orchestration."""

from __future__ import annotations

import subprocess
import threading
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

    def set_position(
        self, fen: str | None, moves: list[str]
    ) -> None:
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
        return parts[1]  # "bestmove e2e4" -> "e2e4"

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
        import time

        if self._process is None or self._process.stdout is None:
            msg = "Engine process not running"
            raise RuntimeError(msg)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Use a short readline with implicit timeout via the deadline
            line = self._readline(timeout=deadline - time.monotonic())
            if line is None:
                continue
            if line.startswith(prefix):
                return line
        msg = f"Timeout waiting for '{prefix}' from {self._config.name}"
        raise TimeoutError(msg)

    def _readline(self, timeout: float) -> str | None:
        """Read one line from stdout with timeout."""
        import concurrent.futures

        if self._process is None or self._process.stdout is None:
            return None
        # Use a thread to avoid blocking the caller indefinitely
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
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_gui/test_uci_engine.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/denoisr/gui/uci_engine.py tests/test_gui/test_uci_engine.py tests/test_gui/mock_engine.py
git commit -m "feat: add UCI engine subprocess wrapper"
```

---

### Task 4: Game orchestration (play_game + run_match)

**Files:**
- Create: `src/denoisr/gui/match_engine.py`
- Create: `tests/test_gui/test_match_engine.py`

**Step 1: Write failing tests for play_game and run_match**

```python
# tests/test_gui/test_match_engine.py
import sys
from pathlib import Path

import chess

from denoisr.gui.match_engine import play_game, run_match
from denoisr.gui.types import EngineConfig, MatchConfig, TimeControl
from denoisr.gui.uci_engine import UCIEngine

MOCK_ENGINE = str(Path(__file__).parent / "mock_engine.py")


def _mock_config(name: str = "MockEngine") -> EngineConfig:
    return EngineConfig(
        command=sys.executable,
        args=(MOCK_ENGINE,),
        name=name,
    )


class TestPlayGame:
    def test_game_completes(self) -> None:
        tc = TimeControl(base_seconds=60.0, increment=0.0)
        e1_config = _mock_config("White")
        e2_config = _mock_config("Black")
        with UCIEngine(e1_config) as white, UCIEngine(e2_config) as black:
            white.start()
            black.start()
            result = play_game(
                white=white,
                black=black,
                time_control=tc,
                max_moves=200,
            )
        assert result.result in {"1-0", "0-1", "1/2-1/2"}
        assert len(result.moves) > 0
        assert result.engine1_color == "white"

    def test_on_move_callback_called(self) -> None:
        tc = TimeControl(base_seconds=60.0, increment=0.0)
        e1_config = _mock_config("White")
        e2_config = _mock_config("Black")
        move_count = 0

        def on_move(board: chess.Board, uci: str) -> None:
            nonlocal move_count
            move_count += 1

        with UCIEngine(e1_config) as white, UCIEngine(e2_config) as black:
            white.start()
            black.start()
            play_game(
                white=white,
                black=black,
                time_control=tc,
                max_moves=200,
                on_move=on_move,
            )
        assert move_count > 0


class TestRunMatch:
    def test_match_completes(self) -> None:
        tc = TimeControl(base_seconds=60.0, increment=0.0)
        config = MatchConfig(
            engine1=_mock_config("Engine1"),
            engine2=_mock_config("Engine2"),
            games=2,
            time_control=tc,
        )
        results = run_match(config, max_moves_per_game=200)
        assert len(results) == 2
        for r in results:
            assert r.result in {"1-0", "0-1", "1/2-1/2"}

    def test_engines_alternate_colors(self) -> None:
        tc = TimeControl(base_seconds=60.0, increment=0.0)
        config = MatchConfig(
            engine1=_mock_config("Engine1"),
            engine2=_mock_config("Engine2"),
            games=2,
            time_control=tc,
        )
        results = run_match(config, max_moves_per_game=200)
        colors = [r.engine1_color for r in results]
        assert colors == ["white", "black"]
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gui/test_match_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement `match_engine.py`**

```python
# src/denoisr/gui/match_engine.py
"""Headless UCI match orchestration — plays games between two engines."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import chess

from denoisr.gui.types import GameResult
from denoisr.gui.uci_engine import UCIEngine

if TYPE_CHECKING:
    from collections.abc import Callable

    from denoisr.gui.types import MatchConfig, TimeControl


def play_game(
    white: UCIEngine,
    black: UCIEngine,
    time_control: TimeControl,
    max_moves: int = 500,
    on_move: Callable[[chess.Board, str], None] | None = None,
    engine1_color: str = "white",
) -> GameResult:
    """Play a single game between two UCI engines.

    Tracks time via wall-clock. Returns a GameResult when the game ends.
    """
    board = chess.Board()
    moves: list[str] = []
    wtime_ms = int(time_control.base_seconds * 1000)
    btime_ms = int(time_control.base_seconds * 1000)

    for _ in range(max_moves):
        if board.is_game_over():
            break

        current = white if board.turn == chess.WHITE else black

        move_list = [m for m in moves]
        current.set_position(fen=None, moves=move_list)

        t0 = time.monotonic()
        uci_move = current.go(
            time_control=time_control,
            wtime_ms=wtime_ms,
            btime_ms=btime_ms,
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # Deduct time
        if board.turn == chess.WHITE:
            wtime_ms = wtime_ms - elapsed_ms + int(
                time_control.increment * 1000
            )
            if wtime_ms <= 0:
                result_str = "0-1"
                return GameResult(
                    moves=tuple(moves),
                    result=result_str,
                    reason="timeout",
                    engine1_color=engine1_color,
                )
        else:
            btime_ms = btime_ms - elapsed_ms + int(
                time_control.increment * 1000
            )
            if btime_ms <= 0:
                result_str = "1-0"
                return GameResult(
                    moves=tuple(moves),
                    result=result_str,
                    reason="timeout",
                    engine1_color=engine1_color,
                )

        board.push_uci(uci_move)
        moves.append(uci_move)

        if on_move is not None:
            on_move(board, uci_move)

    # Determine result
    result_str, reason = _game_outcome(board, len(moves) >= max_moves)
    return GameResult(
        moves=tuple(moves),
        result=result_str,
        reason=reason,
        engine1_color=engine1_color,
    )


def run_match(
    config: MatchConfig,
    max_moves_per_game: int = 500,
    on_game_complete: Callable[[int, GameResult], None] | None = None,
    on_move: Callable[[int, chess.Board, str], None] | None = None,
) -> list[GameResult]:
    """Run a multi-game match between two engines.

    Engines alternate colors each game. Returns all game results.
    """
    results: list[GameResult] = []

    for game_num in range(config.games):
        # Alternate colors: even games engine1=white, odd games engine1=black
        engine1_is_white = game_num % 2 == 0

        if engine1_is_white:
            white_config = config.engine1
            black_config = config.engine2
            e1_color = "white"
        else:
            white_config = config.engine2
            black_config = config.engine1
            e1_color = "black"

        with UCIEngine(white_config) as white, UCIEngine(
            black_config
        ) as black:
            white.start()
            black.start()

            move_cb = None
            if on_move is not None:
                gn = game_num

                def move_cb(
                    board: chess.Board,
                    uci: str,
                    _gn: int = gn,
                ) -> None:
                    on_move(_gn, board, uci)  # type: ignore[misc]

            result = play_game(
                white=white,
                black=black,
                time_control=config.time_control,
                max_moves=max_moves_per_game,
                on_move=move_cb,
                engine1_color=e1_color,
            )

        results.append(result)

        if on_game_complete is not None:
            on_game_complete(game_num, result)

    return results


def _game_outcome(
    board: chess.Board, max_moves_reached: bool
) -> tuple[str, str]:
    """Determine game result string and reason from board state."""
    if board.is_checkmate():
        # The side to move is in checkmate — they lost
        if board.turn == chess.WHITE:
            return ("0-1", "checkmate")
        return ("1-0", "checkmate")
    if board.is_stalemate():
        return ("1/2-1/2", "stalemate")
    if board.is_insufficient_material():
        return ("1/2-1/2", "insufficient_material")
    if board.can_claim_fifty_moves():
        return ("1/2-1/2", "fifty_moves")
    if board.can_claim_threefold_repetition():
        return ("1/2-1/2", "threefold_repetition")
    if max_moves_reached:
        return ("1/2-1/2", "max_moves")
    return ("1/2-1/2", "unknown")
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gui/test_match_engine.py -v --timeout=60`
Expected: All PASS (games between mock engines should complete quickly)

**Step 5: Commit**

```bash
git add src/denoisr/gui/match_engine.py tests/test_gui/test_match_engine.py
git commit -m "feat: add game orchestration (play_game + run_match)"
```

---

### Task 5: Board widget — rendering and coordinate math

**Files:**
- Create: `src/denoisr/gui/board_widget.py`
- Create: `tests/test_gui/test_board_widget.py`

The board widget has two testable parts: (1) coordinate math (square ↔ pixel conversion, piece lookup) which is pure logic, and (2) Tkinter Canvas rendering which needs a display. We test the coordinate math thoroughly; Canvas rendering is verified manually.

**Step 1: Write failing tests for coordinate math and board state**

```python
# tests/test_gui/test_board_widget.py
import chess
import pytest

from denoisr.gui.board_widget import (
    PIECE_SYMBOLS,
    file_rank_to_pixel,
    pixel_to_file_rank,
    square_to_file_rank,
)


class TestCoordinateMath:
    def test_a1_white_orientation(self) -> None:
        """a1 is bottom-left when white is at bottom (flipped=False)."""
        fr = square_to_file_rank(chess.A1, flipped=False)
        assert fr == (0, 7)  # file=0 (left), rank=7 (bottom row)

    def test_h8_white_orientation(self) -> None:
        fr = square_to_file_rank(chess.H8, flipped=False)
        assert fr == (7, 0)  # file=7 (right), rank=0 (top row)

    def test_a1_black_orientation(self) -> None:
        """a1 is top-right when black is at bottom (flipped=True)."""
        fr = square_to_file_rank(chess.A1, flipped=True)
        assert fr == (7, 0)

    def test_file_rank_to_pixel_center(self) -> None:
        sq_size = 60
        x, y = file_rank_to_pixel(0, 0, sq_size)
        assert x == 30  # center of first column
        assert y == 30  # center of first row

    def test_pixel_to_file_rank_roundtrip(self) -> None:
        sq_size = 60
        for f in range(8):
            for r in range(8):
                cx, cy = file_rank_to_pixel(f, r, sq_size)
                rf, rr = pixel_to_file_rank(cx, cy, sq_size)
                assert (rf, rr) == (f, r)

    def test_pixel_outside_board_returns_none(self) -> None:
        result = pixel_to_file_rank(-5, 10, 60)
        assert result is None
        result = pixel_to_file_rank(500, 10, 60)
        assert result is None


class TestPieceSymbols:
    def test_white_king(self) -> None:
        assert PIECE_SYMBOLS[(chess.KING, chess.WHITE)] == "\u2654"

    def test_black_pawn(self) -> None:
        assert PIECE_SYMBOLS[(chess.PAWN, chess.BLACK)] == "\u265f"

    def test_all_pieces_present(self) -> None:
        for color in (chess.WHITE, chess.BLACK):
            for piece_type in range(1, 7):
                assert (piece_type, color) in PIECE_SYMBOLS
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gui/test_board_widget.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement `board_widget.py`**

```python
# src/denoisr/gui/board_widget.py
"""Tkinter Canvas chess board widget with click-click interaction."""

from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING

import chess

if TYPE_CHECKING:
    from collections.abc import Callable

# Colors
LIGHT_SQUARE = "#F0D9B5"
DARK_SQUARE = "#B58863"
SELECTED_COLOR = "#6495ED"
LEGAL_MOVE_COLOR = "#66BB6A"
LAST_MOVE_COLOR = "#FFFF66"

# Unicode chess pieces
PIECE_SYMBOLS: dict[tuple[int, bool], str] = {
    (chess.KING, chess.WHITE): "\u2654",
    (chess.QUEEN, chess.WHITE): "\u2655",
    (chess.ROOK, chess.WHITE): "\u2656",
    (chess.BISHOP, chess.WHITE): "\u2657",
    (chess.KNIGHT, chess.WHITE): "\u2658",
    (chess.PAWN, chess.WHITE): "\u2659",
    (chess.KING, chess.BLACK): "\u265a",
    (chess.QUEEN, chess.BLACK): "\u265b",
    (chess.ROOK, chess.BLACK): "\u265c",
    (chess.BISHOP, chess.BLACK): "\u265d",
    (chess.KNIGHT, chess.BLACK): "\u265e",
    (chess.PAWN, chess.BLACK): "\u265f",
}


def square_to_file_rank(
    square: int, flipped: bool
) -> tuple[int, int]:
    """Convert chess square (0-63) to (file, rank) in display coordinates.

    file=0 is left column, rank=0 is top row.
    When flipped=False (white at bottom): a1=(0,7), h8=(7,0).
    When flipped=True (black at bottom): a1=(7,0), h8=(0,7).
    """
    f = chess.square_file(square)
    r = chess.square_rank(square)
    if flipped:
        return (7 - f, r)
    return (f, 7 - r)


def file_rank_to_pixel(
    file: int, rank: int, square_size: int
) -> tuple[int, int]:
    """Convert display (file, rank) to pixel center coordinates."""
    x = file * square_size + square_size // 2
    y = rank * square_size + square_size // 2
    return (x, y)


def pixel_to_file_rank(
    x: int, y: int, square_size: int
) -> tuple[int, int] | None:
    """Convert pixel coordinates to display (file, rank). None if outside."""
    f = x // square_size
    r = y // square_size
    if 0 <= f < 8 and 0 <= r < 8:
        return (f, r)
    return None


def _file_rank_to_square(
    file: int, rank: int, flipped: bool
) -> int:
    """Convert display (file, rank) back to chess square index."""
    if flipped:
        return chess.square(7 - file, rank)
    return chess.square(file, 7 - rank)


class BoardWidget(tk.Canvas):
    """Interactive chess board rendered on a Tkinter Canvas."""

    def __init__(
        self, parent: tk.Widget, square_size: int = 60
    ) -> None:
        self._sq = square_size
        size = square_size * 8
        super().__init__(
            parent, width=size, height=size, highlightthickness=0
        )

        self._board = chess.Board()
        self._flipped = False
        self._interactive = False
        self._on_move_cb: Callable[[chess.Move], None] | None = None
        self._selected_square: int | None = None
        self._last_move: chess.Move | None = None

        self.bind("<Button-1>", self._on_click)
        self._draw()

    def set_board(self, board: chess.Board) -> None:
        """Update the displayed position."""
        self._board = board.copy()
        self._selected_square = None
        self._draw()

    def set_interactive(self, enabled: bool) -> None:
        """Enable/disable human move input."""
        self._interactive = enabled
        if not enabled:
            self._selected_square = None
            self._draw()

    def set_on_move(
        self, callback: Callable[[chess.Move], None]
    ) -> None:
        """Register callback when human makes a move."""
        self._on_move_cb = callback

    def flip(self) -> None:
        """Flip board orientation."""
        self._flipped = not self._flipped
        self._draw()

    def highlight_last_move(self, move: chess.Move) -> None:
        """Highlight the last move played."""
        self._last_move = move
        self._draw()

    def _on_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if not self._interactive:
            return

        fr = pixel_to_file_rank(event.x, event.y, self._sq)
        if fr is None:
            return

        clicked_sq = _file_rank_to_square(fr[0], fr[1], self._flipped)

        if self._selected_square is None:
            # Select a piece
            piece = self._board.piece_at(clicked_sq)
            if piece is not None and piece.color == self._board.turn:
                self._selected_square = clicked_sq
                self._draw()
        else:
            # Try to make a move
            move = chess.Move(self._selected_square, clicked_sq)

            # Check for pawn promotion
            piece = self._board.piece_at(self._selected_square)
            if piece is not None and piece.piece_type == chess.PAWN:
                dest_rank = chess.square_rank(clicked_sq)
                if dest_rank in (0, 7):
                    promo = self._ask_promotion()
                    if promo is not None:
                        move = chess.Move(
                            self._selected_square,
                            clicked_sq,
                            promotion=promo,
                        )

            if move in self._board.legal_moves:
                self._selected_square = None
                self._last_move = move
                if self._on_move_cb is not None:
                    self._on_move_cb(move)
            else:
                # Deselect or reselect
                new_piece = self._board.piece_at(clicked_sq)
                if (
                    new_piece is not None
                    and new_piece.color == self._board.turn
                ):
                    self._selected_square = clicked_sq
                else:
                    self._selected_square = None
            self._draw()

    def _ask_promotion(self) -> int | None:
        """Show a promotion dialog. Returns piece type or None."""
        dialog = tk.Toplevel(self)
        dialog.title("Promote to")
        dialog.resizable(False, False)
        dialog.grab_set()

        result: list[int | None] = [None]

        for piece_type, label in [
            (chess.QUEEN, "Q"),
            (chess.ROOK, "R"),
            (chess.BISHOP, "B"),
            (chess.KNIGHT, "N"),
        ]:
            pt = piece_type

            def choose(p: int = pt) -> None:
                result[0] = p
                dialog.destroy()

            tk.Button(
                dialog, text=label, width=4, command=choose
            ).pack(side=tk.LEFT, padx=2, pady=4)

        dialog.wait_window()
        return result[0]

    def _draw(self) -> None:
        """Redraw the entire board."""
        self.delete("all")
        sq = self._sq

        # Draw squares
        for f in range(8):
            for r in range(8):
                x0 = f * sq
                y0 = r * sq
                is_light = (f + r) % 2 == 0
                color = LIGHT_SQUARE if is_light else DARK_SQUARE

                # Last move highlight
                if self._last_move is not None:
                    sq_here = _file_rank_to_square(
                        f, r, self._flipped
                    )
                    if sq_here in (
                        self._last_move.from_square,
                        self._last_move.to_square,
                    ):
                        color = LAST_MOVE_COLOR

                self.create_rectangle(
                    x0, y0, x0 + sq, y0 + sq, fill=color, outline=""
                )

        # Selected square highlight
        if self._selected_square is not None:
            sf, sr = square_to_file_rank(
                self._selected_square, self._flipped
            )
            self.create_rectangle(
                sf * sq,
                sr * sq,
                sf * sq + sq,
                sr * sq + sq,
                outline=SELECTED_COLOR,
                width=3,
            )

            # Legal move indicators
            for move in self._board.legal_moves:
                if move.from_square == self._selected_square:
                    tf, tr = square_to_file_rank(
                        move.to_square, self._flipped
                    )
                    cx, cy = file_rank_to_pixel(tf, tr, sq)
                    radius = sq // 6
                    self.create_oval(
                        cx - radius,
                        cy - radius,
                        cx + radius,
                        cy + radius,
                        fill=LEGAL_MOVE_COLOR,
                        outline="",
                        stipple="gray50",
                    )

        # Draw pieces
        for square in chess.SQUARES:
            piece = self._board.piece_at(square)
            if piece is None:
                continue
            symbol = PIECE_SYMBOLS.get(
                (piece.piece_type, piece.color)
            )
            if symbol is None:
                continue
            df, dr = square_to_file_rank(square, self._flipped)
            cx, cy = file_rank_to_pixel(df, dr, sq)
            self.create_text(
                cx,
                cy,
                text=symbol,
                font=("Arial", sq // 2),
                anchor="center",
            )

        # File/rank labels
        label_font = ("Arial", sq // 6)
        for i in range(8):
            # File labels (a-h) at bottom
            file_idx = (7 - i) if self._flipped else i
            self.create_text(
                i * sq + sq - 4,
                8 * sq - 4,
                text=chr(ord("a") + file_idx),
                font=label_font,
                anchor="se",
                fill="#666666",
            )
            # Rank labels (1-8) on left
            rank_idx = (i + 1) if self._flipped else (8 - i)
            self.create_text(
                4,
                i * sq + 4,
                text=str(rank_idx),
                font=label_font,
                anchor="nw",
                fill="#666666",
            )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gui/test_board_widget.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/denoisr/gui/board_widget.py tests/test_gui/test_board_widget.py
git commit -m "feat: add chess board widget with coordinate math and rendering"
```

---

### Task 6: GUI application — Play mode

**Files:**
- Create: `src/denoisr/gui/app.py`

This is the main application window. Play mode lets a human play against a UCI engine. No automated tests for this task — it's pure Tkinter wiring verified by manual interaction.

**Step 1: Implement `app.py` with play mode**

```python
# src/denoisr/gui/app.py
"""Main Denoisr Chess GUI application."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from typing import TYPE_CHECKING

import chess

from denoisr.gui.board_widget import BoardWidget
from denoisr.gui.types import EngineConfig, TimeControl
from denoisr.gui.uci_engine import UCIEngine

if TYPE_CHECKING:
    pass


class DenoisrApp:
    """Main GUI application with Play and Match modes."""

    def __init__(self) -> None:
        self._root = tk.Tk()
        self._root.title("Denoisr Chess")
        self._root.resizable(False, False)

        self._board = chess.Board()
        self._engine: UCIEngine | None = None
        self._engine_thread: threading.Thread | None = None
        self._move_queue: queue.Queue[chess.Move | str] = queue.Queue()
        self._moves: list[str] = []
        self._human_color = chess.WHITE

        self._build_ui()
        self._poll_queue()

    def run(self) -> None:
        """Start the Tkinter main loop."""
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.mainloop()

    def _build_ui(self) -> None:
        # Main frame
        main = ttk.Frame(self._root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")

        # Left: board
        self._board_widget = BoardWidget(main, square_size=60)
        self._board_widget.grid(row=0, column=0, rowspan=2, padx=(0, 8))
        self._board_widget.set_on_move(self._on_human_move)

        # Right: controls
        ctrl = ttk.LabelFrame(main, text="Controls", padding=8)
        ctrl.grid(row=0, column=1, sticky="new")

        # Mode selector
        self._mode_var = tk.StringVar(value="play")
        ttk.Radiobutton(
            ctrl, text="Play", variable=self._mode_var, value="play",
            command=self._on_mode_change,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            ctrl, text="Match", variable=self._mode_var, value="match",
            command=self._on_mode_change,
        ).grid(row=0, column=1, sticky="w")

        # Checkpoint
        ttk.Label(ctrl, text="Checkpoint:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self._ckpt_var = tk.StringVar()
        ckpt_frame = ttk.Frame(ctrl)
        ckpt_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Entry(ckpt_frame, textvariable=self._ckpt_var, width=20).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(ckpt_frame, text="Browse", command=self._browse_ckpt).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        # Engine mode
        ttk.Label(ctrl, text="Engine mode:").grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )
        self._engine_mode_var = tk.StringVar(value="single")
        ttk.Combobox(
            ctrl,
            textvariable=self._engine_mode_var,
            values=["single", "diffusion"],
            state="readonly",
            width=12,
        ).grid(row=3, column=1, sticky="w", pady=(8, 0))

        # Human color
        ttk.Label(ctrl, text="Play as:").grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )
        self._color_var = tk.StringVar(value="white")
        ttk.Combobox(
            ctrl,
            textvariable=self._color_var,
            values=["white", "black"],
            state="readonly",
            width=12,
        ).grid(row=4, column=1, sticky="w", pady=(8, 0))

        # Buttons
        btn_frame = ttk.Frame(ctrl)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=(12, 0))
        ttk.Button(
            btn_frame, text="New Game", command=self._new_game
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            btn_frame, text="Stop", command=self._stop_game
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            btn_frame, text="Flip", command=self._board_widget.flip
        ).pack(side=tk.LEFT, padx=2)

        # Bottom: status
        status_frame = ttk.LabelFrame(main, text="Game", padding=8)
        status_frame.grid(
            row=1, column=1, sticky="nsew", pady=(8, 0)
        )

        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(
            status_frame, textvariable=self._status_var, wraplength=200
        ).pack(anchor="w")

        ttk.Label(status_frame, text="Moves:").pack(
            anchor="w", pady=(8, 0)
        )
        self._moves_text = tk.Text(
            status_frame, width=28, height=12, wrap=tk.WORD, state=tk.DISABLED
        )
        self._moves_text.pack(fill=tk.BOTH, expand=True)

    def _browse_ckpt(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self._ckpt_var.set(path)

    def _on_mode_change(self) -> None:
        # Mode switching handled in future tasks
        pass

    def _new_game(self) -> None:
        self._stop_game()
        self._board = chess.Board()
        self._moves = []
        self._human_color = (
            chess.WHITE if self._color_var.get() == "white" else chess.BLACK
        )

        self._board_widget.set_board(self._board)
        self._update_moves_text()
        self._status_var.set("White to move")

        ckpt = self._ckpt_var.get().strip()
        if not ckpt:
            self._status_var.set("Set a checkpoint first")
            return

        # Build engine config
        mode = self._engine_mode_var.get()
        args = ["run", "denoisr-play", "--checkpoint", ckpt, "--mode", mode]
        config = EngineConfig(command="uv", args=tuple(args), name="Denoisr")

        try:
            self._engine = UCIEngine(config)
            self._engine.start(timeout=30.0)
        except (TimeoutError, OSError) as e:
            self._status_var.set(f"Engine failed: {e}")
            self._engine = None
            return

        # If human is black, engine moves first
        if self._human_color == chess.BLACK:
            self._board_widget.set_interactive(False)
            self._request_engine_move()
        else:
            self._board_widget.set_interactive(True)

    def _stop_game(self) -> None:
        self._board_widget.set_interactive(False)
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def _on_human_move(self, move: chess.Move) -> None:
        self._board.push(move)
        self._moves.append(move.uci())
        self._board_widget.set_board(self._board)
        self._board_widget.highlight_last_move(move)
        self._board_widget.set_interactive(False)
        self._update_moves_text()

        if self._board.is_game_over():
            self._show_game_over()
            return

        self._status_var.set("Engine thinking...")
        self._request_engine_move()

    def _request_engine_move(self) -> None:
        if self._engine is None:
            return
        tc = TimeControl(base_seconds=60.0, increment=0.0)

        def _think() -> None:
            assert self._engine is not None
            try:
                self._engine.set_position(fen=None, moves=list(self._moves))
                uci = self._engine.go(
                    time_control=tc, wtime_ms=60000, btime_ms=60000
                )
                self._move_queue.put(chess.Move.from_uci(uci))
            except Exception as e:
                self._move_queue.put(f"error:{e}")

        self._engine_thread = threading.Thread(target=_think, daemon=True)
        self._engine_thread.start()

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self._move_queue.get_nowait()
                if isinstance(item, chess.Move):
                    self._apply_engine_move(item)
                elif isinstance(item, str) and item.startswith("error:"):
                    self._status_var.set(item)
        except queue.Empty:
            pass
        self._root.after(50, self._poll_queue)

    def _apply_engine_move(self, move: chess.Move) -> None:
        self._board.push(move)
        self._moves.append(move.uci())
        self._board_widget.set_board(self._board)
        self._board_widget.highlight_last_move(move)
        self._update_moves_text()

        if self._board.is_game_over():
            self._show_game_over()
            return

        turn = "White" if self._board.turn == chess.WHITE else "Black"
        self._status_var.set(f"{turn} to move")
        self._board_widget.set_interactive(True)

    def _show_game_over(self) -> None:
        result = self._board.result()
        self._status_var.set(f"Game over: {result}")
        self._board_widget.set_interactive(False)

    def _update_moves_text(self) -> None:
        self._moves_text.config(state=tk.NORMAL)
        self._moves_text.delete("1.0", tk.END)
        # Format as numbered move pairs
        text_parts: list[str] = []
        for i in range(0, len(self._moves), 2):
            num = i // 2 + 1
            white_move = self._moves[i]
            if i + 1 < len(self._moves):
                black_move = self._moves[i + 1]
                text_parts.append(f"{num}. {white_move} {black_move}")
            else:
                text_parts.append(f"{num}. {white_move}")
        self._moves_text.insert("1.0", " ".join(text_parts))
        self._moves_text.config(state=tk.DISABLED)

    def _on_close(self) -> None:
        self._stop_game()
        self._root.destroy()
```

**Step 2: Manually verify by running**

Run: `uv run python -c "from denoisr.gui.app import DenoisrApp; DenoisrApp().run()"`
Expected: Window opens with chess board, controls panel, status area. Board renders with Unicode pieces. Click interaction works on squares.

**Step 3: Commit**

```bash
git add src/denoisr/gui/app.py
git commit -m "feat: add GUI application with play mode"
```

---

### Task 7: GUI application — Match mode

**Files:**
- Modify: `src/denoisr/gui/app.py`

Add match mode: engine-vs-engine with live board visualization, Elo/SPRT tracking, adjustable speed.

**Step 1: Add match mode UI and logic to `app.py`**

Add the following to the `DenoisrApp` class. The exact changes:

1. Add match-mode controls to the right panel (opponent config, games count, time control, speed slider)
2. Add match status display (W/D/L, Elo, SPRT)
3. Add `_start_match()` method that runs `run_match()` in a background thread
4. Add move visualization callback that queues board updates

The key additions:

In `_build_ui`, after the existing controls, add a match-mode panel:

```python
# --- Match mode controls (initially hidden) ---
self._match_frame = ttk.LabelFrame(main, text="Match Settings", padding=8)

# Opponent checkpoint
ttk.Label(self._match_frame, text="Opponent:").grid(row=0, column=0, sticky="w")
self._opp_var = tk.StringVar(value="stockfish")
ttk.Entry(self._match_frame, textvariable=self._opp_var, width=20).grid(
    row=0, column=1, sticky="ew"
)

# Games
ttk.Label(self._match_frame, text="Games:").grid(row=1, column=0, sticky="w", pady=(4,0))
self._games_var = tk.IntVar(value=100)
ttk.Spinbox(self._match_frame, from_=2, to=10000, textvariable=self._games_var, width=8).grid(
    row=1, column=1, sticky="w", pady=(4,0)
)

# Time control
ttk.Label(self._match_frame, text="Time (sec):").grid(row=2, column=0, sticky="w", pady=(4,0))
self._tc_base_var = tk.DoubleVar(value=10.0)
ttk.Spinbox(self._match_frame, from_=1, to=3600, textvariable=self._tc_base_var, width=8).grid(
    row=2, column=1, sticky="w", pady=(4,0)
)

ttk.Label(self._match_frame, text="Increment:").grid(row=3, column=0, sticky="w", pady=(4,0))
self._tc_inc_var = tk.DoubleVar(value=0.1)
ttk.Spinbox(self._match_frame, from_=0, to=60, increment=0.1, textvariable=self._tc_inc_var, width=8).grid(
    row=3, column=1, sticky="w", pady=(4,0)
)

# Move delay for visualization
ttk.Label(self._match_frame, text="Move delay (ms):").grid(row=4, column=0, sticky="w", pady=(4,0))
self._delay_var = tk.IntVar(value=200)
ttk.Scale(self._match_frame, from_=0, to=2000, variable=self._delay_var, orient=tk.HORIZONTAL).grid(
    row=4, column=1, sticky="ew", pady=(4,0)
)

# SPRT
self._sprt_var = tk.BooleanVar(value=False)
ttk.Checkbutton(self._match_frame, text="SPRT", variable=self._sprt_var).grid(
    row=5, column=0, sticky="w", pady=(4,0)
)
sprt_frame = ttk.Frame(self._match_frame)
sprt_frame.grid(row=5, column=1, sticky="w", pady=(4,0))
self._sprt_elo0_var = tk.DoubleVar(value=0.0)
self._sprt_elo1_var = tk.DoubleVar(value=50.0)
ttk.Entry(sprt_frame, textvariable=self._sprt_elo0_var, width=5).pack(side=tk.LEFT)
ttk.Label(sprt_frame, text="-").pack(side=tk.LEFT)
ttk.Entry(sprt_frame, textvariable=self._sprt_elo1_var, width=5).pack(side=tk.LEFT)

# Match stats display
self._match_stats_var = tk.StringVar(value="")
ttk.Label(self._match_frame, textvariable=self._match_stats_var, wraplength=200).grid(
    row=6, column=0, columnspan=2, sticky="w", pady=(8,0)
)
```

Add `_on_mode_change()` to show/hide match controls:

```python
def _on_mode_change(self) -> None:
    if self._mode_var.get() == "match":
        self._match_frame.grid(row=2, column=1, sticky="new", pady=(8, 0))
    else:
        self._match_frame.grid_remove()
```

Modify `_new_game()` to dispatch to `_start_match()` in match mode:

```python
def _new_game(self) -> None:
    self._stop_game()
    if self._mode_var.get() == "match":
        self._start_match()
        return
    # ... existing play mode logic ...
```

Add `_start_match()`:

```python
def _start_match(self) -> None:
    from denoisr.gui.elo import compute_elo, likelihood_of_superiority, sprt_test
    from denoisr.gui.match_engine import run_match
    from denoisr.gui.types import MatchConfig

    ckpt = self._ckpt_var.get().strip()
    if not ckpt:
        self._status_var.set("Set a checkpoint first")
        return

    mode = self._engine_mode_var.get()
    e1_args = ("run", "denoisr-play", "--checkpoint", ckpt, "--mode", mode)
    e1 = EngineConfig(command="uv", args=e1_args, name="Denoisr")

    opp_cmd = self._opp_var.get().strip()
    opp_parts = opp_cmd.split()
    e2 = EngineConfig(command=opp_parts[0], args=tuple(opp_parts[1:]), name="Opponent")

    tc = TimeControl(
        base_seconds=self._tc_base_var.get(),
        increment=self._tc_inc_var.get(),
    )
    config = MatchConfig(
        engine1=e1, engine2=e2, games=self._games_var.get(), time_control=tc
    )

    self._match_running = True
    self._board_widget.set_interactive(False)
    self._status_var.set("Match running...")
    wins, draws, losses = 0, 0, 0

    def on_move(game_num: int, board: chess.Board, uci: str) -> None:
        # Queue board update for Tkinter thread
        self._move_queue.put(("match_board", board.copy(), chess.Move.from_uci(uci)))

    def on_game_complete(game_num: int, result: GameResult) -> None:
        nonlocal wins, draws, losses
        # Tally from engine1 perspective
        if result.engine1_color == "white":
            if result.result == "1-0":
                wins += 1
            elif result.result == "0-1":
                losses += 1
            else:
                draws += 1
        else:
            if result.result == "0-1":
                wins += 1
            elif result.result == "1-0":
                losses += 1
            else:
                draws += 1

        elo, error = compute_elo(wins, draws, losses)
        los = likelihood_of_superiority(wins, draws, losses)
        stats = f"Game {game_num + 1}/{config.games}\nW={wins} D={draws} L={losses}"
        if elo != float("inf") and elo != float("-inf"):
            stats += f"\nElo: {elo:+.1f} ± {error:.1f}"
        stats += f"\nLOS: {los:.1f}%"

        if self._sprt_var.get():
            sprt = sprt_test(
                wins, draws, losses,
                self._sprt_elo0_var.get(), self._sprt_elo1_var.get(),
            )
            if sprt is not None:
                stats += f"\nSPRT: {sprt} accepted"

        self._move_queue.put(("match_stats", stats))

    def run_in_bg() -> None:
        from denoisr.gui.types import GameResult  # noqa: F811
        run_match(
            config,
            on_game_complete=on_game_complete,
            on_move=on_move,
        )
        self._move_queue.put(("match_done", None))

    self._engine_thread = threading.Thread(target=run_in_bg, daemon=True)
    self._engine_thread.start()
```

Update `_poll_queue()` to handle match events:

```python
def _poll_queue(self) -> None:
    try:
        while True:
            item = self._move_queue.get_nowait()
            if isinstance(item, chess.Move):
                self._apply_engine_move(item)
            elif isinstance(item, tuple):
                kind = item[0]
                if kind == "match_board":
                    _, board, move = item
                    self._board_widget.set_board(board)
                    self._board_widget.highlight_last_move(move)
                elif kind == "match_stats":
                    self._match_stats_var.set(item[1])
                elif kind == "match_done":
                    self._status_var.set("Match complete")
            elif isinstance(item, str) and item.startswith("error:"):
                self._status_var.set(item)
    except queue.Empty:
        pass
    self._root.after(50, self._poll_queue)
```

**Step 2: Manually verify**

Run: `uv run python -c "from denoisr.gui.app import DenoisrApp; DenoisrApp().run()"`
Expected: Switch to Match mode shows additional controls. Starting a match against stockfish visualizes games on the board.

**Step 3: Commit**

```bash
git add src/denoisr/gui/app.py
git commit -m "feat: add match mode with Elo/SPRT tracking to GUI"
```

---

### Task 8: Entry point, pyproject.toml, and README update

**Files:**
- Create: `src/denoisr/scripts/gui.py`
- Modify: `pyproject.toml` (add `denoisr-gui` script)
- Modify: `README.md` (replace CuteChess instructions with built-in GUI)

**Step 1: Create the entry point script**

```python
# src/denoisr/scripts/gui.py
"""Launch the Denoisr chess GUI."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoisr Chess GUI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to model checkpoint (pre-fills the GUI field)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "diffusion"],
        default="single",
        help="Engine inference mode",
    )
    args = parser.parse_args()

    from denoisr.gui.app import DenoisrApp

    app = DenoisrApp()
    if args.checkpoint:
        app._ckpt_var.set(args.checkpoint)
    app._engine_mode_var.set(args.mode)
    app.run()
```

**Step 2: Add console script to pyproject.toml**

The current `[project.scripts]` section is:

```toml
[project.scripts]
denoisr-init = "denoisr.scripts.init_model:main"
denoisr-generate-data = "denoisr.scripts.generate_data:main"
denoisr-train-phase1 = "denoisr.scripts.train_phase1:main"
denoisr-train-phase2 = "denoisr.scripts.train_phase2:main"
denoisr-train-phase3 = "denoisr.scripts.train_phase3:main"
denoisr-play = "denoisr.scripts.play:main"
denoisr-benchmark = "denoisr.scripts.benchmark:main"
denoisr-export-mlx = "denoisr.scripts.export_mlx:main"
```

Add after `denoisr-export-mlx`:

```toml
denoisr-gui = "denoisr.scripts.gui:main"
```

**Step 3: Update README.md**

Replace the "Install a chess GUI" section (### 3) and "Connect the engine to the GUI" section (### 4) — currently lines 38-95 — with:

```markdown
### 3. Play against the engine

Denoisr includes a built-in chess GUI — no external software needed:

\`\`\`bash
uv run denoisr-gui --checkpoint outputs/random_model.pt
\`\`\`

This opens a window where you can play against the engine with click-to-move interaction.

> **Tip:** Denoisr also speaks the UCI protocol, so it works with any UCI-compatible GUI (CuteChess, Arena, Lucas Chess, etc.) if you prefer:
>
> \`\`\`bash
> uv run denoisr-play --checkpoint outputs/random_model.pt --mode single
> \`\`\`
>
> Then type `uci`, `isready`, `position startpos`, `go movetime 1000`, etc.

### 4. Play a game

In the GUI:
1. The checkpoint is pre-filled from the command line (or use **Browse** to select one)
2. Choose **single** or **diffusion** mode
3. Choose your color (white or black)
4. Click **New Game**
5. Click a piece, then click its destination to make moves

The engine responds automatically after each of your moves.
```

In the "Benchmarking with cutechess-cli" section (## Benchmarking..., currently line 312), prepend the GUI match mode before the existing cutechess-cli instructions:

```markdown
## Benchmarking

### GUI match mode (no external tools needed)

Switch to **Match** mode in the GUI to run engine-vs-engine matches with live Elo/SPRT tracking:

\`\`\`bash
uv run denoisr-gui --checkpoint outputs/phase3.pt --mode diffusion
\`\`\`

### cutechess-cli (advanced)

For headless benchmarking, the `denoisr-benchmark` command wraps cutechess-cli:
```

Update the "All available commands" table (currently line 403) to add:

```markdown
| `uv run denoisr-gui`           | Chess GUI for play and engine-vs-engine matches |
```

Note: The table already has `denoisr-export-mlx` (added recently). Place `denoisr-gui` after it.

Update the "Project structure" tree (currently line 436) to add the gui package:

```markdown
│   ├── gui/           # Built-in chess GUI (play, match, Elo/SPRT)
```

**Step 4: Verify the entry point works**

Run: `uv run denoisr-gui --help`
Expected: Shows help text with `--checkpoint` and `--mode` options.

**Step 5: Commit**

```bash
git add src/denoisr/scripts/gui.py pyproject.toml README.md
git commit -m "feat: add denoisr-gui entry point and update README"
```

---

### Task 9: Run full test suite and verify

**Step 1: Run all new GUI tests**

Run: `uv run pytest tests/test_gui/ -v`
Expected: All tests pass.

**Step 2: Run full test suite to ensure no regressions**

Run: `uv run pytest tests/ -x -q`
Expected: All existing tests still pass.

**Step 3: Run linter**

Run: `uvx ruff check src/denoisr/gui/ tests/test_gui/`
Expected: No issues.

**Step 4: Run type checker**

Run: `uv run --with mypy mypy --strict src/denoisr/gui/`
Expected: No type errors (or only expected Tkinter-related limitations).

**Step 5: Fix any issues found, then commit**

```bash
git add -A
git commit -m "fix: address lint and type issues in GUI module"
```
