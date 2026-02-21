# Benchmark Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace cutechess-cli dependency with a self-contained parallel benchmark using a shared `engine/` package extracted from `gui/`.

**Architecture:** Move `gui/{types,uci_engine,match_engine,elo}.py` into `engine/`, add UCI option setting + opening book + FEN starts, rewrite `evaluation/benchmark.py` with `multiprocessing.Pool` parallelism (same pattern as `generate_data.py`), rewrite CLI.

**Tech Stack:** python-chess, multiprocessing, existing UCIEngine/match_engine/elo modules

---

### Task 1: Create `engine/` Package — Move Types

Move `gui/types.py` to `engine/types.py`. Update all imports. Delete `gui/types.py`.

**Files:**
- Create: `src/denoisr/engine/__init__.py`
- Create: `src/denoisr/engine/types.py` (copy from `src/denoisr/gui/types.py`)
- Delete: `src/denoisr/gui/types.py`
- Modify: `src/denoisr/gui/app.py:15,19,474`
- Modify: `src/denoisr/gui/match_engine.py:11,17`
- Modify: `src/denoisr/gui/uci_engine.py:11`
- Move: `tests/test_gui/test_gui_types.py` → `tests/test_engine/test_engine_types.py`

**Step 1: Create `engine/` package with types**

Create `src/denoisr/engine/__init__.py` (empty).

Copy `src/denoisr/gui/types.py` verbatim to `src/denoisr/engine/types.py`.

**Step 2: Update all imports**

In `src/denoisr/gui/app.py`, replace:
- Line 15: `from denoisr.gui.types import` → `from denoisr.engine.types import`
- Line 19: `from denoisr.gui.types import` → `from denoisr.engine.types import`
- Line 474: `from denoisr.gui.types import` → `from denoisr.engine.types import`

In `src/denoisr/gui/match_engine.py`, replace:
- Line 11: `from denoisr.gui.types import` → `from denoisr.engine.types import`
- Line 17: `from denoisr.gui.types import` → `from denoisr.engine.types import`

In `src/denoisr/gui/uci_engine.py`, replace:
- Line 11: `from denoisr.gui.types import` → `from denoisr.engine.types import`

**Step 3: Move test file**

Create `tests/test_engine/__init__.py` (empty).

Copy `tests/test_gui/test_gui_types.py` to `tests/test_engine/test_engine_types.py`, replacing:
```python
from denoisr.gui.types import (
```
with:
```python
from denoisr.engine.types import (
```

Delete `tests/test_gui/test_gui_types.py`.

**Step 4: Delete old file**

Delete `src/denoisr/gui/types.py`.

**Step 5: Run tests**

Run: `uv run pytest tests/test_engine/test_engine_types.py tests/test_gui/ -x -q`
Expected: All pass (types work from new location, gui tests still pass via transitive imports).

**Step 6: Commit**

```bash
git add -A && git commit -m "refactor: move types.py from gui/ to engine/ package"
```

---

### Task 2: Move `elo.py` to `engine/`

**Files:**
- Create: `src/denoisr/engine/elo.py` (copy from `src/denoisr/gui/elo.py`)
- Delete: `src/denoisr/gui/elo.py`
- Modify: `src/denoisr/gui/app.py:472`
- Move: `tests/test_gui/test_elo.py` → `tests/test_engine/test_elo.py`

**Step 1: Copy module**

Copy `src/denoisr/gui/elo.py` verbatim to `src/denoisr/engine/elo.py`.

**Step 2: Update imports**

In `src/denoisr/gui/app.py`, line 472:
```python
# OLD
from denoisr.gui.elo import compute_elo, likelihood_of_superiority, sprt_test
# NEW
from denoisr.engine.elo import compute_elo, likelihood_of_superiority, sprt_test
```

**Step 3: Move test**

Copy `tests/test_gui/test_elo.py` to `tests/test_engine/test_elo.py`, replacing:
```python
from denoisr.gui.elo import
```
with:
```python
from denoisr.engine.elo import
```

Delete `tests/test_gui/test_elo.py`.

**Step 4: Delete old file**

Delete `src/denoisr/gui/elo.py`.

**Step 5: Run tests**

Run: `uv run pytest tests/test_engine/test_elo.py -x -q`
Expected: All 11 tests pass.

**Step 6: Commit**

```bash
git add -A && git commit -m "refactor: move elo.py from gui/ to engine/"
```

---

### Task 3: Move `uci_engine.py` to `engine/`

**Files:**
- Create: `src/denoisr/engine/uci_engine.py` (copy from `src/denoisr/gui/uci_engine.py`)
- Delete: `src/denoisr/gui/uci_engine.py`
- Modify: `src/denoisr/gui/app.py:16`
- Modify: `src/denoisr/gui/match_engine.py:12`
- Move: `tests/test_gui/test_uci_engine.py` → `tests/test_engine/test_uci_engine.py`
- Move: `tests/test_gui/mock_engine.py` → `tests/test_engine/mock_engine.py`

**Step 1: Copy module**

Copy `src/denoisr/gui/uci_engine.py` to `src/denoisr/engine/uci_engine.py`. Update the TYPE_CHECKING import inside it:
```python
# OLD
from denoisr.gui.types import EngineConfig, TimeControl
# NEW
from denoisr.engine.types import EngineConfig, TimeControl
```

**Step 2: Update imports in consumers**

In `src/denoisr/gui/app.py`, line 16:
```python
from denoisr.engine.uci_engine import UCIEngine
```

In `src/denoisr/gui/match_engine.py`, line 12:
```python
from denoisr.engine.uci_engine import UCIEngine
```

**Step 3: Move test + mock**

Copy `tests/test_gui/mock_engine.py` to `tests/test_engine/mock_engine.py` (verbatim — no import changes needed, it only uses `chess` and `sys`).

Copy `tests/test_gui/test_uci_engine.py` to `tests/test_engine/test_uci_engine.py`, replacing:
```python
from denoisr.gui.types import EngineConfig, TimeControl
from denoisr.gui.uci_engine import UCIEngine
```
with:
```python
from denoisr.engine.types import EngineConfig, TimeControl
from denoisr.engine.uci_engine import UCIEngine
```

Also update mock engine path:
```python
# This stays the same relative reference, but the file is now in test_engine/
MOCK_ENGINE = str(Path(__file__).parent / "mock_engine.py")
```

Delete `tests/test_gui/test_uci_engine.py` and `tests/test_gui/mock_engine.py`.

**Step 4: Delete old files**

Delete `src/denoisr/gui/uci_engine.py`.

**Step 5: Run tests**

Run: `uv run pytest tests/test_engine/test_uci_engine.py -x -q`
Expected: All 5 tests pass.

**Step 6: Commit**

```bash
git add -A && git commit -m "refactor: move uci_engine.py and mock_engine.py from gui/ to engine/"
```

---

### Task 4: Move `match_engine.py` to `engine/`

**Files:**
- Create: `src/denoisr/engine/match_engine.py` (copy from `src/denoisr/gui/match_engine.py`)
- Delete: `src/denoisr/gui/match_engine.py`
- Modify: `src/denoisr/gui/app.py:473`
- Move: `tests/test_gui/test_match_engine.py` → `tests/test_engine/test_match_engine.py`

**Step 1: Copy module**

Copy `src/denoisr/gui/match_engine.py` to `src/denoisr/engine/match_engine.py`. Update all internal imports:
```python
# OLD
from denoisr.gui.types import GameResult
from denoisr.gui.uci_engine import UCIEngine
# ...
from denoisr.gui.types import MatchConfig, TimeControl
# NEW
from denoisr.engine.types import GameResult
from denoisr.engine.uci_engine import UCIEngine
# ...
from denoisr.engine.types import MatchConfig, TimeControl
```

**Step 2: Update imports in app.py**

In `src/denoisr/gui/app.py`, line 473:
```python
from denoisr.engine.match_engine import run_match
```

**Step 3: Move test**

Copy `tests/test_gui/test_match_engine.py` to `tests/test_engine/test_match_engine.py`, replacing all imports:
```python
from denoisr.engine.match_engine import play_game, run_match
from denoisr.engine.types import EngineConfig, GameResult, MatchConfig, TimeControl
from denoisr.engine.uci_engine import UCIEngine
```

Delete `tests/test_gui/test_match_engine.py`.

**Step 4: Delete old file**

Delete `src/denoisr/gui/match_engine.py`.

**Step 5: Run full test suite**

Run: `uv run pytest tests/test_engine/ tests/test_gui/ -x -q`
Expected: All pass. The `gui/` tests should only be `test_board_widget.py` now.

**Step 6: Commit**

```bash
git add -A && git commit -m "refactor: move match_engine.py from gui/ to engine/"
```

---

### Task 5: Add `set_option()` and `new_game()` to UCIEngine

**Files:**
- Modify: `src/denoisr/engine/uci_engine.py`
- Modify: `tests/test_engine/test_uci_engine.py`
- Modify: `tests/test_engine/mock_engine.py`

**Step 1: Write failing tests**

Add to `tests/test_engine/test_uci_engine.py`:

```python
class TestUCIEngineOptions:
    def test_set_option_sends_command(self) -> None:
        engine = UCIEngine(_mock_config())
        engine.start()
        # Should not raise — mock engine ignores unknown commands
        # but responds to isready with readyok
        engine.set_option("UCI_LimitStrength", "true")
        engine.set_option("UCI_Elo", "1200")
        assert engine.is_alive()
        engine.quit()

    def test_new_game_resets_state(self) -> None:
        engine = UCIEngine(_mock_config())
        engine.start()
        engine.set_position(fen=None, moves=["e2e4"])
        engine.new_game()
        # After new_game, engine should still be alive and responsive
        engine.set_position(fen=None, moves=[])
        move = engine.go(
            time_control=TimeControl(base_seconds=10.0, increment=0.1),
            wtime_ms=10000, btime_ms=10000,
        )
        board = chess.Board()
        assert chess.Move.from_uci(move) in board.legal_moves
        engine.quit()
```

**Step 2: Update mock engine to handle `ucinewgame` and `setoption`**

In `tests/test_engine/mock_engine.py`, add handling inside the main loop (after the `isready` branch):

```python
elif line == "ucinewgame":
    board = chess.Board()
elif line.startswith("setoption"):
    pass  # Acknowledge silently
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_engine/test_uci_engine.py::TestUCIEngineOptions -x -v`
Expected: FAIL — `UCIEngine` has no `set_option` or `new_game` method.

**Step 4: Implement `set_option()` and `new_game()`**

Add to `src/denoisr/engine/uci_engine.py` class `UCIEngine`, after the `go()` method:

```python
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
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_engine/test_uci_engine.py -x -q`
Expected: All pass (original 5 + 2 new).

**Step 6: Commit**

```bash
git add -A && git commit -m "feat: add set_option() and new_game() to UCIEngine"
```

---

### Task 6: Add `start_fen` Support to `play_game()`

**Files:**
- Modify: `src/denoisr/engine/match_engine.py`
- Modify: `tests/test_engine/test_match_engine.py`

**Step 1: Write failing test**

Add to `tests/test_engine/test_match_engine.py`:

```python
class TestPlayGameFromFen:
    def test_game_from_fen_completes(self) -> None:
        """Game starting from a custom FEN should complete normally."""
        # Scholars mate position — white has a clear advantage
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        tc = TimeControl(base_seconds=60.0, increment=0.0)
        e1_config = _mock_config("White")
        e2_config = _mock_config("Black")
        with UCIEngine(e1_config) as white, UCIEngine(e2_config) as black:
            white.start()
            black.start()
            result = play_game(
                white=white, black=black, time_control=tc,
                max_moves=200, start_fen=fen,
            )
        assert result.result in {"1-0", "0-1", "1/2-1/2"}
        assert len(result.moves) > 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_engine/test_match_engine.py::TestPlayGameFromFen -x -v`
Expected: FAIL — `play_game()` doesn't accept `start_fen` parameter.

**Step 3: Implement `start_fen` in `play_game()`**

In `src/denoisr/engine/match_engine.py`, modify `play_game()` signature:

```python
def play_game(
    white: UCIEngine,
    black: UCIEngine,
    time_control: TimeControl,
    max_moves: int = 500,
    on_move: Callable[[chess.Board, str], None] | None = None,
    engine1_color: str = "white",
    stop_event: threading.Event | None = None,
    move_delay_ms: int = 0,
    start_fen: str | None = None,
) -> GameResult:
```

Change the board initialization:

```python
# OLD
board = chess.Board()
# NEW
board = chess.Board(start_fen) if start_fen is not None else chess.Board()
```

Change the `set_position` call:

```python
# OLD
current.set_position(fen=None, moves=move_list)
# NEW
current.set_position(fen=start_fen, moves=move_list)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_engine/test_match_engine.py -x -q`
Expected: All pass (original tests unaffected, new test passes).

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add start_fen parameter to play_game()"
```

---

### Task 7: Create `engine/openings.py` — EPD Opening Book Loader

**Files:**
- Create: `src/denoisr/engine/openings.py`
- Create: `tests/test_engine/test_openings.py`

**Step 1: Write failing tests**

Create `tests/test_engine/test_openings.py`:

```python
from pathlib import Path

import chess

from denoisr.engine.openings import load_openings


class TestLoadOpenings:
    def test_loads_fen_lines(self, tmp_path: Path) -> None:
        epd = tmp_path / "openings.epd"
        epd.write_text(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\n"
            "rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2\n"
        )
        fens = load_openings(epd)
        assert len(fens) == 2
        # Each should be a valid FEN
        for fen in fens:
            chess.Board(fen)  # raises if invalid

    def test_skips_comments_and_blanks(self, tmp_path: Path) -> None:
        epd = tmp_path / "openings.epd"
        epd.write_text(
            "# This is a comment\n"
            "\n"
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\n"
            "   \n"
            "# Another comment\n"
        )
        fens = load_openings(epd)
        assert len(fens) == 1

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        epd = tmp_path / "openings.epd"
        epd.write_text("")
        fens = load_openings(epd)
        assert fens == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_engine/test_openings.py -x -v`
Expected: FAIL — `denoisr.engine.openings` doesn't exist.

**Step 3: Implement**

Create `src/denoisr/engine/openings.py`:

```python
"""EPD opening book loader for benchmark matches."""

from __future__ import annotations

from pathlib import Path


def load_openings(path: Path) -> list[str]:
    """Load FEN positions from an EPD file (one per line).

    Lines starting with '#' and blank lines are skipped.
    """
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_engine/test_openings.py -x -q`
Expected: 3 passed.

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: add EPD opening book loader"
```

---

### Task 8: Bundle Default Opening Book

**Files:**
- Create: `src/denoisr/data/openings/default.epd`

**Step 1: Create opening book**

Create `src/denoisr/data/openings/default.epd` with ~50 common opening positions. These are well-known positions after 2-4 moves from standard openings:

```
# Common chess opening positions for benchmark matches
# Each line is a FEN after 1-2 opening moves
#
# King's Pawn
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2
rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2
rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 2
# Sicilian
rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2
rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2
rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3
rnbqkbnr/pp2pppp/3p4/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq d3 0 3
r1bqkbnr/pp2pppp/2np4/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 1 4
# French
rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2
rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2
rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq d6 0 3
# Caro-Kann
rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2
rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2
rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq d6 0 3
# Queen's Pawn
rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1
rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2
rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2
rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 1 3
rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2
rnbqkbnr/pppp1ppp/4p3/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2
rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 1 3
rnbqkb1r/pppp1ppp/4pn2/8/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 2 3
rnbqkb1r/pppp1ppp/4pn2/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 2 3
# Queen's Gambit Declined
rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3
rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 1 3
# Slav
rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3
rnbqkbnr/pp2pppp/2p5/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 1 3
# English
rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1
rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq e6 0 2
rnbqkb1r/pppppppp/5n2/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 1 2
# Reti
rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1
rnbqkbnr/ppp1pppp/8/3p4/8/5N2/PPPPPPPP/RNBQKB1R w KQkq d6 0 2
rnbqkbnr/ppp1pppp/8/3p4/2P5/5N2/PP1PPPPP/RNBQKB1R b KQkq c3 0 2
# King's Indian
rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3
rnbqkb1r/pppppp1p/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 1 3
rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4
# Nimzo-Indian
rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4
# Grunfeld
rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq d6 0 4
# Pirc
rnbqkbnr/ppp1pppp/3p4/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2
rnbqkbnr/ppp1pppp/3p4/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2
rnbqkb1r/ppp1pppp/3p1n2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 1 3
rnbqkb1r/ppp1pppp/3p1n2/8/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq - 2 3
# Scandinavian
rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2
rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2
# Alekhine
rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2
rnbqkb1r/pppppppp/5n2/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2
# Italian
r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3
```

**Step 2: Verify FENs are valid**

Run: `uv run python -c "
import chess
from pathlib import Path
lines = [l.strip() for l in Path('src/denoisr/data/openings/default.epd').read_text().splitlines() if l.strip() and not l.strip().startswith('#')]
for i, fen in enumerate(lines):
    chess.Board(fen)
print(f'{len(lines)} valid FENs')
"`

Expected: `50 valid FENs` (or similar count, no errors).

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: bundle default EPD opening book (50 positions)"
```

---

### Task 9: Rewrite `evaluation/benchmark.py` — Parallel Benchmark

This is the core task. Replace cutechess-cli wrapper with `multiprocessing.Pool`-based parallel game execution.

**Files:**
- Rewrite: `src/denoisr/evaluation/benchmark.py`
- Rewrite: `tests/test_evaluation/test_benchmark.py`

**Step 1: Write failing tests**

Rewrite `tests/test_evaluation/test_benchmark.py`:

```python
import sys
from pathlib import Path

from denoisr.engine.types import TimeControl
from denoisr.evaluation.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark,
)

MOCK_ENGINE = str(Path(__file__).parents[1] / "test_engine" / "mock_engine.py")


def _mock_cmd() -> str:
    return sys.executable


def _mock_args() -> tuple[str, ...]:
    return (MOCK_ENGINE,)


class TestRunBenchmark:
    def test_completes_fixed_games(self) -> None:
        config = BenchmarkConfig(
            engine_cmd=_mock_cmd(),
            engine_args=_mock_args(),
            opponent_cmd=_mock_cmd(),
            opponent_args=_mock_args(),
            games=4,
            time_control=TimeControl(base_seconds=60.0, increment=0.0),
            concurrency=2,
        )
        result = run_benchmark(config)
        assert isinstance(result, BenchmarkResult)
        assert result.games_played == 4
        assert result.wins + result.draws + result.losses == 4

    def test_sprt_can_stop_early(self) -> None:
        config = BenchmarkConfig(
            engine_cmd=_mock_cmd(),
            engine_args=_mock_args(),
            opponent_cmd=_mock_cmd(),
            opponent_args=_mock_args(),
            games=1000,
            time_control=TimeControl(base_seconds=60.0, increment=0.0),
            sprt_elo0=0.0,
            sprt_elo1=400.0,
            concurrency=2,
        )
        result = run_benchmark(config)
        # With identical mock engines, SPRT should conclude H0 well before 1000 games
        assert result.games_played < 1000
        assert result.sprt_result in {"H0", "H1", None}

    def test_openings_are_used(self, tmp_path: Path) -> None:
        epd = tmp_path / "test.epd"
        epd.write_text(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\n"
        )
        config = BenchmarkConfig(
            engine_cmd=_mock_cmd(),
            engine_args=_mock_args(),
            opponent_cmd=_mock_cmd(),
            opponent_args=_mock_args(),
            games=2,
            time_control=TimeControl(base_seconds=60.0, increment=0.0),
            openings_path=epd,
            concurrency=1,
        )
        result = run_benchmark(config)
        assert result.games_played == 2
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluation/test_benchmark.py -x -v`
Expected: FAIL — old `BenchmarkConfig` has different fields.

**Step 3: Implement `evaluation/benchmark.py`**

Rewrite `src/denoisr/evaluation/benchmark.py`:

```python
"""Self-contained parallel Elo benchmark — no cutechess-cli required.

Mirrors the parallelization pattern from generate_data.py:
each worker owns a persistent engine + opponent subprocess pair.
"""

from __future__ import annotations

import atexit
import multiprocessing
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from denoisr.engine.elo import compute_elo, likelihood_of_superiority, sprt_test
from denoisr.engine.match_engine import play_game
from denoisr.engine.openings import load_openings
from denoisr.engine.types import EngineConfig, TimeControl
from denoisr.engine.uci_engine import UCIEngine

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Per-worker process globals (set by _init_worker)
# ---------------------------------------------------------------------------

_engine: UCIEngine | None = None
_opponent: UCIEngine | None = None
_time_control: TimeControl | None = None


def _cleanup_engines() -> None:
    global _engine, _opponent
    for eng in (_engine, _opponent):
        if eng is None:
            continue
        try:
            eng.quit()
        except Exception:  # noqa: BLE001
            pass
    _engine = None
    _opponent = None


def _init_worker(
    engine_cmd: str,
    engine_args: tuple[str, ...],
    opponent_cmd: str,
    opponent_args: tuple[str, ...],
    opponent_elo: int | None,
    time_control: TimeControl,
) -> None:
    global _engine, _opponent, _time_control
    _time_control = time_control

    _engine = UCIEngine(EngineConfig(engine_cmd, engine_args, "Denoisr"))
    _engine.start()

    _opponent = UCIEngine(EngineConfig(opponent_cmd, opponent_args, "Opponent"))
    _opponent.start()
    if opponent_elo is not None:
        _opponent.set_option("UCI_LimitStrength", "true")
        _opponent.set_option("UCI_Elo", str(opponent_elo))

    atexit.register(_cleanup_engines)


# ---------------------------------------------------------------------------
# Work item
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GameTask:
    game_num: int
    start_fen: str | None
    engine_is_white: bool


def _play_one_game(task: _GameTask) -> tuple[int, str, str]:
    """Play a single game in a worker process.

    Returns (game_num, result_str, engine1_color) — lightweight
    picklable tuple instead of full GameResult to avoid IPC issues.
    """
    if _engine is None or _opponent is None or _time_control is None:
        raise RuntimeError("Worker not initialized")

    _engine.new_game()
    _opponent.new_game()

    if task.engine_is_white:
        white, black = _engine, _opponent
        e1_color = "white"
    else:
        white, black = _opponent, _engine
        e1_color = "black"

    result = play_game(
        white=white,
        black=black,
        time_control=_time_control,
        start_fen=task.start_fen,
        engine1_color=e1_color,
    )
    return (task.game_num, result.result, result.engine1_color)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _default_concurrency() -> int:
    return (os.cpu_count() or 1) * 2 + 1


@dataclass(frozen=True)
class BenchmarkConfig:
    engine_cmd: str
    engine_args: tuple[str, ...] = ()
    opponent_cmd: str = "stockfish"
    opponent_args: tuple[str, ...] = ()
    opponent_elo: int | None = None
    games: int = 100
    time_control: TimeControl = TimeControl(base_seconds=10.0, increment=0.1)
    openings_path: Path | None = None
    sprt_elo0: float | None = None
    sprt_elo1: float | None = None
    concurrency: int = _default_concurrency()


@dataclass(frozen=True)
class BenchmarkResult:
    wins: int
    draws: int
    losses: int
    elo_diff: float
    elo_error: float
    los: float
    sprt_result: str | None
    games_played: int


def run_benchmark(
    config: BenchmarkConfig,
    on_game: Callable[[int, int, int, int], None] | None = None,
) -> BenchmarkResult:
    """Run a parallel benchmark match and return Elo/SPRT results.

    on_game callback receives (games_played, wins, draws, losses).
    """
    # Load openings
    openings: list[str | None]
    if config.openings_path is not None:
        fens = load_openings(config.openings_path)
        openings = [f for f in fens] if fens else [None]
    else:
        openings = [None]

    random.shuffle(openings)

    # Build game tasks — pairs share the same opening
    tasks: list[_GameTask] = []
    for i in range(config.games):
        opening_idx = (i // 2) % len(openings)
        fen = openings[opening_idx]
        engine_is_white = i % 2 == 0
        tasks.append(_GameTask(game_num=i, start_fen=fen, engine_is_white=engine_is_white))

    wins = 0
    draws = 0
    losses = 0
    games_played = 0
    sprt_result: str | None = None

    with multiprocessing.Pool(
        min(config.concurrency, config.games),
        initializer=_init_worker,
        initargs=(
            config.engine_cmd,
            config.engine_args,
            config.opponent_cmd,
            config.opponent_args,
            config.opponent_elo,
            config.time_control,
        ),
    ) as pool:
        for game_num, result_str, e1_color in pool.imap_unordered(
            _play_one_game, tasks
        ):
            games_played += 1

            # Tally from engine's (Denoisr's) perspective
            engine_won = (
                (result_str == "1-0" and e1_color == "white")
                or (result_str == "0-1" and e1_color == "black")
            )
            engine_lost = (
                (result_str == "0-1" and e1_color == "white")
                or (result_str == "1-0" and e1_color == "black")
            )
            if engine_won:
                wins += 1
            elif engine_lost:
                losses += 1
            else:
                draws += 1

            if on_game is not None:
                on_game(games_played, wins, draws, losses)

            # Check SPRT
            if config.sprt_elo0 is not None and config.sprt_elo1 is not None:
                sprt_result = sprt_test(
                    wins, draws, losses, config.sprt_elo0, config.sprt_elo1
                )
                if sprt_result is not None:
                    pool.terminate()
                    break

    elo_diff, elo_error = compute_elo(wins, draws, losses)
    los = likelihood_of_superiority(wins, draws, losses)

    return BenchmarkResult(
        wins=wins,
        draws=draws,
        losses=losses,
        elo_diff=elo_diff,
        elo_error=elo_error,
        los=los,
        sprt_result=sprt_result,
        games_played=games_played,
    )
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_evaluation/test_benchmark.py -x -v`
Expected: All 3 tests pass.

**Step 5: Commit**

```bash
git add -A && git commit -m "feat: rewrite benchmark with parallel game execution"
```

---

### Task 10: Rewrite `scripts/benchmark.py` CLI

**Files:**
- Rewrite: `src/denoisr/scripts/benchmark.py`

**Step 1: Rewrite the CLI**

```python
"""Elo benchmarking — self-contained parallel match against a reference engine.

Runs the trained Denoisr engine against an opponent (e.g. Stockfish) using
parallel game execution with optional SPRT for statistical confidence.
"""

import argparse
import math
import shutil
import sys
from importlib import resources
from pathlib import Path

from denoisr.engine.types import TimeControl
from denoisr.evaluation.benchmark import (
    BenchmarkConfig,
    _default_concurrency,
    run_benchmark,
)


def _parse_time_control(tc_str: str) -> TimeControl:
    """Parse 'base+increment' string into TimeControl."""
    parts = tc_str.split("+")
    base = float(parts[0])
    increment = float(parts[1]) if len(parts) > 1 else 0.0
    return TimeControl(base_seconds=base, increment=increment)


def _default_openings_path() -> Path:
    """Locate the bundled default.epd opening book."""
    ref = resources.files("denoisr.data.openings").joinpath("default.epd")
    return Path(str(ref))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Denoisr against a reference engine"
    )
    parser.add_argument(
        "--engine-cmd", required=True,
        help="Command to run the Denoisr UCI engine",
    )
    parser.add_argument(
        "--engine-args", default="",
        help="Additional args for Denoisr engine (space-separated)",
    )
    parser.add_argument(
        "--opponent-cmd", default=None,
        help="Command to run the opponent engine (default: auto-detect stockfish)",
    )
    parser.add_argument(
        "--opponent-args", default="",
        help="Additional args for opponent engine (space-separated)",
    )
    parser.add_argument(
        "--opponent-elo", type=int, default=None,
        help="Limit opponent strength via UCI_Elo (e.g. 1200)",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--time-control", default="10+0.1",
        help="Time control as 'base+increment' in seconds (default: 10+0.1)")
    parser.add_argument(
        "--openings", type=str, default=None,
        help="Path to EPD opening book (default: bundled 50-position book)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=_default_concurrency(),
        help=f"Parallel games (default: cpu_count*2+1 = {_default_concurrency()})",
    )
    parser.add_argument("--sprt-elo0", type=float, default=None)
    parser.add_argument("--sprt-elo1", type=float, default=None)
    args = parser.parse_args()

    opponent_cmd = args.opponent_cmd or shutil.which("stockfish")
    if opponent_cmd is None:
        print(
            "Error: Stockfish not found. Install it or pass --opponent-cmd",
            file=sys.stderr,
        )
        sys.exit(1)

    openings_path: Path | None
    if args.openings is not None:
        openings_path = Path(args.openings)
    else:
        openings_path = _default_openings_path()

    tc = _parse_time_control(args.time_control)
    engine_args = tuple(args.engine_args.split()) if args.engine_args else ()
    opponent_args = tuple(args.opponent_args.split()) if args.opponent_args else ()

    config = BenchmarkConfig(
        engine_cmd=args.engine_cmd,
        engine_args=engine_args,
        opponent_cmd=opponent_cmd,
        opponent_args=opponent_args,
        opponent_elo=args.opponent_elo,
        games=args.games,
        time_control=tc,
        openings_path=openings_path,
        sprt_elo0=args.sprt_elo0,
        sprt_elo1=args.sprt_elo1,
        concurrency=args.concurrency,
    )

    sprt_msg = ""
    if config.sprt_elo0 is not None and config.sprt_elo1 is not None:
        sprt_msg = f", SPRT[{config.sprt_elo0:.0f},{config.sprt_elo1:.0f}]"
    elo_msg = f" vs Elo {config.opponent_elo}" if config.opponent_elo else ""
    print(
        f"Benchmark: {config.games} games, {config.concurrency} workers, "
        f"TC {args.time_control}{elo_msg}{sprt_msg}"
    )

    def on_game(played: int, w: int, d: int, l: int) -> None:
        from denoisr.engine.elo import compute_elo
        elo, err = compute_elo(w, d, l)
        elo_str = f"{elo:+.1f} ± {err:.1f}" if not math.isinf(elo) else "N/A"
        print(f"  Game {played}/{config.games}: +{w} ={d} -{l} | Elo: {elo_str}")

    result = run_benchmark(config, on_game=on_game)

    print(f"\nResult: +{result.wins} ={result.draws} -{result.losses}"
          f" ({result.games_played} games)")
    if not math.isinf(result.elo_diff):
        print(f"Elo: {result.elo_diff:+.1f} ± {result.elo_error:.1f}")
    print(f"LOS: {result.los:.1f}%")
    if result.sprt_result is not None:
        print(f"SPRT: {result.sprt_result}")


if __name__ == "__main__":
    main()
```

**Step 2: Run a quick smoke test**

Run: `uv run denoisr-benchmark --help`
Expected: Help text with new flags.

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: rewrite benchmark CLI for self-contained parallel execution"
```

---

### Task 11: Ensure `data/openings/` is Included in Package

**Files:**
- Modify: `pyproject.toml` (if needed — check if `package-data` or `include` is configured)

**Step 1: Check current package data config**

Read `pyproject.toml` and check if `src/denoisr/data/openings/` will be included in the built package.

**Step 2: Add `__init__.py` files if needed**

Create `src/denoisr/data/openings/__init__.py` (empty) so `importlib.resources` can find it.

**Step 3: Verify resource loading**

Run: `uv run python -c "
from importlib import resources
ref = resources.files('denoisr.data.openings').joinpath('default.epd')
print(ref)
print(len(open(str(ref)).readlines()), 'lines')
"`
Expected: Prints path and line count.

**Step 4: Commit**

```bash
git add -A && git commit -m "fix: ensure openings EPD is discoverable as package data"
```

---

### Task 12: Update README

**Files:**
- Modify: `README.md` — update the benchmarking section

**Step 1: Replace cutechess-cli references**

Remove any mention of `cutechess-cli`. Update the benchmark CLI usage section with:

```markdown
### Benchmarking

Estimate Elo against Stockfish (no external tools needed):

```bash
uv run denoisr-benchmark \
  --engine-cmd "uv run denoisr-play --checkpoint outputs/phase1_v2.pt" \
  --opponent-elo 1200 \
  --games 100
```

| Flag | Default | Description |
|------|---------|-------------|
| `--engine-cmd` | (required) | Command to run the Denoisr UCI engine |
| `--opponent-cmd` | auto-detect `stockfish` | Opponent engine command |
| `--opponent-elo` | full strength | Limit opponent via UCI_Elo |
| `--games` | `100` | Number of games |
| `--time-control` | `10+0.1` | Base+increment seconds |
| `--openings` | bundled 50-position book | EPD opening book path |
| `--concurrency` | `cpu_count*2+1` | Parallel game workers |
| `--sprt-elo0` | None | SPRT null hypothesis |
| `--sprt-elo1` | None | SPRT alternative hypothesis |
```

**Step 2: Commit**

```bash
git add README.md && git commit -m "docs: update README benchmark section (no cutechess-cli)"
```

---

### Task 13: Final Verification

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass, no import errors.

**Step 2: Run lint**

Run: `uvx ruff check src/denoisr/engine/ src/denoisr/evaluation/ src/denoisr/scripts/benchmark.py`
Expected: No errors.

**Step 3: Run type check**

Run: `uv run --with mypy mypy --strict src/denoisr/engine/ src/denoisr/evaluation/benchmark.py`
Expected: No errors (or pre-existing ones only).

**Step 4: Verify no stale gui imports remain**

Run: `grep -r "from denoisr.gui.types\|from denoisr.gui.elo\|from denoisr.gui.uci_engine\|from denoisr.gui.match_engine" src/ tests/`
Expected: No matches.

**Step 5: Commit any fixes**

```bash
git add -A && git commit -m "fix: address lint/type issues from benchmark overhaul"
```
