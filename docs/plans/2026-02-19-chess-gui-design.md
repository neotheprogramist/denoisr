# Chess GUI Design — Replace CuteChess with Built-in Tkinter GUI

## Goal

Replace the external CuteChess dependency with a built-in chess GUI that supports:
1. Human vs engine interactive play
2. Engine vs engine matches with Elo/SPRT computation (replacing cutechess-cli)

## Constraints

- **Tkinter only** — no extra GUI dependencies (stdlib)
- **Unicode chess symbols** — no image assets to ship
- **Click-click** move input with legal move highlighting
- **Pure Python** benchmarking — no cutechess-cli dependency
- Existing UCI engine (`denoisr-play`) and `uci.py` are untouched

## Architecture

Three layers, cleanly separated:

```
┌──────────────────────────────────────┐
│  GUI App (app.py)                    │  Tkinter window, controls, modes
│  ├─ BoardWidget (board_widget.py)    │  Canvas rendering + click interaction
│  └─ MatchEngine (match_engine.py)    │  Headless UCI orchestration
│       └─ Elo/SPRT (elo.py)          │  Statistical computation
└──────────────────────────────────────┘
```

## Module Layout

```
src/denoisr/
├── gui/
│   ├── __init__.py
│   ├── match_engine.py       # Headless UCI match orchestration
│   ├── elo.py                # Elo + SPRT computation
│   ├── board_widget.py       # Tkinter Canvas chess board
│   └── app.py                # Main GUI application
└── scripts/
    └── gui.py                # Entry point: denoisr-gui
```

New console script in pyproject.toml:
```
denoisr-gui = "denoisr.scripts.gui:main"
```

## Layer 1: Match Engine (headless)

### Data Types

```python
@dataclass(frozen=True)
class EngineConfig:
    command: str              # e.g. "uv run denoisr-play"
    args: tuple[str, ...]    # e.g. ("--checkpoint", "model.pt", "--mode", "single")
    name: str                # Display name

@dataclass(frozen=True)
class TimeControl:
    base_seconds: float       # e.g. 10.0
    increment: float          # e.g. 0.1

@dataclass(frozen=True)
class MatchConfig:
    engine1: EngineConfig
    engine2: EngineConfig
    games: int
    time_control: TimeControl
    concurrency: int = 1

@dataclass(frozen=True)
class GameResult:
    moves: tuple[str, ...]    # UCI move sequence
    result: str               # "1-0", "0-1", "1/2-1/2"
    reason: str               # "checkmate", "timeout", "stalemate", etc.
    engine1_color: str        # "white" or "black"

class GameOutcome(Enum):
    WIN = auto()
    DRAW = auto()
    LOSS = auto()
```

### Core API

```python
class UCIEngine:
    """Manages a single UCI engine subprocess."""
    def __init__(self, config: EngineConfig) -> None: ...
    def send(self, command: str) -> None: ...
    def wait_for(self, expected: str, timeout: float) -> str: ...
    def start_game(self) -> None: ...
    def set_position(self, fen: str | None, moves: list[str]) -> None: ...
    def go(self, time_control: TimeControl, color: str) -> str: ...
    def quit(self) -> None: ...

def play_game(
    white: UCIEngine,
    black: UCIEngine,
    time_control: TimeControl,
    on_move: Callable[[chess.Board, str], None] | None = None,
) -> GameResult: ...

def run_match(
    config: MatchConfig,
    on_game_complete: Callable[[int, GameResult], None] | None = None,
    on_move: Callable[[int, chess.Board, str], None] | None = None,
) -> list[GameResult]: ...
```

Engines alternate colors each game. Time is tracked via wall-clock.

### Elo + SPRT

```python
def compute_elo(wins: int, draws: int, losses: int) -> tuple[float, float]: ...
def likelihood_of_superiority(wins: int, draws: int, losses: int) -> float: ...
def sprt_test(
    wins: int, draws: int, losses: int,
    elo0: float, elo1: float,
    alpha: float = 0.05, beta: float = 0.05,
) -> str | None: ...  # "H0", "H1", or None (continue)
```

## Layer 2: Board Widget

Tkinter Canvas widget, self-contained.

### Rendering

- 480x480 default (60px/square), resizable
- Square colors: light #F0D9B5, dark #B58863
- Pieces: Unicode symbols (♔♕♖♗♘♙ / ♚♛♜♝♞♟) as large centered text
- Highlights: blue border (selected), green circles (legal moves), yellow tint (last move)
- Flippable orientation

### Interaction

- Click piece → select, show legal destinations
- Click legal destination → make move
- Click elsewhere → deselect
- Pawn promotion → 4-button dialog (Q/R/B/N)

### API

```python
class BoardWidget(tk.Canvas):
    def __init__(self, parent: tk.Widget, square_size: int = 60) -> None: ...
    def set_board(self, board: chess.Board) -> None: ...
    def set_interactive(self, enabled: bool) -> None: ...
    def set_on_move(self, callback: Callable[[chess.Move], None]) -> None: ...
    def flip(self) -> None: ...
    def highlight_last_move(self, move: chess.Move) -> None: ...
```

## Layer 3: GUI Application

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  Denoisr Chess                                          │
├─────────────────────────────────────────────────────────┤
│   ┌───────────────────────┐   ┌───────────────────────┐ │
│   │                       │   │ Mode: ○ Play  ○ Match │ │
│   │    Chess Board        │   │ Engine 1: [dropdown]  │ │
│   │    (BoardWidget)      │   │ Checkpoint: [Browse]  │ │
│   │    480 x 480          │   │ Engine 2: [dropdown]  │ │
│   │                       │   │ Time: [10] + [0.1]    │ │
│   │                       │   │ Games: [100]          │ │
│   └───────────────────────┘   │ [New Game] [Stop]     │ │
│                               │ [Flip Board]          │ │
│   ┌───────────────────────────┴───────────────────────┐ │
│   │ Status / Move list / Eval / Match stats           │ │
│   └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Two Modes

**Play mode:** Human vs Engine. Click-click moves for human. Engine runs in background thread. Shows W/D/L evaluation and move list.

**Match mode:** Engine vs Engine. Visualized on board with adjustable speed. Live Elo/SPRT updates. Pause/resume.

### Threading Model

Tkinter is single-threaded. Engine communication runs in background threads. Moves posted to main thread via `queue.Queue` + `widget.after()` polling (~50ms interval).

### Entry Point

```bash
uv run denoisr-gui
uv run denoisr-gui --checkpoint outputs/phase3.pt --mode diffusion
```

## Testing Strategy

- `test_elo.py` — unit tests for Elo/SPRT math with known values
- `test_match_engine.py` — integration tests with a mock UCI engine (echo-based)
- `test_board_widget.py` — programmatic tests: set_board renders correct squares, click sequences produce correct moves
- Match engine is fully testable headless (no Tkinter needed)

## What This Replaces

| Before                         | After                              |
| ------------------------------ | ---------------------------------- |
| Install CuteChess externally   | `uv run denoisr-gui` (built-in)   |
| cutechess-cli for benchmarks   | Pure Python match engine           |
| `denoisr-benchmark` wrapper    | GUI match mode + headless fallback |
