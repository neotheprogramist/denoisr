# Benchmark Overhaul Design: Drop cutechess-cli, Self-Contained Parallel Elo Estimation

## Problem

`denoisr-benchmark` shells out to `cutechess-cli`, an external C++ tool that must be installed separately. This creates friction — users can't estimate Elo without finding, compiling, or installing cutechess-cli.

The codebase already has a complete pure-Python match engine (`gui/match_engine.py`), UCI subprocess manager (`gui/uci_engine.py`), and Elo/SPRT calculator (`gui/elo.py`) — all built for the GUI. These do everything cutechess-cli does for our use case.

## Approach: Extract `engine/` Package, Parallel Benchmark

1. Move shared engine infrastructure from `gui/` into a new `engine/` package
2. Rewrite benchmark to use `engine/` directly with `multiprocessing.Pool` parallelism (same pattern as `generate_data.py`)
3. Add UCI option setting (Stockfish Elo limiting), opening book support, and SPRT early stopping

## Package Structure

```
src/denoisr/
├── engine/                    # NEW shared package
│   ├── __init__.py
│   ├── types.py               # MOVED from gui/types.py
│   ├── uci_engine.py          # MOVED from gui/uci_engine.py (+ set_option, new_game)
│   ├── match_engine.py        # MOVED from gui/match_engine.py (+ start_fen)
│   ├── elo.py                 # MOVED from gui/elo.py
│   └── openings.py            # NEW — EPD opening book loader
├── gui/
│   └── app.py                 # imports from engine/ now
│   (types.py, uci_engine.py, match_engine.py, elo.py — DELETED)
├── evaluation/
│   └── benchmark.py           # REWRITTEN — parallel games via engine/
├── scripts/
│   └── benchmark.py           # REWRITTEN CLI
└── data/
    └── openings/
        └── default.epd        # bundled ~50-100 opening positions
```

## New Features for `engine/uci_engine.py`

### `set_option(name, value)`

```python
def set_option(self, name: str, value: str) -> None:
    self._send(f"setoption name {name} value {value}")
    self._send("isready")
    self._wait_for("readyok", timeout=10.0)
```

### `new_game()`

```python
def new_game(self) -> None:
    self._send("ucinewgame")
    self._send("isready")
    self._wait_for("readyok", timeout=10.0)
```

## Opening Book Support

### `engine/openings.py`

```python
def load_openings(path: Path) -> list[str]:
    """Load FEN positions from an EPD file (one per line)."""
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
```

### FEN start positions in `play_game()`

`play_game()` gains `start_fen: str | None`. When set, the board initializes from FEN and `set_position(fen=start_fen, moves=[])` is used.

### Fair pairing

Each game pair (2 games) uses the same opening — engine plays white in game N, black in game N+1. Openings are shuffled and cycled when exhausted.

### Bundled book

`data/openings/default.epd` — ~50-100 common opening positions from public-domain sources.

## Parallel Game Execution

Mirrors `generate_data.py` pattern: `multiprocessing.Pool` with per-worker persistent subprocesses.

### Worker model

```python
# Per-worker process globals
_denoisr_engine: UCIEngine | None = None
_opponent_engine: UCIEngine | None = None

def _init_worker(
    engine_cmd: str, engine_args: tuple[str, ...],
    opponent_cmd: str, opponent_args: tuple[str, ...],
    opponent_elo: int | None,
) -> None:
    """Each worker owns a persistent denoisr + stockfish subprocess pair."""
    global _denoisr_engine, _opponent_engine
    _denoisr_engine = UCIEngine(EngineConfig(engine_cmd, engine_args, "Denoisr"))
    _denoisr_engine.start()
    _opponent_engine = UCIEngine(EngineConfig(opponent_cmd, opponent_args, "Stockfish"))
    _opponent_engine.start()
    if opponent_elo is not None:
        _opponent_engine.set_option("UCI_LimitStrength", "true")
        _opponent_engine.set_option("UCI_Elo", str(opponent_elo))
    atexit.register(_cleanup_engines)
```

### Work item

```python
@dataclass(frozen=True)
class _GameTask:
    game_num: int
    start_fen: str | None
    engine_is_white: bool

def _play_one_game(task: _GameTask) -> GameResult:
    """Played inside worker process using persistent engine subprocesses."""
    _denoisr_engine.new_game()
    _opponent_engine.new_game()
    white = _denoisr_engine if task.engine_is_white else _opponent_engine
    black = _opponent_engine if task.engine_is_white else _denoisr_engine
    e1_color = "white" if task.engine_is_white else "black"
    return play_game(white, black, time_control, start_fen=task.start_fen, engine1_color=e1_color)
```

### Main process collection

```python
with multiprocessing.Pool(num_workers, initializer=_init_worker, initargs=(...)) as pool:
    for result in pool.imap_unordered(_play_one_game, game_tasks):
        # Tally W/D/L from denoisr's perspective
        # Print live: Game 12/100: +5 =4 -3 | Elo: -42.3 ± 180.1
        # Check SPRT — pool.terminate() if concluded
```

### Concurrency

```python
def _default_concurrency() -> int:
    return (os.cpu_count() or 1) * 2 + 1
```

Each worker holds 2 subprocesses (denoisr + stockfish). On 8 cores: 17 workers = 34 subprocesses.

Workers send `ucinewgame` + `isready` between games to reset state without restarting subprocesses.

SPRT early termination: `pool.terminate()` kills all workers and their subprocesses immediately.

## Rewritten `evaluation/benchmark.py`

### Config

```python
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
```

### `run_benchmark()` flow

1. Load openings from EPD (or `[None]` for startpos)
2. Shuffle openings, build game task list — each pair shares the same opening, alternating engine color
3. Create `multiprocessing.Pool(concurrency, initializer=_init_worker, ...)`
4. `pool.imap_unordered(_play_one_game, tasks)` — collect results as they complete
5. After each result: tally W/D/L, print progress with live Elo estimate, check SPRT
6. If SPRT concludes: `pool.terminate()`, break
7. Return `BenchmarkResult`

## Rewritten CLI (`scripts/benchmark.py`)

```
uv run denoisr-benchmark \
  --engine-cmd "uv run denoisr-play --checkpoint outputs/phase1_v2.pt" \
  --opponent-cmd stockfish \
  --opponent-elo 1200 \
  --games 100 \
  --time-control "10+0.1" \
  --openings data/openings/default.epd \
  --concurrency 17 \
  --sprt-elo0 0 --sprt-elo1 50
```

| Flag | Default | Description |
|------|---------|-------------|
| `--engine-cmd` | (required) | Command to run the Denoisr UCI engine |
| `--opponent-cmd` | `stockfish` | Command to run the opponent engine |
| `--opponent-elo` | None (full strength) | Set Stockfish UCI_Elo + UCI_LimitStrength |
| `--games` | `100` | Number of games to play |
| `--time-control` | `10+0.1` | Time control as `base+increment` |
| `--openings` | bundled `default.epd` | Path to EPD opening book |
| `--concurrency` | `cpu_count()*2+1` | Parallel games (each = 2 subprocesses) |
| `--sprt-elo0` | None | SPRT null hypothesis Elo |
| `--sprt-elo1` | None | SPRT alternative hypothesis Elo |

Removed: `--dry-run` (no external command to preview).

## Import Migration

All consumers of `gui/{types,uci_engine,match_engine,elo}.py` updated to import from `engine/`. The `gui/` copies are deleted — clean break, no re-export shims.

Affected files:
- `gui/app.py`
- `evaluation/benchmark.py`
- `scripts/benchmark.py`
- All tests under `test_gui/` for these modules → move to `test_engine/`

## Tests

### Moved (gui/ → engine/)
- `test_gui/test_match_engine.py` → `test_engine/test_match_engine.py`
- `test_gui/test_uci_engine.py` → `test_engine/test_uci_engine.py`
- `test_gui/test_elo.py` → `test_engine/test_elo.py`
- `test_gui/mock_engine.py` → `test_engine/mock_engine.py`

### New
- `test_engine/test_openings.py` — loads EPD, handles comments/blank lines, empty file
- `test_engine/test_uci_engine.py` — `set_option()` and `new_game()` protocol tests
- `test_evaluation/test_benchmark.py` — rewritten: SPRT early stop, opponent Elo config, opening rotation, concurrency

### Deleted
- Old `test_evaluation/test_benchmark.py` (tested `build_cutechess_command`/`parse_cutechess_output`)
