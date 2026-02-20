# GUI UX Polish Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix broken Stop button, add match/engine cancellation, and polish the GUI for a smooth development-testing workflow.

**Architecture:** Thread-safe cancellation via `threading.Event` passed into match/game functions. UI improvements replace non-selectable widgets with read-only `tk.Text`. Controls are disabled during active games to prevent invalid state changes.

**Tech Stack:** Python, Tkinter, python-chess, threading

---

### Task 1: Add `stop_event` to `play_game` and `run_match`

**Files:**
- Modify: `src/denoisr/gui/match_engine.py`
- Modify: `src/denoisr/gui/types.py:39-44`
- Test: `tests/test_gui/test_match_engine.py`

**Context:** The Stop button in the GUI sets `self._match_running = False` but `run_match()` never checks that flag. Engine subprocesses inside `run_match` are managed by local `with UCIEngine(...)` context managers, not tracked by `self._engine`. We need a `threading.Event` that both `play_game` and `run_match` check, so setting it causes immediate loop exit.

**Step 1: Write the failing tests**

Add to `tests/test_gui/test_match_engine.py`:

```python
import threading

def test_play_game_stops_on_event(tmp_path: Path) -> None:
    """play_game should return '*' result when stop_event is set."""
    mock = tmp_path / "mock_engine.py"
    mock.write_text(MOCK_ENGINE_SCRIPT)
    config = EngineConfig(command=sys.executable, args=(str(mock),), name="Mock")
    tc = TimeControl(base_seconds=60.0, increment=0.0)

    stop = threading.Event()
    stop.set()  # Pre-set: should stop immediately

    with UCIEngine(config) as white, UCIEngine(config) as black:
        white.start()
        black.start()
        result = play_game(white, black, tc, stop_event=stop)

    assert result.result == "*"
    assert result.reason == "stopped"


def test_run_match_stops_between_games(tmp_path: Path) -> None:
    """run_match should stop between games when stop_event is set."""
    mock = tmp_path / "mock_engine.py"
    mock.write_text(MOCK_ENGINE_SCRIPT)
    config = EngineConfig(command=sys.executable, args=(str(mock),), name="Mock")
    tc = TimeControl(base_seconds=60.0, increment=0.0)

    stop = threading.Event()
    games_played: list[int] = []

    def on_complete(game_num: int, result: GameResult) -> None:
        games_played.append(game_num)
        stop.set()  # Stop after first game

    match_config = MatchConfig(
        engine1=config, engine2=config, games=10, time_control=tc
    )
    results = run_match(match_config, on_game_complete=on_complete, stop_event=stop)

    assert len(results) == 1  # Only first game completed
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gui/test_match_engine.py::test_play_game_stops_on_event tests/test_gui/test_match_engine.py::test_run_match_stops_between_games -v`
Expected: FAIL — `play_game` and `run_match` don't accept `stop_event` parameter.

**Step 3: Update `GameResult` to accept `"*"` for cancelled games**

In `src/denoisr/gui/types.py`, change the valid results set:

```python
@dataclass(frozen=True)
class GameResult:
    moves: tuple[str, ...]
    result: str
    reason: str
    engine1_color: str

    def __post_init__(self) -> None:
        valid_results = {"1-0", "0-1", "1/2-1/2", "*"}
        if self.result not in valid_results:
            raise ValueError(
                f"result must be one of {valid_results}, got {self.result!r}"
            )
        if self.engine1_color not in {"white", "black"}:
            raise ValueError(
                f"engine1_color must be 'white' or 'black', "
                f"got {self.engine1_color!r}"
            )
```

**Step 4: Add `stop_event` parameter to `play_game`**

In `src/denoisr/gui/match_engine.py`:

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
) -> GameResult:
```

At the top of the `for _ in range(max_moves)` loop, after `if board.is_game_over(): break`, add:

```python
        if stop_event is not None and stop_event.is_set():
            return GameResult(
                moves=tuple(moves),
                result="*",
                reason="stopped",
                engine1_color=engine1_color,
            )
```

At the bottom of the loop (after `on_move` callback), add the move delay that respects cancellation:

```python
        if move_delay_ms > 0:
            if stop_event is not None:
                stop_event.wait(timeout=move_delay_ms / 1000)
                if stop_event.is_set():
                    return GameResult(
                        moves=tuple(moves),
                        result="*",
                        reason="stopped",
                        engine1_color=engine1_color,
                    )
            else:
                time.sleep(move_delay_ms / 1000)
```

**Step 5: Add `stop_event` parameter to `run_match`**

```python
def run_match(
    config: MatchConfig,
    max_moves_per_game: int = 500,
    on_game_complete: Callable[[int, GameResult], None] | None = None,
    on_move: Callable[[int, chess.Board, str], None] | None = None,
    stop_event: threading.Event | None = None,
    move_delay_ms: int = 0,
) -> list[GameResult]:
```

At the top of the `for game_num in range(config.games)` loop, add:

```python
        if stop_event is not None and stop_event.is_set():
            break
```

Pass `stop_event` and `move_delay_ms` through to `play_game()`:

```python
            result = play_game(
                white=white,
                black=black,
                time_control=config.time_control,
                max_moves=max_moves_per_game,
                on_move=move_cb,
                engine1_color=e1_color,
                stop_event=stop_event,
                move_delay_ms=move_delay_ms,
            )
```

Don't forget to add `import threading` at the top of `match_engine.py`.

**Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_gui/test_match_engine.py -v`
Expected: ALL PASS (existing tests + 2 new ones).

**Step 7: Lint and type-check**

Run: `uvx ruff check src/denoisr/gui/match_engine.py src/denoisr/gui/types.py`
Run: `uv run --with mypy mypy --strict src/denoisr/gui/match_engine.py src/denoisr/gui/types.py`
Expected: No issues.

**Step 8: Commit**

```bash
git add src/denoisr/gui/match_engine.py src/denoisr/gui/types.py tests/test_gui/test_match_engine.py
git commit -m "feat: add stop_event and move_delay_ms to match engine"
```

---

### Task 2: Wire Stop button to match cancellation + startup cancellation

**Files:**
- Modify: `src/denoisr/gui/app.py`

**Context:** Now that `run_match` and `play_game` accept `stop_event`, wire it to the GUI's Stop button. Also add `_startup_cancelled` flag so pressing Stop during engine loading doesn't leave a zombie engine.

**Step 1: Add `_stop_event` and `_startup_generation` to `__init__`**

In `DenoisrApp.__init__`, add after `self._match_running = False`:

```python
        self._stop_event = threading.Event()
        self._startup_generation = 0
```

**Step 2: Update `_stop_game` to set the event**

```python
    def _stop_game(self) -> None:
        self._board_widget.set_interactive(False)
        self._match_running = False
        self._stop_event.set()
        self._startup_generation += 1
        if self._engine is not None:
            self._engine.quit()
            self._engine = None
```

**Step 3: Update `_new_game` to create fresh event and use generation counter**

In `_new_game`, right after `self._stop_game()`:

```python
        self._stop_event = threading.Event()
```

In the `_start_engine` closure, capture the generation counter:

```python
        gen = self._startup_generation

        def _start_engine() -> None:
            try:
                engine = UCIEngine(config)
                engine.start(timeout=120.0)
                if self._startup_generation != gen:
                    engine.quit()
                    return
                self._move_queue.put(("engine_ready", engine))
            except Exception as e:
                if self._startup_generation != gen:
                    return
                self._move_queue.put(f"error:Engine failed: {e}")
```

**Step 4: Update `_start_match` to pass `stop_event` and `move_delay_ms`**

In the `run_in_bg` closure:

```python
        delay = self._delay_var.get()
        stop = self._stop_event

        def run_in_bg() -> None:
            try:
                run_match(
                    config,
                    on_game_complete=on_game_complete,
                    on_move=on_move,
                    stop_event=stop,
                    move_delay_ms=delay,
                )
                self._move_queue.put(("match_done",))
            except Exception as e:
                self._move_queue.put(f"error:Match failed: {e}")
```

**Step 5: Run all GUI tests**

Run: `uv run pytest tests/test_gui/ -v`
Expected: ALL PASS.

**Step 6: Lint and type-check**

Run: `uvx ruff check src/denoisr/gui/app.py`
Run: `uv run --with mypy mypy --strict src/denoisr/gui/app.py`
Expected: No issues.

**Step 7: Commit**

```bash
git add src/denoisr/gui/app.py
git commit -m "feat: wire Stop button to match cancellation and startup guard"
```

---

### Task 3: Replace status and match stats with scrollable `tk.Text` widgets

**Files:**
- Modify: `src/denoisr/gui/app.py`

**Context:** The status `ttk.Entry` truncates long error messages. The match stats `ttk.Label` is not copiable. Replace both with read-only `tk.Text` widgets.

**Step 1: Replace status widget**

In `_build_ui`, replace:

```python
        self._status_var = tk.StringVar(value="Set checkpoint, then New Game")
        status_entry = ttk.Entry(
            status_frame, textvariable=self._status_var, state="readonly",
        )
        status_entry.pack(anchor="w", fill=tk.X)
```

With:

```python
        self._status_text = tk.Text(
            status_frame, height=3, wrap=tk.WORD, state=tk.DISABLED,
            relief=tk.FLAT, bg=status_frame.cget("background"),
        )
        self._status_text.pack(anchor="w", fill=tk.X)
        self._set_status("Set checkpoint, then New Game")
```

**Step 2: Add `_set_status` helper method**

```python
    def _set_status(self, text: str) -> None:
        self._status_text.config(state=tk.NORMAL)
        self._status_text.delete("1.0", tk.END)
        self._status_text.insert("1.0", text)
        self._status_text.config(state=tk.DISABLED)
```

**Step 3: Replace all `self._status_var.set(...)` calls with `self._set_status(...)`**

Search and replace throughout `app.py`. Remove `self._status_var` entirely.

Affected locations:
- `_new_game`: "Set a checkpoint first", "Starting engine..."
- `_on_engine_ready`: "{turn} to move"
- `_on_human_move`: "Engine thinking..."
- `_apply_engine_move`: "{turn} to move"
- `_show_game_over`: "Game over: {result}"
- `_poll_queue`: `item` (error string), "Match complete"
- `_start_match`: "Set a checkpoint first", "Match running..."

**Step 4: Replace match stats widget**

In `_build_ui`, replace:

```python
        self._match_stats_var = tk.StringVar(value="")
        ttk.Label(
            self._match_frame, textvariable=self._match_stats_var,
            wraplength=200,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
```

With:

```python
        self._match_stats_text = tk.Text(
            self._match_frame, height=5, wrap=tk.WORD, state=tk.DISABLED,
            width=28, relief=tk.FLAT,
            bg=self._match_frame.cget("background"),
        )
        self._match_stats_text.grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0),
        )
```

**Step 5: Add `_set_match_stats` helper and update caller**

```python
    def _set_match_stats(self, text: str) -> None:
        self._match_stats_text.config(state=tk.NORMAL)
        self._match_stats_text.delete("1.0", tk.END)
        self._match_stats_text.insert("1.0", text)
        self._match_stats_text.config(state=tk.DISABLED)
```

In `_poll_queue`, replace `self._match_stats_var.set(item[1])` with `self._set_match_stats(item[1])`.

**Step 6: Handle error display in `_poll_queue`**

The error strings start with `"error:"`. Strip the prefix when displaying:

```python
                elif isinstance(item, str) and item.startswith("error:"):
                    self._set_status(item[6:])  # Strip "error:" prefix
```

**Step 7: Run tests, lint, type-check**

Run: `uv run pytest tests/test_gui/ -v`
Run: `uvx ruff check src/denoisr/gui/app.py`
Run: `uv run --with mypy mypy --strict src/denoisr/gui/app.py`
Expected: ALL PASS, no issues.

**Step 8: Commit**

```bash
git add src/denoisr/gui/app.py
git commit -m "feat: replace status and stats with scrollable tk.Text widgets"
```

---

### Task 4: Disable controls during active game/match + auto-start from CLI

**Files:**
- Modify: `src/denoisr/gui/app.py`
- Modify: `src/denoisr/scripts/gui.py`

**Context:** Users can change checkpoint/mode/color mid-game, causing confusing state. Greying out controls signals an active game. Auto-start eliminates the "Set checkpoint" step when `--checkpoint` is passed from CLI.

**Step 1: Store references to widgets that need disabling**

In `_build_ui`, save references (some are already saved via `textvariable`, others need explicit tracking):

```python
        self._ckpt_entry = ttk.Entry(ckpt_frame, textvariable=self._ckpt_var, width=20)
        self._ckpt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._browse_btn = ttk.Button(ckpt_frame, text="Browse", command=self._browse_ckpt)
        self._browse_btn.pack(side=tk.LEFT, padx=(4, 0))
```

```python
        self._engine_mode_combo = ttk.Combobox(...)
        self._color_combo = ttk.Combobox(...)
        self._play_radio = ttk.Radiobutton(ctrl, text="Play", ...)
        self._match_radio = ttk.Radiobutton(ctrl, text="Match", ...)
        self._new_game_btn = ttk.Button(btn_frame, text="New Game", ...)
```

**Step 2: Add `_set_controls_enabled` helper**

```python
    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        readonly_state = "readonly" if enabled else "disabled"
        for w in (
            self._ckpt_entry,
            self._browse_btn,
            self._play_radio,
            self._match_radio,
            self._new_game_btn,
        ):
            w.config(state=state)
        for w in (self._engine_mode_combo, self._color_combo):
            w.config(state=readonly_state)
```

**Step 3: Call `_set_controls_enabled(False)` when game starts**

In `_new_game`, right after `self._status_var` is set to "Starting engine..." (or equivalent), add:

```python
        self._set_controls_enabled(False)
```

In `_start_match`, after setting "Match running...":

```python
        self._set_controls_enabled(False)
```

**Step 4: Call `_set_controls_enabled(True)` when game ends**

In `_stop_game`:

```python
        self._set_controls_enabled(True)
```

In `_show_game_over`:

```python
        self._set_controls_enabled(True)
```

In `_poll_queue`, in the error handler and "match_done" handler:

```python
                    elif kind == "match_done":
                        self._set_status("Match complete")
                        self._match_running = False
                        self._set_controls_enabled(True)
                elif isinstance(item, str) and item.startswith("error:"):
                    self._set_status(item[6:])
                    self._set_controls_enabled(True)
```

**Step 5: Add auto-start from CLI**

In `src/denoisr/scripts/gui.py`, add an `auto_start` method to `DenoisrApp` or use `after`:

```python
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
        app.auto_start()
    else:
        app._engine_mode_var.set(args.mode)
    app.run()
```

In `app.py`, add `auto_start` method:

```python
    def auto_start(self) -> None:
        """Schedule a New Game after the main loop starts."""
        self._root.after(100, self._new_game)
```

**Step 6: Run all tests, lint, type-check**

Run: `uv run pytest tests/test_gui/ -v`
Run: `uvx ruff check src/denoisr/gui/app.py src/denoisr/scripts/gui.py`
Run: `uv run --with mypy mypy --strict src/denoisr/gui/app.py src/denoisr/scripts/gui.py`
Expected: ALL PASS, no issues.

**Step 7: Commit**

```bash
git add src/denoisr/gui/app.py src/denoisr/scripts/gui.py
git commit -m "feat: disable controls during game, auto-start from CLI"
```

---

### Task 5: Final verification and cleanup

**Files:**
- All GUI files

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: ALL PASS.

**Step 2: Full lint + type-check**

Run: `uvx ruff check src/denoisr/gui/ src/denoisr/scripts/gui.py`
Run: `uv run --with mypy mypy --strict src/denoisr/gui/ src/denoisr/scripts/gui.py`
Expected: No issues.

**Step 3: Manual smoke test**

Run: `uv run denoisr-gui --checkpoint outputs/random_model.pt --mode single`

Verify:
- Game auto-starts (engine loads, board becomes interactive)
- Human can make moves, engine responds
- Stop button stops the game, re-enables controls
- Switch to Match mode, start match, verify Stop kills it mid-match
- Move delay slider visibly slows match replay
- Status area shows errors scrollably
- Match stats are selectable/copiable

**Step 4: Commit if any final adjustments needed**

```bash
git add -A
git commit -m "fix: final GUI polish adjustments"
```
