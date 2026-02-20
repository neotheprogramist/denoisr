# GUI UX Polish — Design Document

## Goal

Fix broken Stop button, add match/engine cancellation, and polish the GUI for a smooth development-testing workflow.

## Changes

### Tier 1: Bug Fixes

**A. Match cancellation via `threading.Event`**

- Add `stop_event: threading.Event | None = None` parameter to `run_match()` and `play_game()`.
- `play_game` checks `stop_event.is_set()` before each move; returns early with a `"*"` (incomplete) result.
- `run_match` checks `stop_event.is_set()` between games; breaks the loop.
- `_stop_game()` in `app.py` sets the event. The `with UCIEngine(...)` context managers handle subprocess cleanup.
- Store `self._stop_event` on `DenoisrApp`; create a fresh one in `_start_match()`.

**B. Play-mode engine startup cancellation**

- Add `self._startup_cancelled` boolean flag.
- `_start_engine()` closure checks the flag after `engine.start()` completes — if set, calls `engine.quit()` instead of posting `engine_ready`.
- `_stop_game()` sets the flag to cancel any in-flight startup.

**C. Use the move delay slider**

- Pass `move_delay_ms` from `self._delay_var.get()` through `_start_match` → `run_match` → `play_game`.
- `play_game` sleeps `time.sleep(delay_ms / 1000)` after each move (respects `stop_event` during the sleep by using `stop_event.wait(timeout=delay_ms/1000)` instead of `time.sleep`).

### Tier 2: Core UX

**D. Scrollable status display**

- Replace `ttk.Entry` for status with `tk.Text(height=3, wrap=tk.WORD, state=tk.DISABLED)`.
- Helper method `_set_status(text)` toggles NORMAL, replaces content, toggles DISABLED.
- Drop `self._status_var` — use the Text widget directly.

**E. Copiable match stats**

- Replace `ttk.Label` for match stats with `tk.Text(height=5, wrap=tk.WORD, state=tk.DISABLED)`.
- Helper method `_set_match_stats(text)` same pattern.
- Drop `self._match_stats_var`.

**F. Disable controls during active game/match**

- Helper `_set_controls_enabled(enabled: bool)` toggles state of: checkpoint entry, browse button, engine mode combobox, color combobox, mode radio buttons, match settings widgets.
- Called with `False` in `_new_game()` / `_start_match()`, `True` in `_stop_game()` and on game completion.
- Also serves as the "thinking indicator" — user sees controls are greyed out while engine is active.

### Tier 3: Polish

**G. Auto-start from CLI**

- In `gui.py`, after setting checkpoint/mode vars, if checkpoint is provided call `root.after(100, app._new_game)`.
- Requires exposing `_root` or adding an `auto_start()` method on `DenoisrApp`.

## Files Modified

- `src/denoisr/gui/app.py` — main changes (cancellation, UI widgets, control toggling)
- `src/denoisr/gui/match_engine.py` — add `stop_event` + `move_delay_ms` params
- `src/denoisr/gui/types.py` — add `"*"` to valid GameResult results (for cancelled games)
- `src/denoisr/scripts/gui.py` — auto-start logic
- `tests/test_gui/test_match_engine.py` — test stop_event cancellation

## Testing

- Unit test: `run_match` with `stop_event` set after N games stops early
- Unit test: `play_game` with `stop_event` set mid-game returns incomplete result
- Manual: verify Stop button kills running match, verify move delay works, verify scrollable status
