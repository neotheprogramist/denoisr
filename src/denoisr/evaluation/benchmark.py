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
    picklable tuple instead of full GameResult.
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
        tasks.append(
            _GameTask(
                game_num=i, start_fen=fen, engine_is_white=engine_is_white
            )
        )

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
            engine_won = (result_str == "1-0" and e1_color == "white") or (
                result_str == "0-1" and e1_color == "black"
            )
            engine_lost = (result_str == "0-1" and e1_color == "white") or (
                result_str == "1-0" and e1_color == "black"
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
            if (
                config.sprt_elo0 is not None
                and config.sprt_elo1 is not None
            ):
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
