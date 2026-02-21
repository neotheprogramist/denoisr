"""Stockfish ACPL analysis and Elo estimation for benchmark games.

Replays each game, evaluates every position with Stockfish at a fixed depth,
and computes Average Centipawn Loss (ACPL) per game. ACPL is then mapped to
an approximate Elo rating.
"""

from __future__ import annotations

import atexit
import multiprocessing
from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess
import chess.engine

if TYPE_CHECKING:
    from collections.abc import Sequence

    from denoisr.evaluation.benchmark import CompletedGame

# ---------------------------------------------------------------------------
# Per-worker process globals
# ---------------------------------------------------------------------------

_sf: chess.engine.SimpleEngine | None = None


def _cleanup_sf() -> None:
    global _sf
    if _sf is not None:
        try:
            _sf.quit()
        except Exception:  # noqa: BLE001
            pass
        _sf = None


def _init_analysis_worker(stockfish_cmd: str) -> None:
    global _sf
    _sf = chess.engine.SimpleEngine.popen_uci(stockfish_cmd)
    atexit.register(_cleanup_sf)


# ---------------------------------------------------------------------------
# Analysis task
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AnalysisTask:
    game_num: int
    moves: tuple[str, ...]
    start_fen: str | None
    engine1_color: str
    depth: int


@dataclass(frozen=True)
class GameAnalysis:
    game_num: int
    acpl: float
    move_losses: tuple[float, ...]
    blunders: int
    engine1_color: str
    num_moves: int


def _analyze_one_game(task: _AnalysisTask) -> tuple[int, float, tuple[float, ...], int, str, int]:
    """Analyze a single game in a worker process.

    Returns (game_num, acpl, move_losses, blunders, engine1_color, num_moves).
    """
    if _sf is None:
        raise RuntimeError("Analysis worker not initialized")

    board = chess.Board(task.start_fen) if task.start_fen else chess.Board()
    limit = chess.engine.Limit(depth=task.depth)

    # Evaluate initial position
    prev_info = _sf.analyse(board, limit)
    prev_score = prev_info["score"].white()

    losses: list[float] = []
    blunders = 0

    for uci_move in task.moves:
        moving_side = board.turn
        board.push_uci(uci_move)
        curr_info = _sf.analyse(board, limit)
        curr_score = curr_info["score"].white()

        # Compute centipawn loss from the moving side's perspective.
        # A positive loss means the move was worse than optimal.
        prev_cp = _score_to_cp(prev_score)
        curr_cp = _score_to_cp(curr_score)

        if moving_side == chess.WHITE:
            loss = prev_cp - curr_cp
        else:
            loss = curr_cp - prev_cp

        # Clamp: a negative loss means the opponent blundered and we
        # improved — not a loss for the moving side.
        loss = max(0.0, loss)
        losses.append(loss)

        if loss >= 200:
            blunders += 1

        prev_score = curr_score

    # Filter to only engine1's moves for ACPL
    is_white = task.engine1_color == "white"
    engine_losses = [
        loss
        for i, loss in enumerate(losses)
        if (i % 2 == 0) == is_white
    ]
    acpl = sum(engine_losses) / max(len(engine_losses), 1)

    return (
        task.game_num,
        acpl,
        tuple(losses),
        blunders,
        task.engine1_color,
        len(task.moves),
    )


def _score_to_cp(score: chess.engine.Score) -> float:
    """Convert a Score to centipawns, capping mates at +/-10000."""
    cp = score.score(mate_score=10000)
    if cp is None:
        return 0.0
    return float(cp)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ACPL_ELO_SLOPE = 33.0
ACPL_ELO_INTERCEPT = 3300.0
ELO_MIN = 400.0
ELO_MAX = 3300.0


def acpl_to_elo(acpl: float) -> float:
    """Estimate Elo from ACPL using a linear approximation.

    Elo ≈ 3300 - 33 * ACPL, clamped to [400, 3300].
    """
    elo = ACPL_ELO_INTERCEPT - ACPL_ELO_SLOPE * acpl
    return max(ELO_MIN, min(ELO_MAX, elo))


@dataclass(frozen=True)
class AnalysisResult:
    game_analyses: tuple[GameAnalysis, ...]
    overall_acpl: float
    total_blunders: int
    estimated_elo: float


def run_analysis(
    games: Sequence[CompletedGame],
    stockfish_cmd: str = "stockfish",
    depth: int = 12,
    concurrency: int = 4,
) -> AnalysisResult:
    """Run Stockfish ACPL analysis on a sequence of completed games."""
    tasks = [
        _AnalysisTask(
            game_num=g.game_num,
            moves=g.moves,
            start_fen=g.start_fen,
            engine1_color=g.engine1_color,
            depth=depth,
        )
        for g in games
        if len(g.moves) > 0
    ]

    if not tasks:
        return AnalysisResult(
            game_analyses=(),
            overall_acpl=0.0,
            total_blunders=0,
            estimated_elo=acpl_to_elo(0.0),
        )

    analyses: list[GameAnalysis] = []

    with multiprocessing.Pool(
        min(concurrency, len(tasks)),
        initializer=_init_analysis_worker,
        initargs=(stockfish_cmd,),
    ) as pool:
        for game_num, acpl, move_losses, blunders, e1_color, num_moves in (
            pool.imap_unordered(_analyze_one_game, tasks)
        ):
            analyses.append(
                GameAnalysis(
                    game_num=game_num,
                    acpl=acpl,
                    move_losses=move_losses,
                    blunders=blunders,
                    engine1_color=e1_color,
                    num_moves=num_moves,
                )
            )

    analyses.sort(key=lambda a: a.game_num)

    # Weighted average ACPL (by number of engine moves per game)
    total_loss = 0.0
    total_moves = 0
    total_blunders = 0
    for a in analyses:
        is_white = a.engine1_color == "white"
        engine_move_count = sum(
            1 for i in range(a.num_moves) if (i % 2 == 0) == is_white
        )
        total_loss += a.acpl * engine_move_count
        total_moves += engine_move_count
        total_blunders += a.blunders

    overall_acpl = total_loss / max(total_moves, 1)

    return AnalysisResult(
        game_analyses=tuple(analyses),
        overall_acpl=overall_acpl,
        total_blunders=total_blunders,
        estimated_elo=acpl_to_elo(overall_acpl),
    )
