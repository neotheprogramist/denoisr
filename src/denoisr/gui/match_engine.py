"""Headless UCI match orchestration — plays games between two engines."""

from __future__ import annotations

import threading
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
    stop_event: threading.Event | None = None,
    move_delay_ms: int = 0,
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

        if stop_event is not None and stop_event.is_set():
            return GameResult(
                moves=tuple(moves),
                result="*",
                reason="stopped",
                engine1_color=engine1_color,
            )

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
    stop_event: threading.Event | None = None,
    move_delay_ms: int = 0,
) -> list[GameResult]:
    """Run a multi-game match between two engines.

    Engines alternate colors each game. Returns all game results.
    """
    results: list[GameResult] = []

    for game_num in range(config.games):
        if stop_event is not None and stop_event.is_set():
            break

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
                    on_move(_gn, board, uci)

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
