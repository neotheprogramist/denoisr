"""PGN export for benchmark games."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import chess
import chess.pgn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from denoisr.evaluation.benchmark import CompletedGame


def _build_pgn_game(game: CompletedGame, event: str = "Denoisr Benchmark") -> chess.pgn.Game:
    """Convert a CompletedGame to a chess.pgn.Game with proper headers."""
    pgn_game = chess.pgn.Game()

    pgn_game.headers["Event"] = event
    pgn_game.headers["Site"] = "Local"
    pgn_game.headers["Date"] = datetime.now(tz=timezone.utc).strftime("%Y.%m.%d")
    pgn_game.headers["Round"] = str(game.game_num + 1)

    if game.engine1_color == "white":
        pgn_game.headers["White"] = "Denoisr"
        pgn_game.headers["Black"] = "Opponent"
    else:
        pgn_game.headers["White"] = "Opponent"
        pgn_game.headers["Black"] = "Denoisr"

    pgn_game.headers["Result"] = game.result

    if game.start_fen is not None:
        pgn_game.headers["FEN"] = game.start_fen
        pgn_game.headers["SetUp"] = "1"

    board = chess.Board(game.start_fen) if game.start_fen else chess.Board()
    node: chess.pgn.Game | chess.pgn.ChildNode = pgn_game
    for uci_move in game.moves:
        move = board.parse_uci(uci_move)
        node = node.add_variation(move)
        board.push(move)

    return pgn_game


def write_pgn(game: CompletedGame, directory: Path, event: str = "Denoisr Benchmark") -> Path:
    """Write a single game as a PGN file. Returns the path written."""
    directory.mkdir(parents=True, exist_ok=True)
    pgn_game = _build_pgn_game(game, event=event)
    path = directory / f"game_{game.game_num:04d}.pgn"
    path.write_text(str(pgn_game) + "\n")
    return path


def write_combined_pgn(
    games: Sequence[CompletedGame],
    path: Path,
    event: str = "Denoisr Benchmark",
) -> Path:
    """Write all games into a single PGN file. Returns the path written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    for game in games:
        pgn_game = _build_pgn_game(game, event=event)
        print(pgn_game, file=buf)
        print(file=buf)
    path.write_text(buf.getvalue())
    return path
