"""Tests for PGN writer — no Stockfish required."""

from pathlib import Path

import chess.pgn

from denoisr.evaluation.benchmark import CompletedGame
from denoisr.evaluation.pgn_writer import write_combined_pgn, write_pgn


def _sample_game(
    game_num: int = 0,
    engine1_color: str = "white",
    start_fen: str | None = None,
) -> CompletedGame:
    """Scholar's mate: 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6 4.Qxf7#"""
    return CompletedGame(
        game_num=game_num,
        result="1-0",
        engine1_color=engine1_color,
        moves=("e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"),
        start_fen=start_fen,
        reason="checkmate",
    )


class TestWritePgn:
    def test_creates_file(self, tmp_path: Path) -> None:
        game = _sample_game()
        path = write_pgn(game, tmp_path)
        assert path.exists()
        assert path.name == "game_0000.pgn"

    def test_pgn_parseable(self, tmp_path: Path) -> None:
        game = _sample_game()
        path = write_pgn(game, tmp_path)
        with path.open() as f:
            parsed = chess.pgn.read_game(f)
        assert parsed is not None
        assert parsed.headers["Result"] == "1-0"
        assert parsed.headers["White"] == "Denoisr"
        assert parsed.headers["Black"] == "Opponent"

    def test_black_engine_headers(self, tmp_path: Path) -> None:
        game = _sample_game(engine1_color="black")
        path = write_pgn(game, tmp_path)
        with path.open() as f:
            parsed = chess.pgn.read_game(f)
        assert parsed is not None
        assert parsed.headers["White"] == "Opponent"
        assert parsed.headers["Black"] == "Denoisr"

    def test_custom_fen_header(self, tmp_path: Path) -> None:
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        game = CompletedGame(
            game_num=0,
            result="1/2-1/2",
            engine1_color="white",
            moves=("e7e5",),
            start_fen=fen,
            reason="unknown",
        )
        path = write_pgn(game, tmp_path)
        with path.open() as f:
            parsed = chess.pgn.read_game(f)
        assert parsed is not None
        assert parsed.headers["FEN"] == fen
        assert parsed.headers["SetUp"] == "1"

    def test_move_count_matches(self, tmp_path: Path) -> None:
        game = _sample_game()
        path = write_pgn(game, tmp_path)
        with path.open() as f:
            parsed = chess.pgn.read_game(f)
        assert parsed is not None
        moves = list(parsed.mainline_moves())
        assert len(moves) == 7

    def test_game_numbering(self, tmp_path: Path) -> None:
        game = _sample_game(game_num=42)
        path = write_pgn(game, tmp_path)
        assert path.name == "game_0042.pgn"


class TestWriteCombinedPgn:
    def test_multiple_games(self, tmp_path: Path) -> None:
        games = [_sample_game(game_num=i) for i in range(3)]
        path = tmp_path / "all.pgn"
        write_combined_pgn(games, path)
        assert path.exists()

        parsed_games = []
        with path.open() as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                parsed_games.append(g)
        assert len(parsed_games) == 3

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "all.pgn"
        games = [_sample_game()]
        write_combined_pgn(games, path)
        assert path.exists()
