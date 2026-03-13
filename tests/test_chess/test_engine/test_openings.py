from pathlib import Path

import chess

from denoisr_chess.engine.openings import load_openings


class TestLoadOpenings:
    def test_loads_fen_lines(self, tmp_path: Path) -> None:
        epd = tmp_path / "openings.epd"
        epd.write_text(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\n"
            "rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2\n"
        )
        fens = load_openings(epd)
        assert len(fens) == 2
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
