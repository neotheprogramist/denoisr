from pathlib import Path

import pytest

from denoisr_chess.data.pgn_streamer import SimplePGNStreamer

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures"


class TestSimplePGNStreamer:
    @pytest.fixture
    def streamer(self) -> SimplePGNStreamer:
        return SimplePGNStreamer()

    def test_streams_correct_count(self, streamer: SimplePGNStreamer) -> None:
        games = list(streamer.stream(FIXTURES / "sample_games.pgn"))
        assert len(games) == 3

    def test_scholars_mate(self, streamer: SimplePGNStreamer) -> None:
        games = list(streamer.stream(FIXTURES / "sample_games.pgn"))
        game = games[0]
        assert game.result == 1.0
        assert len(game.actions) == 7  # 4 white + 3 black moves

    def test_fools_mate(self, streamer: SimplePGNStreamer) -> None:
        games = list(streamer.stream(FIXTURES / "sample_games.pgn"))
        game = games[1]
        assert game.result == -1.0
        assert len(game.actions) == 4

    def test_draw(self, streamer: SimplePGNStreamer) -> None:
        games = list(streamer.stream(FIXTURES / "sample_games.pgn"))
        game = games[2]
        assert game.result == 0.0

    def test_actions_are_valid(self, streamer: SimplePGNStreamer) -> None:
        games = list(streamer.stream(FIXTURES / "sample_games.pgn"))
        for game in games:
            for action in game.actions:
                assert 0 <= action.from_square < 64
                assert 0 <= action.to_square < 64

    def test_empty_file(self, streamer: SimplePGNStreamer, tmp_path: Path) -> None:
        empty = tmp_path / "empty.pgn"
        empty.write_text("")
        games = list(streamer.stream(empty))
        assert len(games) == 0

    def test_elo_extraction(self, streamer: SimplePGNStreamer, tmp_path: Path) -> None:
        pgn = tmp_path / "elo.pgn"
        pgn.write_text(
            '[Event "Test"]\n'
            '[White "Alice"]\n'
            '[Black "Bob"]\n'
            '[Result "1-0"]\n'
            '[WhiteElo "1500"]\n'
            '[BlackElo "1800"]\n'
            "\n"
            "1. e4 e5 1-0\n\n"
        )
        games = list(streamer.stream(pgn))
        assert len(games) == 1
        assert games[0].white_elo == 1500
        assert games[0].black_elo == 1800

    def test_missing_elo_returns_none(
        self, streamer: SimplePGNStreamer, tmp_path: Path
    ) -> None:
        pgn = tmp_path / "no_elo.pgn"
        pgn.write_text('[Event "Test"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n\n')
        games = list(streamer.stream(pgn))
        assert games[0].white_elo is None
        assert games[0].black_elo is None

    def test_question_mark_elo_returns_none(
        self, streamer: SimplePGNStreamer, tmp_path: Path
    ) -> None:
        pgn = tmp_path / "q_elo.pgn"
        pgn.write_text(
            '[Event "Test"]\n'
            '[Result "1-0"]\n'
            '[WhiteElo "?"]\n'
            '[BlackElo "?"]\n'
            "\n"
            "1. e4 e5 1-0\n\n"
        )
        games = list(streamer.stream(pgn))
        assert games[0].white_elo is None
        assert games[0].black_elo is None
